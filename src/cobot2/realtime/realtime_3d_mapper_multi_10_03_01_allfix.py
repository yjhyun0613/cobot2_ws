import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
import sensor_msgs_py.point_cloud2 as pc2
from tf2_ros import Buffer, TransformListener, StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
import numpy as np
from scipy.spatial.transform import Rotation as R
import struct
import cv2
import math
from cv_bridge import CvBridge
from ultralytics import YOLO
import os
import json
import time
from datetime import datetime
from pathlib import Path
import sys

# 🌟 ROS 2 표준 서비스 (통신용)
from std_srvs.srv import Trigger

try:
    from message_filters import Subscriber, ApproximateTimeSynchronizer
except Exception:
    Subscriber = None
    ApproximateTimeSynchronizer = None

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from common import settings
from common import db_paths
from common.firebase_client import get_db_reference, get_storage_bucket

class VisionServerNode(Node):
    def __init__(self):
        super().__init__('vision_server_node')

        # TF 설정
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        self.publish_static_transform()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # YOLO 로딩
        self.bridge = CvBridge()
        model_path = str(ROOT_DIR / 'YJH' / 'resource' / 'hyupdong2_yolo11x_realtest_corrected_best.pt')
        self.get_logger().info(f'YOLO 모델 로딩 중... ({model_path})')
        self.yolo_model = YOLO(model_path)
        self.get_logger().info('✅ YOLO 모델 로딩 완료!')

        # 세션 관리
        self.session_id = None
        self.session_started_at_ms = None
        self.session_total_captures = 0
        self.session_total_markers = 0
        self.session_normal_count = 0
        self.session_defect_count = 0

        # Firebase 초기화
        try:
            self.bucket = get_storage_bucket()
            self.inspections_ref = get_db_reference(db_paths.inspections_path())
            self.twin_state_ref = get_db_reference(db_paths.twin_state_path())
            self.site_ref = get_db_reference(db_paths.site_path())
            self.robot_ref = get_db_reference(db_paths.robot_path())
            self.get_logger().info('✅ Firebase Storage & Realtime Database 연결 완료!')
            self.init_base_database_nodes()
            self.init_capture_count()
        except Exception as e:
            self.get_logger().error(f'❌ Firebase 초기화 실패: {e}')
            self.bucket = None
            self.inspections_ref = None
            self.twin_state_ref = None
            self.site_ref = None
            self.robot_ref = None
            self.capture_count = 0

        # 데이터 구독
        self.latest_pc_msg = None
        self.latest_img_msg = None
        self.latest_depth_msg = None
        self.cam_info = None
        self.latest_sync_stamp_ms = None

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.create_subscription(
            PointCloud2,
            '/camera/camera/depth/color/points',
            self.pc_callback,
            sensor_qos
        )

        self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.img_callback,
            sensor_qos
        )

        self.create_subscription(
            Image,
            '/camera/camera/aligned_depth_to_color/image_raw',
            self.depth_callback,
            sensor_qos
        )

        self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self.info_callback,
            sensor_qos
        )

        self.get_logger().info('✅ RealSense RGB/Depth/PointCloud/CameraInfo SensorDataQoS 구독 활성화')

        # 🌟 로봇의 검사 요청을 기다리는 서비스 서버 생성
        self.srv = self.create_service(Trigger, '/vision_inspect', self.inspect_callback)
        self.get_logger().info('🤖 비전 서버 대기 중... 로봇의 이동 완료 신호를 기다립니다.')

    def init_capture_count(self):
        self.capture_count = 0
        self.get_logger().info('🗂️ 새 검사 세션 기준으로 작업대 번호를 1부터 시작합니다.')

    def ensure_session_started(self, now_ms):
        if self.session_id:
            return
        self.session_id = db_paths.now_session_id()
        self.session_started_at_ms = now_ms
        self.get_logger().info(f'🗂️ 검사 세션 시작: {self.session_id}')

        # 🔄 [초기화] 실시간 통신용 live_scan 노드 초기화 (이전 작업 데이터 자동 삭제)
        live_scan_ref = get_db_reference('live_scan')
        if live_scan_ref:
            live_scan_ref.set({
                'session_id': self.session_id,
                'started_at': now_ms,
                'workstations': {}
            })

        if self.site_ref:
            self.site_ref.update({'latest_session_id': self.session_id, 'updated_at': now_ms})
        if self.robot_ref:
            self.robot_ref.update({'current_session_id': self.session_id, 'updated_at': now_ms})
        if self.twin_state_ref:
            self.twin_state_ref.child('current_session').set({
                'session_id': self.session_id, 'status': 'running',
                'site_id': settings.SITE_ID, 'robot_id': settings.ROBOT_ID,
                'started_at': now_ms, 'updated_at': now_ms,
            })
        session_ref = get_db_reference(db_paths.inspection_session_path(self.session_id))
        session_ref.child('metadata').set({
            'session_id': self.session_id, 'company_id': settings.COMPANY_ID,
            'company_name': settings.COMPANY_NAME, 'site_id': settings.SITE_ID,
            'site_name': settings.SITE_NAME, 'robot_id': settings.ROBOT_ID,
            'status': 'running', 'started_at': now_ms, 'updated_at': now_ms,
            'schema_version': '1.0.0',
        })
        session_ref.child('summary').set({
            'total_workstations': 0, 'total_captures': 0, 'total_markers': 0,
            'normal_count': 0, 'defect_count': 0, 'resolved_count': 0,
            'failed_count': 0, 'updated_at': now_ms,
        })

    def init_base_database_nodes(self):
        now_ms = int(time.time() * 1000)
        get_db_reference(db_paths.system_path()).update({
            'project_name': settings.COMPANY_NAME, 'schema_version': '1.0.0', 'updated_at': now_ms,
        })
        get_db_reference(db_paths.company_path()).update({
            'name': settings.COMPANY_NAME, 'created_at': now_ms, 'schema_version': '1.0.0',
        })
        self.site_ref.update({
            'company_id': settings.COMPANY_ID, 'site_name': settings.SITE_NAME,
            'active_robot_id': settings.ROBOT_ID, 'updated_at': now_ms,
        })
        self.robot_ref.update({
            'company_id': settings.COMPANY_ID, 'site_id': settings.SITE_ID,
            'robot_name': settings.ROBOT_NAME, 'robot_model': 'Doosan M0609',
            'gripper': 'OnRobot RG2', 'status': 'ready', 'updated_at': now_ms,
        })
        get_db_reference(db_paths.twin_static_path()).update({
            'company_id': settings.COMPANY_ID, 'site_id': settings.SITE_ID,
            'robots': {settings.ROBOT_ID: {
                'robot_name': settings.ROBOT_NAME, 'robot_model': 'Doosan M0609',
                'gripper': 'OnRobot RG2', 'base_frame': settings.BASE_FRAME,
                'tool_frame': 'tool0', 'camera_frame': settings.CAMERA_FRAME,
            }},
            'cameras': {settings.CAMERA_ID: {
                'camera_name': settings.CAMERA_NAME, 'type': 'depth_camera',
                'frame_id': settings.CAMERA_FRAME, 'parent_frame': 'tool0',
            }},
            'workstations': {settings.WORKSTATION_ID: {
                'display_name': settings.WORKSTATION_NAME, 'line_id': settings.LINE_ID,
                'base_frame': settings.BASE_FRAME,
            }},
            'coordinate_frames': {
                settings.BASE_FRAME: {'type': 'root'},
                settings.CAMERA_FRAME: {'parent': 'tool0'},
            },
            'updated_at': now_ms,
        })

    def upload_to_firebase_storage(self, file_path, destination_blob_name, content_type):
        if not self.bucket:
            return None
        try:
            blob = self.bucket.blob(destination_blob_name)
            blob.upload_from_filename(file_path, content_type=content_type)
            blob.make_public()
            return blob.public_url
        except Exception as e:
            self.get_logger().error(f'❌ Storage 업로드 실패: {e}')
            return None

    def build_marker_data(self, screw_key, status_bool, delta_z_mm, confidence, bbox, pt_base_mm, pose_angles, time_ms, protection_radius_m=0.04):
        status_text = 'normal' if status_bool else 'defect'
        defect_type = 'none' if status_bool else 'height_defect'
        rx, ry, rz = pose_angles if pose_angles else (0.0, 0.0, 0.0)
        return {
            'screw_id': screw_key, 'status': status_text,
            'status_bool_legacy': status_bool, 'defect_type': defect_type,
            'confidence': float(confidence), 'delta_z_mm': float(delta_z_mm),
            'position': {
                'x': float(pt_base_mm[0]), 'y': float(pt_base_mm[1]),
                'z': float(pt_base_mm[2]), 'frame_id': settings.BASE_FRAME,
            },
            'orientation': {
                'rx': float(rx), 'ry': float(ry), 'rz': float(rz)
            },
            'bbox': {'x1': int(bbox[0]), 'y1': int(bbox[1]), 'x2': int(bbox[2]), 'y2': int(bbox[3])},
            'protection_radius_m': float(protection_radius_m),
            'updated_at': time_ms,
        }

    def publish_static_transform(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'link_6'
        t.child_frame_id = settings.CAMERA_FRAME
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0767196
        t.transform.translation.z = 0.03991
        t.transform.rotation.x = -0.5
        t.transform.rotation.y = -0.5
        t.transform.rotation.z = -0.5
        t.transform.rotation.w = 0.5
        self.tf_static_broadcaster.sendTransform(t)

    def pc_callback(self, msg): self.latest_pc_msg = msg
    def img_callback(self, msg): self.latest_img_msg = msg
    def depth_callback(self, msg): self.latest_depth_msg = msg
    def info_callback(self, msg): self.cam_info = msg

    def synced_frame_callback(self, pc_msg, img_msg, depth_msg):
        self.latest_pc_msg = pc_msg
        self.latest_img_msg = img_msg
        self.latest_depth_msg = depth_msg
        stamps = [self.msg_stamp_ms(pc_msg), self.msg_stamp_ms(img_msg), self.msg_stamp_ms(depth_msg)]
        if all(v is not None for v in stamps):
            self.latest_sync_stamp_ms = max(stamps)

    def msg_stamp_ms(self, msg):
        try:
            return int(msg.header.stamp.sec * 1000 + msg.header.stamp.nanosec / 1e6)
        except Exception:
            return None

    def get_stamp_skew_ms(self, msgs):
        stamps = [self.msg_stamp_ms(m) for m in msgs if m is not None]
        if len(stamps) < 2 or any(v is None for v in stamps):
            return None
        return max(stamps) - min(stamps)

    def get_tf_matrix(self, target, source):
        try:
            trans = self.tf_buffer.lookup_transform(target, source, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0))
            rot = R.from_quat([trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w]).as_matrix()
            mat = np.eye(4); mat[:3, :3] = rot; mat[:3, 3] = [trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z]
            return mat
        except: return None

    def normalize_vec(self, v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-6 else v

    def rot_to_zyz(self, R_mat):
        beta = math.acos(max(min(R_mat[2, 2], 1.0), -1.0))
        if abs(beta) < 1e-6:
            alpha, gamma = 0.0, math.atan2(R_mat[1, 0], R_mat[0, 0])
        else:
            alpha = math.atan2(R_mat[1, 2], R_mat[0, 2])
            gamma = math.atan2(R_mat[2, 1], -R_mat[2, 0])
        return [math.degrees(alpha), math.degrees(beta), math.degrees(gamma)]

    def wrap_angle(self, angle):
        while angle > 180.0: angle -= 360.0
        while angle < -180.0: angle += 360.0
        return angle

    def calculate_target_pose(self, normal_vec):
        if np.linalg.norm(normal_vec) < 1e-6: return None
        n = self.normalize_vec(normal_vec)
        z_axis_final = -n
        global_y = np.array([0.0, 1.0, 0.0])
        x_axis = np.cross(global_y, z_axis_final)
        
        if np.linalg.norm(x_axis) < 1e-6:
            global_x = np.array([1.0, 0.0, 0.0])
            y_axis = self.normalize_vec(np.cross(z_axis_final, global_x))
            x_axis = self.normalize_vec(np.cross(y_axis, z_axis_final))
        else:
            x_axis = self.normalize_vec(x_axis)
            y_axis = self.normalize_vec(np.cross(z_axis_final, x_axis))
            
        rot_matrix = np.column_stack((x_axis, y_axis, z_axis_final))
        rx, ry, rz = self.rot_to_zyz(rot_matrix)
        return self.wrap_angle(rx), self.wrap_angle(ry), self.wrap_angle(rz)

    def get_3d_point(self, u, v, depth_map, fx, fy, cx, cy, tf_matrix):
        z = float(depth_map[v, u]) / 1000.0
        if z <= 0.1 or z > 2.0: return None
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        pt_cam = np.array([x, y, z, 1.0])
        pt_base = tf_matrix @ pt_cam
        return pt_base[:3]

    def get_surface_point(self, u, v, depth_map, fx, fy, cx, cy, tf_matrix, radius=5):
        h, w = depth_map.shape[:2]
        pts = []
        for du in range(-radius, radius + 1, 2):
            for dv in range(-radius, radius + 1, 2):
                nu, nv = u + du, v + dv
                if 0 <= nu < w and 0 <= nv < h:
                    pt = self.get_3d_point(nu, nv, depth_map, fx, fy, cx, cy, tf_matrix)
                    if pt is not None: pts.append(pt)
        return np.mean(pts, axis=0) if pts else None

    def pixel_to_3d_base(self, u, v, z_m, fx, fy, cx, cy, tf_mat):
        if z_m <= 0.1 or z_m > 2.0:
            return None
        x = (u - cx) * z_m / fx
        y = (v - cy) * z_m / fy
        return (tf_mat @ np.array([x, y, z_m, 1.0]))[:3]

    def estimate_screw_protection_radius(self, bbox, depth_img, fx, fy):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = depth_img.shape[:2]
        x1 = max(0, min(x1, w - 1)); x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1)); y2 = max(0, min(y2, h - 1))
        patch = depth_img[y1:y2+1, x1:x2+1]
        valid = patch[(patch > 100) & (patch < 2000)]
        z = float(np.median(valid)) / 1000.0 if valid.size > 0 else 0.4
        radius_x = max(0.0, (x2 - x1) * z / max(fx, 1e-6) * 0.5)
        radius_y = max(0.0, (y2 - y1) * z / max(fy, 1e-6) * 0.5)
        return float(np.clip(max(0.035, 0.75 * max(radius_x, radius_y)), 0.035, 0.065))

    def component_circularity(self, component_mask):
        contours, _ = cv2.findContours(component_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 1e-6:
            return 0.0
        return float(4.0 * math.pi * area / (perimeter * perimeter))

    def get_robust_screw_center(self, bbox, depth_img, cv_image, fx, fy, cx, cy, tf_mat, plane_point, normal):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h_img, w_img = depth_img.shape[:2]

        x1 = max(0, min(x1, w_img - 1))
        x2 = max(0, min(x2, w_img - 1))
        y1 = max(0, min(y1, h_img - 1))
        y2 = max(0, min(y2, h_img - 1))

        if x2 <= x1 or y2 <= y1:
            return None, None

        w = x2 - x1 + 1
        h = y2 - y1 + 1
        bbox_center_2d = np.array([w / 2.0, h / 2.0], dtype=np.float32)
        diag = math.sqrt(w * w + h * h)

        cx_img, cy_img = (x1 + x2) // 2, (y1 + y2) // 2
        fallback_center_pt = self.get_surface_point(cx_img, cy_img, depth_img, fx, fy, cx, cy, tf_mat, radius=3)

        depth_patch_raw = depth_img[y1:y2 + 1, x1:x2 + 1].copy()
        if depth_patch_raw.size == 0:
            return None, None

        depth_patch = cv2.medianBlur(depth_patch_raw, 3)

        fallback_median_pt = None
        valid_depths = depth_patch[(depth_patch > 100) & (depth_patch < 2000)]
        if valid_depths.size > 0:
            median_z = float(np.median(valid_depths)) / 1000.0
            fallback_median_pt = self.pixel_to_3d_base(cx_img, cy_img, median_z, fx, fy, cx, cy, tf_mat)

        bgr_patch = cv_image[y1:y2 + 1, x1:x2 + 1]
        hsv_patch = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2HSV)

        lower_black = np.array([0, 0, 0], dtype=np.uint8)
        strict_black = cv2.inRange(hsv_patch, lower_black, np.array([180, 120, 70], dtype=np.uint8))
        relaxed_black = cv2.inRange(hsv_patch, lower_black, np.array([180, 255, 90], dtype=np.uint8))

        def collect_component_points(labels, label_id, pts_3d_grid, black_mask):
            selected_pts = []
            selected_heights = []
            black_count = 0

            for vv in range(h):
                for uu in range(w):
                    if labels[vv, uu] != label_id:
                        continue

                    if (vv, uu) in pts_3d_grid:
                        pt_base, height = pts_3d_grid[(vv, uu)]
                        selected_pts.append(pt_base)
                        selected_heights.append(height)

                    if black_mask[vv, uu] > 0:
                        black_count += 1

            return selected_pts, selected_heights, black_count

        def candidate_center_from_component(strategy_name, comp_centroid_2d, selected_pts, selected_heights):
            selected_pts_np = np.asarray(selected_pts, dtype=np.float32)
            selected_heights_np = np.asarray(selected_heights, dtype=np.float32)

            u_center = int(np.clip(round(x1 + comp_centroid_2d[0]), 0, w_img - 1))
            v_center = int(np.clip(round(y1 + comp_centroid_2d[1]), 0, h_img - 1))

            if strategy_name == 'upper_strict_circularity':
                local_depths = []
                for dv in range(-2, 3):
                    for du in range(-2, 3):
                        uu = u_center + du
                        vv = v_center + dv
                        if 0 <= uu < w_img and 0 <= vv < h_img:
                            z_val = float(depth_img[vv, uu]) / 1000.0
                            if 0.1 < z_val <= 2.0:
                                local_depths.append(z_val)

                if local_depths:
                    z_center = float(np.median(local_depths))
                    centroid_3d = self.pixel_to_3d_base(u_center, v_center, z_center, fx, fy, cx, cy, tf_mat)
                    if centroid_3d is not None:
                        return centroid_3d

            return np.median(selected_pts_np, axis=0)

        def build_candidates(strategy_name, black_mask, height_min, height_max, min_area, center_limit,
                             allow_height_only, prefer_circularity):
            binary_mask = np.zeros((h, w), dtype=np.uint8)
            pts_3d_grid = {}

            for vv in range(h):
                for uu in range(w):
                    z = float(depth_patch[vv, uu]) / 1000.0
                    if z <= 0.1 or z > 2.0:
                        continue

                    u = x1 + uu
                    v = y1 + vv
                    pt_base = self.pixel_to_3d_base(u, v, z, fx, fy, cx, cy, tf_mat)
                    if pt_base is None:
                        continue

                    height = float(np.dot(pt_base - plane_point, normal))
                    if height_min <= height <= height_max and black_mask[vv, uu] > 0:
                        binary_mask[vv, uu] = 1
                        pts_3d_grid[(vv, uu)] = (pt_base, height)

            used_height_only = False
            if allow_height_only and int(np.count_nonzero(binary_mask)) < max(10, min_area):
                binary_mask = np.zeros((h, w), dtype=np.uint8)
                pts_3d_grid = {}

                for vv in range(h):
                    for uu in range(w):
                        z = float(depth_patch[vv, uu]) / 1000.0
                        if z <= 0.1 or z > 2.0:
                            continue

                        u = x1 + uu
                        v = y1 + vv
                        pt_base = self.pixel_to_3d_base(u, v, z, fx, fy, cx, cy, tf_mat)
                        if pt_base is None:
                            continue

                        height = float(np.dot(pt_base - plane_point, normal))
                        if max(height_min + 0.001, 0.004) <= height <= height_max:
                            binary_mask[vv, uu] = 1
                            pts_3d_grid[(vv, uu)] = (pt_base, height)

                used_height_only = True

            kernel = np.ones((3, 3), dtype=np.uint8)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask)
            candidates = []

            for label_id in range(1, num_labels):
                area = int(stats[label_id, cv2.CC_STAT_AREA])
                if area < min_area:
                    continue

                comp_centroid_2d = centroids[label_id].astype(np.float32)
                dist_to_center = float(np.linalg.norm(comp_centroid_2d - bbox_center_2d))
                center_ratio = dist_to_center / max(diag, 1e-6)
                if center_ratio > center_limit:
                    continue

                comp_mask = (labels == label_id).astype(np.uint8)
                circularity = self.component_circularity(comp_mask)
                selected_pts, selected_heights, black_count = collect_component_points(
                    labels, label_id, pts_3d_grid, black_mask
                )

                if len(selected_pts) < max(8, min_area // 2):
                    continue

                selected_heights_np = np.asarray(selected_heights, dtype=np.float32)
                median_height = float(np.median(selected_heights_np))
                height95 = float(np.percentile(selected_heights_np, 95))
                height_std = float(np.std(selected_heights_np))
                black_ratio = black_count / max(area, 1)

                if height95 < 0.004:
                    continue

                if used_height_only:
                    if circularity < 0.22 and center_ratio > 0.28:
                        continue
                    if area < 12:
                        continue

                if prefer_circularity and circularity < 0.14 and black_ratio < 0.55:
                    continue

                centroid_3d = candidate_center_from_component(
                    strategy_name, comp_centroid_2d, selected_pts, selected_heights
                )

                if centroid_3d is None:
                    continue

                if fallback_center_pt is not None:
                    dist_3d = float(np.linalg.norm(centroid_3d - fallback_center_pt))
                    if strategy_name == 'upper_strict_circularity':
                        if dist_3d > 0.045 and center_ratio > 0.30:
                            continue
                    else:
                        if dist_3d > 0.040 and center_ratio > 0.25:
                            continue
                else:
                    dist_3d = None

                area_score = min(area / max(0.10 * w * h, 1.0), 1.0)
                height_score = min(height95 / 0.012, 1.0)
                stable_height_score = max(0.0, 1.0 - min(height_std / 0.010, 1.0))
                black_score = min(black_ratio, 1.0)

                if strategy_name == 'upper_strict_circularity':
                    score = (
                        2.6 * circularity
                        + 1.2 * area_score
                        + 1.1 * black_score
                        + 1.2 * height_score
                        + 0.8 * stable_height_score
                        - 1.25 * center_ratio
                    )
                    if used_height_only:
                        score -= 0.35
                else:
                    score = (
                        1.3 * circularity
                        + 1.4 * area_score
                        + 1.6 * black_score
                        + 1.5 * height_score
                        + 0.6 * stable_height_score
                        - 1.0 * center_ratio
                    )

                candidates.append({
                    'strategy': strategy_name + ('_height_only' if used_height_only else ''),
                    'center': centroid_3d,
                    'height95': height95,
                    'score': float(score),
                    'area': area,
                    'circularity': float(circularity),
                    'black_ratio': float(black_ratio),
                    'center_ratio': float(center_ratio),
                    'height_std': float(height_std),
                    'dist_3d': dist_3d,
                })

            return candidates

        candidates = []
        candidates.extend(build_candidates(
            strategy_name='upper_strict_circularity',
            black_mask=strict_black,
            height_min=0.003,
            height_max=0.030,
            min_area=8,
            center_limit=0.55,
            allow_height_only=True,
            prefer_circularity=True,
        ))

        candidates.extend(build_candidates(
            strategy_name='lower_relaxed_height',
            black_mask=relaxed_black,
            height_min=0.005,
            height_max=0.030,
            min_area=10,
            center_limit=0.40,
            allow_height_only=False,
            prefer_circularity=False,
        ))

        if not candidates:
            return None, None

        candidates.sort(key=lambda c: c['score'], reverse=True)
        best = candidates[0]

        self.get_logger().info(
            f'🧭 중심 후보 선택[{best["strategy"]}]: score={best["score"]:.2f}, '
            f'area={best["area"]}, circularity={best["circularity"]:.2f}, '
            f'black_ratio={best["black_ratio"]:.2f}, center_ratio={best["center_ratio"]:.2f}, '
            f'height95={best["height95"] * 1000.0:.1f}mm'
        )

        return best['center'], best['height95']

    def analyze_screw_with_retry(self, bbox, depth_img, cv_image, fx, fy, cx, cy, tf_mat, plane_point, plane_normal):
        if plane_point is None or plane_normal is None or np.linalg.norm(plane_normal) < 1e-6:
            return None, None, None, None

        normal = self.normalize_vec(plane_normal)
        if normal[2] < 0:
            normal = -normal

        center_3d, max_height_m = self.get_robust_screw_center(
            bbox, depth_img, cv_image, fx, fy, cx, cy, tf_mat, plane_point, normal
        )

        if center_3d is None:
            return None, None, None, None

        return normal, center_3d, plane_point, max_height_m

    def fit_plane_ransac(self, pts, max_iterations=100, threshold=0.01):
        if len(pts) < 3:
            return np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.0])
            
        # 속도 향상을 위해 포인트가 많으면 다운샘플링
        if len(pts) > 3000:
            indices = np.random.choice(len(pts), 3000, replace=False)
            pts_sampled = pts[indices]
        else:
            pts_sampled = pts

        best_inliers = []
        best_plane = None  # (normal, point)
        n_pts = len(pts_sampled)

        for _ in range(max_iterations):
            # 무작위로 3개 점 선택
            idx = np.random.choice(n_pts, 3, replace=False)
            p1, p2, p3 = pts_sampled[idx]

            # 두 벡터 및 법선 벡터 계산
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm < 1e-6:
                continue
            normal = normal / norm

            # 모든 점과 평면 사이의 거리 계산
            distances = np.abs(np.dot(pts_sampled - p1, normal))
            inliers = np.where(distances < threshold)[0]

            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_plane = (normal, p1)

        if best_plane is None:
            # 평면 피팅 실패 시 기본 Z축 방향 법선 및 평균값 반환
            return np.array([0.0, 0.0, 1.0]), np.mean(pts, axis=0)

        # 검출된 인라이어(Inliers)들을 바탕으로 평면 재추정 (최소자승법 SVD 활용)
        normal, p0 = best_plane
        inlier_pts = pts_sampled[best_inliers]
        if len(inlier_pts) >= 3:
            centroid = np.mean(inlier_pts, axis=0)
            # 공분산 행렬의 SVD를 통해 법선 벡터 정밀 계산
            _, _, vh = np.linalg.svd(inlier_pts - centroid)
            normal = vh[2, :]  # 가장 작은 특이값에 해당하는 고유벡터가 법선 벡터
            p0 = centroid

        # 법선 벡터의 Z 방향을 항상 위쪽(양수)으로 통일
        if normal[2] < 0:
            normal = -normal

        return normal, p0

    # =================================================================

    # [💡 모듈화 - 격리된 전용 파이어베이스 업데이트 함수 스크립트]
    # =================================================================
    
    def _update_live_scan(self, workstation_id, section_name, capture_id, timestamp_str, public_url, live_screws_data):
        """1단계: 실시간 동기화만을 추구하는 독립 live_scan 노드를 업데이트합니다 (ZYZ 오리엔테이션 각도 매핑, 3D 배경 포함)."""
        try:
            live_scan_ref = get_db_reference('live_scan')
            if live_scan_ref:
                live_scan_ref.child('workstations').child(workstation_id).set({
                    'section_name': section_name,
                    'capture_id': capture_id,
                    'timestamp': timestamp_str,
                    'background_url': public_url,
                    'screws': live_screws_data
                })
                self.get_logger().info(f'📱 [live_scan] {workstation_id} 오리엔테이션 및 3D 배경 실시간 동기화 성공')
        except Exception as e:
            self.get_logger().error(f'❌ live_scan 업데이트 실패: {e}')

    def _update_inspection_records(self, capture_id, workstation_id, section_name, timestamp_str, time_display_str, now_ms, screw_idx, normal_count, defect_count, markers_data, blob_path, public_url, tf_pc, tf_color, pc_msg, img_msg):
        """2단계: 마스터 원본 대용량 검사 아카이브 기록을 영구 누적 저장합니다 (중복 scan_history는 완전 삭제)."""
        capture_data = {
            'metadata': {
                'capture_id': capture_id, 'session_id': self.session_id,
                'company_id': settings.COMPANY_ID, 'company_name': settings.COMPANY_NAME,
                'site_id': settings.SITE_ID, 'site_name': settings.SITE_NAME,
                'line_id': settings.LINE_ID, 'line_name': settings.LINE_NAME,
                'workstation_id': workstation_id, 'workstation_name': section_name,
                'robot_id': settings.ROBOT_ID, 'camera_id': settings.CAMERA_ID,
                'schema_version': '1.0.0', 'created_at': now_ms,
                'timestamp': timestamp_str, 'time_display': time_display_str,
            },
            'summary': {
                'total_count': int(screw_idx), 'normal_count': int(normal_count),
                'defect_count': int(defect_count), 'inspection_status': 'completed',
            },
            'markers': markers_data,
            'transform_snapshot': {
                'pointcloud_frame_to_base': {
                    'source_frame': pc_msg.header.frame_id, 'target_frame': settings.BASE_FRAME,
                    'matrix_4x4': np.round(tf_pc, 8).tolist(),
                },
                'color_frame_to_base': {
                    'source_frame': img_msg.header.frame_id, 'target_frame': settings.BASE_FRAME,
                    'matrix_4x4': np.round(tf_color, 8).tolist(),
                },
            },
            'storage_refs': {'background_js': blob_path, 'background_url': public_url},
        }

        # 세션 하위 영구 아카이브 구조 저장
        get_db_reference(db_paths.session_capture_path(self.session_id, workstation_id, capture_id)).set(capture_data)

        # 구형 GUI 연동 대시보드 미러 노드 유지
        if self.inspections_ref:
            self.inspections_ref.child(capture_id).set(capture_data)

    def _update_twin_states(self, capture_id, workstation_id, now_ms, screw_idx, normal_count, defect_count):
        """3단계: 디지털 트윈 환경 모델 시뮬레이션 및 로봇팔 상호 작용 제어 상태 노드를 업데이트합니다."""
        if self.twin_state_ref:
            self.twin_state_ref.child('current_inspection').set({
                'session_id': self.session_id, 'workstation_id': workstation_id,
                'capture_id': capture_id, 'status': 'completed',
                'total_count': int(screw_idx), 'normal_count': int(normal_count),
                'defect_count': int(defect_count), 'updated_at': now_ms,
            })
            self.twin_state_ref.child('current_session').update({
                'session_id': self.session_id, 'status': 'running',
                'latest_capture_id': capture_id, 'latest_workstation_id': workstation_id,
                'total_workstations': self.capture_count, 'total_captures': self.session_total_captures,
                'total_markers': self.session_total_markers, 'normal_count': self.session_normal_count,
                'defect_count': self.session_defect_count, 'updated_at': now_ms,
            })
            self.twin_state_ref.child('robots').child(settings.ROBOT_ID).update({
                'status': 'ready', 'mode': 'inspection_completed',
                'current_session_id': self.session_id, 'current_workstation_id': workstation_id,
                'current_capture_id': capture_id, 'updated_at': now_ms,
            })
        if self.site_ref:
            self.site_ref.update({
                'latest_session_id': self.session_id, 'latest_capture_id': capture_id, 'updated_at': now_ms,
            })
        if self.robot_ref:
            self.robot_ref.update({
                'status': 'ready', 'current_session_id': self.session_id,
                'current_workstation_id': workstation_id, 'current_capture_id': capture_id, 'updated_at': now_ms,
            })

    def _update_database_indexes(self, capture_id, workstation_id, timestamp_str, now_ms, screw_idx, normal_count, defect_count, markers_data):
        """4단계: 관리 대시보드 조회 조건 필터링에 핵심적인 다차원 조건 검색형 가벼운 인덱싱을 빌드합니다."""
        capture_index_data = {
            'session_id': self.session_id, 'workstation_id': workstation_id,
            'capture_id': capture_id, 'created_at': now_ms,
            'total_count': int(screw_idx), 'normal_count': int(normal_count), 'defect_count': int(defect_count),
        }
        get_db_reference(db_paths.index_latest_path()).update({
            'latest_session_id': self.session_id, 'latest_workstation_id': workstation_id,
            'latest_capture_id': capture_id, 'updated_at': now_ms,
        })
        get_db_reference(db_paths.index_capture_lookup_path(capture_id)).set(capture_index_data)
        get_db_reference(db_paths.index_captures_by_date_path(timestamp_str[:8], capture_id)).set(capture_index_data)
        get_db_reference(db_paths.index_captures_by_workstation_path(workstation_id, capture_id)).set(capture_index_data)

        # 불량 추적 탐색용 인덱스 매핑
        for marker_id, marker in markers_data.items():
            if marker.get('status') in ['defect', 'failed']:
                get_db_reference(db_paths.index_unresolved_defect_path(f'{capture_id}_{marker_id}')).set({
                    'session_id': self.session_id, 'workstation_id': workstation_id,
                    'capture_id': capture_id, 'marker_id': marker_id,
                    'status': marker.get('status'), 'defect_type': marker.get('defect_type'),
                    'delta_z_mm': marker.get('delta_z_mm'), 'created_at': now_ms,
                })

    def _update_events(self, capture_id, workstation_id, now_ms, screw_idx, normal_count, defect_count):
        """5단계: 중앙 서버 이벤트 위험 등급 로그를 남깁니다."""
        session_ref = get_db_reference(db_paths.inspection_session_path(self.session_id))
        if session_ref:
            session_ref.child('summary').update({
                'latest_capture_id': capture_id, 'latest_workstation_id': workstation_id,
                'total_workstations': self.capture_count, 'updated_at': now_ms,
            })

        get_db_reference(db_paths.events_path()).child(db_paths.now_event_id('inspection_completed')).set({
            'event_type': 'inspection_completed', 'company_id': settings.COMPANY_ID,
            'site_id': settings.SITE_ID, 'robot_id': settings.ROBOT_ID, 'capture_id': capture_id,
            'severity': 'info' if defect_count == 0 else 'warning',
            'summary': {'total_count': int(screw_idx), 'normal_count': int(normal_count), 'defect_count': int(defect_count)},
            'created_at': now_ms, 'schema_version': '1.0.0',
        })

    # =================================================================
    # [서비스 콜백 (메인 검사 핸들러)]
    # =================================================================
    def inspect_callback(self, request, response):
        missing = []
        if self.latest_pc_msg is None:
            missing.append('pointcloud')
        if self.latest_img_msg is None:
            missing.append('color_image')
        if self.latest_depth_msg is None:
            missing.append('aligned_depth')
        if self.cam_info is None:
            missing.append('camera_info')

        if missing:
            self.get_logger().warn(f'❌ 데이터 수신 오류: missing={missing}')
            response.success = False
            response.message = f"데이터 수신 오류: {missing}"
            return response

        frame_skew_ms = self.get_stamp_skew_ms([self.latest_pc_msg, self.latest_img_msg, self.latest_depth_msg])
        if frame_skew_ms is not None and frame_skew_ms > 300:
            self.get_logger().warn(f'⚠️ RGB/Depth/PointCloud timestamp 차이가 큽니다: {frame_skew_ms}ms')
            response.success = False
            response.message = "SYNC_ERROR"
            return response

        pc_msg = self.latest_pc_msg
        img_msg = self.latest_img_msg
        depth_msg = self.latest_depth_msg

        cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        tf_pc = self.get_tf_matrix(settings.BASE_FRAME, pc_msg.header.frame_id)
        tf_color = self.get_tf_matrix(settings.BASE_FRAME, img_msg.header.frame_id)
        tf_depth = self.get_tf_matrix(settings.BASE_FRAME, depth_msg.header.frame_id)

        if tf_pc is None or tf_color is None or tf_depth is None:
            response.success = False
            response.message = "TF 변환 실패"
            return response

        fx, fy, cx, cy = self.cam_info.k[0], self.cam_info.k[4], self.cam_info.k[2], self.cam_info.k[5]

        self.get_logger().info('☁️ 3D 포인트 클라우드 배경 맵핑 중...')
        pc_data = list(pc2.read_points(pc_msg, field_names=('x', 'y', 'z', 'rgb'), skip_nans=True))
        if not pc_data:
            response.success = False
            response.message = "POINTCLOUD_EMPTY"
            return response

        pts = np.array([[p[0], p[1], p[2]] for p in pc_data], dtype=np.float32)
        colors = np.array([[(struct.unpack('I', struct.pack('f', p[3]))[0] >> 16 & 0xFF) / 255.0,
                            (struct.unpack('I', struct.pack('f', p[3]))[0] >> 8 & 0xFF) / 255.0,
                            (struct.unpack('I', struct.pack('f', p[3]))[0] & 0xFF) / 255.0] for p in pc_data], dtype=np.float32)

        # 요구 조건 반영: 정면 z 거리 기준 25~50cm만 렌더링/평면추정에 사용
        z_distances = pts[:, 2]
        mask = (z_distances >= 0.25) & (z_distances <= 0.50)
        pts_filtered, colors_filtered = pts[mask], colors[mask]

        if len(pts_filtered) < 50:
            response.success = False
            response.message = "POINTCLOUD_RANGE_EMPTY"
            return response

        pts_transformed = (tf_pc @ np.hstack([pts_filtered, np.ones((len(pts_filtered), 1))]).T).T[:, :3]

        self.get_logger().info('📐 작업대 전역 평면 피팅 중 (RANSAC)...')
        plane_normal, plane_point = self.fit_plane_ransac(pts_transformed, max_iterations=150, threshold=0.008)

        yolo_results = self.yolo_model(cv_image, verbose=False)[0]

        if len(yolo_results.boxes) == 0:
            self.get_logger().info('⏩ 나사 미발견: 검사를 생략하고 다음으로 넘어갑니다.')
            response.success = True
            response.message = "SKIPPED"
            return response

        candidate_capture_count = self.capture_count + 1
        section_name = f'작업대 {candidate_capture_count}'
        workstation_id = f'workstation_{candidate_capture_count:02d}'
        self.get_logger().info(f'📸 나사 후보 발견! [{section_name}] 정밀 분석 시작...')

        capture_id = db_paths.now_capture_id()
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        time_display_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        now_ms = int(time.time() * 1000)

        markers_data = {}
        live_screws_data = {}

        normal_count = 0
        defect_count = 0
        screw_idx = 0

        boxes_list = []
        for box in yolo_results.boxes:
            b = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy()) if box.conf is not None else 0.0
            cls = int(box.cls[0].cpu().numpy()) if box.cls is not None else -1
            boxes_list.append((conf, b, cls, box))

        boxes_list.sort(key=lambda x: x[0], reverse=True)

        filtered_boxes = []
        for conf, b, cls, box in boxes_list:
            overlap = False
            for f_conf, f_b, f_cls, f_box in filtered_boxes:
                x1_max = max(b[0], f_b[0])
                y1_max = max(b[1], f_b[1])
                x2_min = min(b[2], f_b[2])
                y2_min = min(b[3], f_b[3])

                inter_w = max(0.0, x2_min - x1_max)
                inter_h = max(0.0, y2_min - y1_max)
                inter_area = inter_w * inter_h

                area1 = (b[2] - b[0]) * (b[3] - b[1])
                area2 = (f_b[2] - f_b[0]) * (f_b[3] - f_b[1])
                union_area = area1 + area2 - inter_area

                iou = inter_area / union_area if union_area > 0 else 0.0
                min_area = min(area1, area2)
                iomin = inter_area / min_area if min_area > 0 else 0.0

                if iou > 0.4 or iomin > 0.6:
                    overlap = True
                    self.get_logger().info(
                        f"⚠️ 중복 감지 필터링: 클래스 {cls}(신뢰도 {conf:.2f})가 클래스 {f_cls}(신뢰도 {f_conf:.2f})와 중복되어 제외됨 (IoU: {iou:.2f}, IoMin: {iomin:.2f})"
                    )
                    break

            if not overlap:
                filtered_boxes.append((conf, b, cls, box))

        filtered_boxes.sort(
            key=lambda item: (
                (float(item[1][1]) + float(item[1][3])) / 2.0,
                (float(item[1][0]) + float(item[1][2])) / 2.0
            )
        )

        for conf, b, cls, box in filtered_boxes:
            confidence = conf
            screw_key = f'screw_{screw_idx + 1:02d}'

            normal, center_3d, screw_plane_point, max_height_m = self.analyze_screw_with_retry(
                b, cv_depth, cv_image, fx, fy, cx, cy, tf_depth, plane_point, plane_normal
            )

            if normal is not None and center_3d is not None:
                pose_angles = self.calculate_target_pose(normal)
                rx_val, ry_val, rz_val = pose_angles if pose_angles else (0.0, 0.0, 0.0)

                if max_height_m is not None:
                    delta_z_mm = max_height_m * 1000.0
                else:
                    vec_to_center = center_3d - screw_plane_point
                    delta_z_m = abs(np.dot(vec_to_center, normal))
                    delta_z_mm = delta_z_m * 1000.0

                status_bool = bool(delta_z_mm <= 10.0)
                pt_base_mm = center_3d * 1000.0
                protection_radius_m = self.estimate_screw_protection_radius(b, cv_depth, fx, fy)

                marker_data = self.build_marker_data(
                    screw_key, status_bool, delta_z_mm, confidence, b,
                    pt_base_mm, pose_angles, now_ms, protection_radius_m=protection_radius_m
                )
                markers_data[screw_key] = marker_data

                live_screws_data[screw_key] = {
                    'status': 'normal' if status_bool else 'defect',
                    'defect_type': 'none' if status_bool else 'height_defect',
                    'position': {
                        'x': float(pt_base_mm[0]),
                        'y': float(pt_base_mm[1]),
                        'z': float(pt_base_mm[2])
                    },
                    'orientation': {
                        'rx': float(rx_val),
                        'ry': float(ry_val),
                        'rz': float(rz_val)
                    },
                    'protection_radius_m': float(protection_radius_m),
                }

                if status_bool:
                    normal_count += 1
                else:
                    defect_count += 1
                self.get_logger().info(f'🎯 {screw_key} -> {"✅ 정상" if status_bool else "❌ 불량"} (단차: {delta_z_mm:.1f}mm)')
                screw_idx += 1
            else:
                self.get_logger().warn(f'나사 {screw_idx + 1} 데이터 추출 실패 (재시도 초과). 무시합니다.')

        if not markers_data:
            self.get_logger().info('⏩ YOLO 후보는 있었지만 유효한 3D 나사 중심이 없어 검사를 저장하지 않고 건너뜁니다.')
            response.success = True
            response.message = "SKIPPED"
            return response

        self.capture_count = candidate_capture_count
        self.ensure_session_started(now_ms)

        detected_screws = []
        for marker_key, marker in markers_data.items():
            pos = marker['position']
            detected_screws.append({
                'center': np.array([pos['x'], pos['y'], pos['z']], dtype=np.float32) / 1000.0,
                'radius': float(marker.get('protection_radius_m', 0.04)),
            })

        plane_dist_threshold = 0.015
        pts_flattened = pts_transformed.copy()
        dists_to_plane = np.dot(pts_transformed - plane_point, plane_normal)
        flatten_mask = np.abs(dists_to_plane) < plane_dist_threshold

        if len(detected_screws) > 0 and len(pts_transformed) > 0:
            for screw in detected_screws:
                screw_center = screw['center']
                screw_radius_threshold = max(0.035, float(screw['radius']))
                vec_to_screw = pts_transformed - screw_center
                proj_len = np.dot(vec_to_screw, plane_normal)
                radial_vecs = vec_to_screw - np.outer(proj_len, plane_normal)
                radial_dists = np.linalg.norm(radial_vecs, axis=1)
                flatten_mask = flatten_mask & (radial_dists >= screw_radius_threshold)

        pts_flattened[flatten_mask] = pts_transformed[flatten_mask] - np.outer(dists_to_plane[flatten_mask], plane_normal)

        bg_dict = {
            'x': np.round(pts_flattened[:, 0], 4).tolist(),
            'y': np.round(pts_flattened[:, 1], 4).tolist(),
            'z': np.round(pts_flattened[:, 2], 4).tolist(),
            'colors': [f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' for c in colors_filtered],
        }
        js_filename = f'bg_{timestamp_str}.js'
        with open(js_filename, 'w', encoding='utf-8') as f:
            f.write(f'window.latestBackground = {json.dumps(bg_dict)};')

        self.get_logger().info('🚀 Firebase 실시간 데이터베이스 동기화 중...')
        blob_path = db_paths.storage_background_js_path(capture_id, self.session_id, workstation_id)
        public_url = self.upload_to_firebase_storage(js_filename, blob_path, 'application/javascript')

        if public_url and self.inspections_ref and self.twin_state_ref and self.site_ref and self.robot_ref:
            self.session_total_captures += 1
            self.session_total_markers += int(screw_idx)
            self.session_normal_count += int(normal_count)
            self.session_defect_count += int(defect_count)

            self._update_live_scan(workstation_id, section_name, capture_id, timestamp_str, public_url, live_screws_data)

            self._update_inspection_records(
                capture_id, workstation_id, section_name, timestamp_str, time_display_str, now_ms,
                screw_idx, normal_count, defect_count, markers_data, blob_path, public_url,
                tf_pc, tf_color, pc_msg, img_msg
            )

            self._update_twin_states(capture_id, workstation_id, now_ms, screw_idx, normal_count, defect_count)
            self._update_database_indexes(capture_id, workstation_id, timestamp_str, now_ms, screw_idx, normal_count, defect_count, markers_data)
            self._update_events(capture_id, workstation_id, now_ms, screw_idx, normal_count, defect_count)

            self.get_logger().info(f'🎉 [{section_name}] 하이브리드 중심 계산 및 데이터 분산 저장 완료!')
        else:
            self.get_logger().warn('⚠️ 데이터 업로드 및 DB 동기화에 실패했습니다.')

        if os.path.exists(js_filename):
            os.remove(js_filename)

        response.success = True
        response.message = "UPLOADED"
        return response


def main(args=None):
    rclpy.init(args=args)
    node = VisionServerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
import sensor_msgs_py.point_cloud2 as pc2
from tf2_ros import Buffer, TransformListener, StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
import numpy as np
from scipy.spatial.transform import Rotation as R
import struct
import cv2
import math
from cv_bridge import CvBridge
from ultralytics import YOLO
import os
import json
import time
from datetime import datetime
from pathlib import Path
import sys

# 🌟 ROS 2 표준 서비스 (통신용)
from std_srvs.srv import Trigger

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from common import settings
from common import db_paths
from common.firebase_client import get_db_reference, get_storage_bucket

class VisionServerNode(Node):
    def __init__(self):
        super().__init__('vision_server_node')

        # TF 설정
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        self.publish_static_transform()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # YOLO 로딩
        self.bridge = CvBridge()
        model_path = str(ROOT_DIR / 'YJH' / 'resource' / 'hyupdong2_yolo11x_realtest_corrected_best.pt')
        self.get_logger().info(f'YOLO 모델 로딩 중... ({model_path})')
        self.yolo_model = YOLO(model_path)
        self.get_logger().info('✅ YOLO 모델 로딩 완료!')

        # 세션 관리
        self.session_id = None
        self.session_started_at_ms = None
        self.session_total_captures = 0
        self.session_total_markers = 0
        self.session_normal_count = 0
        self.session_defect_count = 0

        # Firebase 초기화
        try:
            self.bucket = get_storage_bucket()
            self.inspections_ref = get_db_reference(db_paths.inspections_path())
            self.twin_state_ref = get_db_reference(db_paths.twin_state_path())
            self.site_ref = get_db_reference(db_paths.site_path())
            self.robot_ref = get_db_reference(db_paths.robot_path())
            self.get_logger().info('✅ Firebase Storage & Realtime Database 연결 완료!')
            self.init_base_database_nodes()
            self.init_capture_count()
        except Exception as e:
            self.get_logger().error(f'❌ Firebase 초기화 실패: {e}')
            self.bucket = None
            self.inspections_ref = None
            self.twin_state_ref = None
            self.site_ref = None
            self.robot_ref = None
            self.capture_count = 0

        # 데이터 구독
        self.latest_pc_msg = None
        self.latest_img_msg = None
        self.latest_depth_msg = None
        self.cam_info = None

        self.pc_topic_used = None
        self.img_topic_used = None
        self.depth_topic_used = None
        self.info_topic_used = None

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.pointcloud_topics = [
            '/camera/camera/depth/color/points',
            '/camera/depth/color/points',
            '/camera/camera/depth/points',
            '/camera/depth/points',
        ]
        self.color_image_topics = [
            '/camera/camera/color/image_raw',
            '/camera/color/image_raw',
        ]
        self.aligned_depth_topics = [
            '/camera/camera/aligned_depth_to_color/image_raw',
            '/camera/aligned_depth_to_color/image_raw',
            '/camera/camera/depth/image_rect_raw',
            '/camera/depth/image_rect_raw',
        ]
        self.camera_info_topics = [
            '/camera/camera/color/camera_info',
            '/camera/color/camera_info',
        ]

        self.subscriptions = []
        for topic in self.pointcloud_topics:
            self.subscriptions.append(
                self.create_subscription(PointCloud2, topic, lambda msg, t=topic: self.pc_callback(msg, t), sensor_qos)
            )
        for topic in self.color_image_topics:
            self.subscriptions.append(
                self.create_subscription(Image, topic, lambda msg, t=topic: self.img_callback(msg, t), sensor_qos)
            )
        for topic in self.aligned_depth_topics:
            self.subscriptions.append(
                self.create_subscription(Image, topic, lambda msg, t=topic: self.depth_callback(msg, t), sensor_qos)
            )
        for topic in self.camera_info_topics:
            self.subscriptions.append(
                self.create_subscription(CameraInfo, topic, lambda msg, t=topic: self.info_callback(msg, t), sensor_qos)
            )

        self.get_logger().info('✅ RealSense 토픽 자동 대응 구독 활성화')
        self.get_logger().info(f'   PointCloud 후보: {self.pointcloud_topics}')
        self.get_logger().info(f'   Color 후보: {self.color_image_topics}')
        self.get_logger().info(f'   Depth 후보: {self.aligned_depth_topics}')
        self.get_logger().info(f'   CameraInfo 후보: {self.camera_info_topics}')

        # 🌟 로봇의 검사 요청을 기다리는 서비스 서버 생성
        self.srv = self.create_service(Trigger, '/vision_inspect', self.inspect_callback)
        self.get_logger().info('🤖 비전 서버 대기 중... 로봇의 이동 완료 신호를 기다립니다.')

    def init_capture_count(self):
        self.capture_count = 0
        self.get_logger().info('🗂️ 새 검사 세션 기준으로 작업대 번호를 1부터 시작합니다.')

    def ensure_session_started(self, now_ms):
        if self.session_id:
            return
        self.session_id = db_paths.now_session_id()
        self.session_started_at_ms = now_ms
        self.get_logger().info(f'🗂️ 검사 세션 시작: {self.session_id}')

        # 🔄 [초기화] 실시간 통신용 live_scan 노드 초기화 (이전 작업 데이터 자동 삭제)
        live_scan_ref = get_db_reference('live_scan')
        if live_scan_ref:
            live_scan_ref.set({
                'session_id': self.session_id,
                'started_at': now_ms,
                'workstations': {}
            })

        if self.site_ref:
            self.site_ref.update({'latest_session_id': self.session_id, 'updated_at': now_ms})
        if self.robot_ref:
            self.robot_ref.update({'current_session_id': self.session_id, 'updated_at': now_ms})
        if self.twin_state_ref:
            self.twin_state_ref.child('current_session').set({
                'session_id': self.session_id, 'status': 'running',
                'site_id': settings.SITE_ID, 'robot_id': settings.ROBOT_ID,
                'started_at': now_ms, 'updated_at': now_ms,
            })
        session_ref = get_db_reference(db_paths.inspection_session_path(self.session_id))
        session_ref.child('metadata').set({
            'session_id': self.session_id, 'company_id': settings.COMPANY_ID,
            'company_name': settings.COMPANY_NAME, 'site_id': settings.SITE_ID,
            'site_name': settings.SITE_NAME, 'robot_id': settings.ROBOT_ID,
            'status': 'running', 'started_at': now_ms, 'updated_at': now_ms,
            'schema_version': '1.0.0',
        })
        session_ref.child('summary').set({
            'total_workstations': 0, 'total_captures': 0, 'total_markers': 0,
            'normal_count': 0, 'defect_count': 0, 'resolved_count': 0,
            'failed_count': 0, 'updated_at': now_ms,
        })

    def init_base_database_nodes(self):
        now_ms = int(time.time() * 1000)
        get_db_reference(db_paths.system_path()).update({
            'project_name': settings.COMPANY_NAME, 'schema_version': '1.0.0', 'updated_at': now_ms,
        })
        get_db_reference(db_paths.company_path()).update({
            'name': settings.COMPANY_NAME, 'created_at': now_ms, 'schema_version': '1.0.0',
        })
        self.site_ref.update({
            'company_id': settings.COMPANY_ID, 'site_name': settings.SITE_NAME,
            'active_robot_id': settings.ROBOT_ID, 'updated_at': now_ms,
        })
        self.robot_ref.update({
            'company_id': settings.COMPANY_ID, 'site_id': settings.SITE_ID,
            'robot_name': settings.ROBOT_NAME, 'robot_model': 'Doosan M0609',
            'gripper': 'OnRobot RG2', 'status': 'ready', 'updated_at': now_ms,
        })
        get_db_reference(db_paths.twin_static_path()).update({
            'company_id': settings.COMPANY_ID, 'site_id': settings.SITE_ID,
            'robots': {settings.ROBOT_ID: {
                'robot_name': settings.ROBOT_NAME, 'robot_model': 'Doosan M0609',
                'gripper': 'OnRobot RG2', 'base_frame': settings.BASE_FRAME,
                'tool_frame': 'tool0', 'camera_frame': settings.CAMERA_FRAME,
            }},
            'cameras': {settings.CAMERA_ID: {
                'camera_name': settings.CAMERA_NAME, 'type': 'depth_camera',
                'frame_id': settings.CAMERA_FRAME, 'parent_frame': 'tool0',
            }},
            'workstations': {settings.WORKSTATION_ID: {
                'display_name': settings.WORKSTATION_NAME, 'line_id': settings.LINE_ID,
                'base_frame': settings.BASE_FRAME,
            }},
            'coordinate_frames': {
                settings.BASE_FRAME: {'type': 'root'},
                settings.CAMERA_FRAME: {'parent': 'tool0'},
            },
            'updated_at': now_ms,
        })

    def upload_to_firebase_storage(self, file_path, destination_blob_name, content_type):
        if not self.bucket:
            return None
        try:
            blob = self.bucket.blob(destination_blob_name)
            blob.upload_from_filename(file_path, content_type=content_type)
            blob.make_public()
            return blob.public_url
        except Exception as e:
            self.get_logger().error(f'❌ Storage 업로드 실패: {e}')
            return None

    def build_marker_data(self, screw_idx, status_bool, delta_z_mm, confidence, bbox, pt_base_mm, pose_angles, time_ms):
        status_text = 'normal' if status_bool else 'defect'
        defect_type = 'none' if status_bool else 'height_defect'
        rx, ry, rz = pose_angles if pose_angles else (0.0, 0.0, 0.0)
        return {
            'screw_id': f'screw_{int(screw_idx)+1}', 'status': status_text,
            'status_bool_legacy': status_bool, 'defect_type': defect_type,  
            'confidence': float(confidence), 'delta_z_mm': float(delta_z_mm),
            'position': {
                'x': float(pt_base_mm[0]), 'y': float(pt_base_mm[1]),
                'z': float(pt_base_mm[2]), 'frame_id': settings.BASE_FRAME,
            },
            'orientation': {
                'rx': float(rx), 'ry': float(ry), 'rz': float(rz)
            },
            'bbox': {'x1': int(bbox[0]), 'y1': int(bbox[1]), 'x2': int(bbox[2]), 'y2': int(bbox[3])},
            'updated_at': time_ms,
        }

    def publish_static_transform(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'link_6'
        t.child_frame_id = settings.CAMERA_FRAME
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0767196
        t.transform.translation.z = 0.03991
        t.transform.rotation.x = -0.5
        t.transform.rotation.y = -0.5
        t.transform.rotation.z = -0.5
        t.transform.rotation.w = 0.5
        self.tf_static_broadcaster.sendTransform(t)

    def pc_callback(self, msg, topic=None):
        self.latest_pc_msg = msg
        if topic and self.pc_topic_used != topic:
            self.pc_topic_used = topic
            self.get_logger().info(f'📡 PointCloud 수신 시작: {topic}')

    def img_callback(self, msg, topic=None):
        self.latest_img_msg = msg
        if topic and self.img_topic_used != topic:
            self.img_topic_used = topic
            self.get_logger().info(f'📡 Color Image 수신 시작: {topic}')

    def depth_callback(self, msg, topic=None):
        self.latest_depth_msg = msg
        if topic and self.depth_topic_used != topic:
            self.depth_topic_used = topic
            self.get_logger().info(f'📡 Depth Image 수신 시작: {topic}')

    def info_callback(self, msg, topic=None):
        self.cam_info = msg
        if topic and self.info_topic_used != topic:
            self.info_topic_used = topic
            self.get_logger().info(f'📡 CameraInfo 수신 시작: {topic}')

    def get_tf_matrix(self, target, source):
        try:
            trans = self.tf_buffer.lookup_transform(target, source, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0))
            rot = R.from_quat([trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w]).as_matrix()
            mat = np.eye(4); mat[:3, :3] = rot; mat[:3, 3] = [trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z]
            return mat
        except: return None

    def normalize_vec(self, v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-6 else v

    def rot_to_zyz(self, R_mat):
        beta = math.acos(max(min(R_mat[2, 2], 1.0), -1.0))
        if abs(beta) < 1e-6:
            alpha, gamma = 0.0, math.atan2(R_mat[1, 0], R_mat[0, 0])
        else:
            alpha = math.atan2(R_mat[1, 2], R_mat[0, 2])
            gamma = math.atan2(R_mat[2, 1], -R_mat[2, 0])
        return [math.degrees(alpha), math.degrees(beta), math.degrees(gamma)]

    def wrap_angle(self, angle):
        while angle > 180.0: angle -= 360.0
        while angle < -180.0: angle += 360.0
        return angle

    def calculate_target_pose(self, normal_vec):
        if np.linalg.norm(normal_vec) < 1e-6: return None
        n = self.normalize_vec(normal_vec)
        z_axis_final = -n
        global_y = np.array([0.0, 1.0, 0.0])
        x_axis = np.cross(global_y, z_axis_final)
        
        if np.linalg.norm(x_axis) < 1e-6:
            global_x = np.array([1.0, 0.0, 0.0])
            y_axis = self.normalize_vec(np.cross(z_axis_final, global_x))
            x_axis = self.normalize_vec(np.cross(y_axis, z_axis_final))
        else:
            x_axis = self.normalize_vec(x_axis)
            y_axis = self.normalize_vec(np.cross(z_axis_final, x_axis))
            
        rot_matrix = np.column_stack((x_axis, y_axis, z_axis_final))
        rx, ry, rz = self.rot_to_zyz(rot_matrix)
        return self.wrap_angle(rx), self.wrap_angle(ry), self.wrap_angle(rz)

    def get_3d_point(self, u, v, depth_map, fx, fy, cx, cy, tf_matrix):
        z = float(depth_map[v, u]) / 1000.0
        if z <= 0.1 or z > 2.0: return None
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        pt_cam = np.array([x, y, z, 1.0])
        pt_base = tf_matrix @ pt_cam
        return pt_base[:3]

    def get_surface_point(self, u, v, depth_map, fx, fy, cx, cy, tf_matrix, radius=5):
        h, w = depth_map.shape[:2]
        pts = []
        for du in range(-radius, radius + 1, 2):
            for dv in range(-radius, radius + 1, 2):
                nu, nv = u + du, v + dv
                if 0 <= nu < w and 0 <= nv < h:
                    pt = self.get_3d_point(nu, nv, depth_map, fx, fy, cx, cy, tf_matrix)
                    if pt is not None: pts.append(pt)
        return np.mean(pts, axis=0) if pts else None

    def get_robust_screw_center(self, bbox, depth_img, cv_image, fx, fy, cx, cy, tf_mat, plane_point, normal):
        # 1. Bounding box coordinates
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h_img, w_img = depth_img.shape[:2]
        
        x1 = max(0, min(x1, w_img - 1))
        x2 = max(0, min(x2, w_img - 1))
        y1 = max(0, min(y1, h_img - 1))
        y2 = max(0, min(y2, h_img - 1))
        
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        
        # Fallback 1: bbox 중심 부근의 3D 점 (기존 방식)
        cx_img, cy_img = (x1 + x2) // 2, (y1 + y2) // 2
        fallback_pt = self.get_surface_point(cx_img, cy_img, depth_img, fx, fy, cx, cy, tf_mat, radius=3)
        
        # Fallback 2: bbox 내부 valid depth의 median을 활용한 3D 점 계산
        fallback_median_pt = None
        valid_depths = []
        valid_coords = []
        
        # 패치 추출 및 3x3 미디언 필터링
        depth_patch = depth_img[y1:y2+1, x1:x2+1].copy()
        if depth_patch.size > 0:
            depth_patch = cv2.medianBlur(depth_patch, 3)
            
        for v_idx in range(depth_patch.shape[0]):
            for u_idx in range(depth_patch.shape[1]):
                z_val = float(depth_patch[v_idx, u_idx]) / 1000.0
                if 0.1 < z_val <= 2.0:
                    valid_depths.append(z_val)
                    valid_coords.append((x1 + u_idx, y1 + v_idx))
                    
        if len(valid_depths) > 0:
            median_z = np.median(valid_depths)
            idx = np.argmin(np.abs(np.array(valid_depths) - median_z))
            u_med, v_med = valid_coords[idx]
            x_med = (u_med - cx) * median_z / fx
            y_med = (v_med - cy) * median_z / fy
            fallback_median_pt = (tf_mat @ np.array([x_med, y_med, median_z, 1.0]))[:3]
            
        best_fallback = fallback_median_pt if fallback_median_pt is not None else fallback_pt
        
        if w <= 2 or h <= 2:
            return None, None
            
        # --- 검은색 마스크 추출 (HSV 공간 활용) ---
        bgr_patch = cv_image[y1:y2+1, x1:x2+1]
        hsv_patch = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2HSV)
        
        # 검은색 임계값 (V가 90 이하인 어두운 영역)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 90])
        black_mask = cv2.inRange(hsv_patch, lower_black, upper_black)
        
        # Step 2: 평면보다 튀어나오고 + 검은색인 점만 추출해서 binary mask 생성
        binary_mask = np.zeros((h, w), dtype=np.uint8)
        pts_3d_grid = {}
        
        for v_idx in range(h):
            for u_idx in range(w):
                # 검은색 영역이 아니면 필터링
                if black_mask[v_idx, u_idx] == 0:
                    continue
                    
                u = x1 + u_idx
                v = y1 + v_idx
                z = float(depth_patch[v_idx, u_idx]) / 1000.0
                
                if z <= 0.1 or z > 2.0:
                    continue
                    
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                pt_base = (tf_mat @ np.array([x, y, z, 1.0]))[:3]
                
                # 높이 계산
                height = np.dot(pt_base - plane_point, normal)
                
                # 5mm 이상, 30mm 이하로 튀어나온 점만 볼트 후보군 (그림자/바닥 노이즈 차단을 위해 하한값 5mm로 상향)
                if 0.005 <= height <= 0.030:
                    binary_mask[v_idx, u_idx] = 1
                    pts_3d_grid[(v_idx, u_idx)] = (pt_base, height)

        # Step 3: connected components 분석
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask)
        
        best_label = -1
        min_dist_to_center = 9999.0
        bbox_center_2d = np.array([w / 2.0, h / 2.0])
        
        # 1차 기준: area >= 20
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 20:
                continue
                
            centroid_2d = centroids[i]
            dist = np.linalg.norm(centroid_2d - bbox_center_2d)
            if dist < min_dist_to_center:
                min_dist_to_center = dist
                best_label = i
                
        # 2차 기준: 검은색 영역이 협소할 경우 완화된 기준으로 재탐색 (area >= 10)
        if best_label == -1:
            min_dist_to_center = 9999.0
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area < 10:
                    continue
                    
                centroid_2d = centroids[i]
                dist = np.linalg.norm(centroid_2d - bbox_center_2d)
                if dist < min_dist_to_center:
                    min_dist_to_center = dist
                    best_label = i
                    
        if best_label == -1:
            return None, None
            
        # 선택된 컴포넌트의 3D 포인트 추출
        selected_pts = []
        selected_heights = []
        for v_idx in range(h):
            for u_idx in range(w):
                if labels[v_idx, u_idx] == best_label:
                    if (v_idx, u_idx) in pts_3d_grid:
                        pt_base, height = pts_3d_grid[(v_idx, u_idx)]
                        selected_pts.append(pt_base)
                        selected_heights.append(height)
                        
        if len(selected_pts) < 10:
            return None, None
            
        centroid_3d = np.mean(selected_pts, axis=0)
        max_height_m = np.max(selected_heights)
        
        # Step 4: 중심점 검증 조건
        # 1) 2D 검증: 대각선 30% 초과시 reject
        diag = math.sqrt(w**2 + h**2)
        comp_centroid_2d = centroids[best_label]
        dist_2d = np.linalg.norm(comp_centroid_2d - bbox_center_2d)
        
        if dist_2d > 0.3 * diag:
            self.get_logger().info(f'⚠️ Centroid 2D distance to bbox center ({dist_2d:.1f} px) exceeded 30% of diagonal ({0.3*diag:.1f} px). Rejecting centroid.')
            return None, None
            
        # 2) 3D 검증: fallback_pt 대비 15mm 초과시 reject
        if fallback_pt is not None:
            dist_3d = np.linalg.norm(centroid_3d - fallback_pt)
            if dist_3d > 0.015:
                self.get_logger().info(f'⚠️ Centroid 3D distance to center point ({dist_3d*1000.0:.1f} mm) exceeded 15mm. Rejecting centroid.')
                return None, None
                
        return centroid_3d, max_height_m

    def analyze_screw_with_retry(self, bbox, depth_img, cv_image, fx, fy, cx, cy, tf_mat, plane_point, normal):
        # 전역 작업대 평면 정보를 전달받아 로컬 평면 피팅 오차를 제거하고 직접 중심 및 최고점 산출
        centroid_3d, max_height_m = self.get_robust_screw_center(bbox, depth_img, cv_image, fx, fy, cx, cy, tf_mat, plane_point, normal)
        if centroid_3d is not None:
            return normal, centroid_3d, plane_point, max_height_m
        return None, None, None, None

    def fit_plane_ransac(self, pts, max_iterations=100, threshold=0.01):
        if len(pts) < 3:
            return np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.0])
            
        # 속도 향상을 위해 포인트가 많으면 다운샘플링
        if len(pts) > 3000:
            indices = np.random.choice(len(pts), 3000, replace=False)
            pts_sampled = pts[indices]
        else:
            pts_sampled = pts

        best_inliers = []
        best_plane = None  # (normal, point)
        n_pts = len(pts_sampled)

        for _ in range(max_iterations):
            # 무작위로 3개 점 선택
            idx = np.random.choice(n_pts, 3, replace=False)
            p1, p2, p3 = pts_sampled[idx]

            # 두 벡터 및 법선 벡터 계산
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm < 1e-6:
                continue
            normal = normal / norm

            # 모든 점과 평면 사이의 거리 계산
            distances = np.abs(np.dot(pts_sampled - p1, normal))
            inliers = np.where(distances < threshold)[0]

            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_plane = (normal, p1)

        if best_plane is None:
            # 평면 피팅 실패 시 기본 Z축 방향 법선 및 평균값 반환
            return np.array([0.0, 0.0, 1.0]), np.mean(pts, axis=0)

        # 검출된 인라이어(Inliers)들을 바탕으로 평면 재추정 (최소자승법 SVD 활용)
        normal, p0 = best_plane
        inlier_pts = pts_sampled[best_inliers]
        if len(inlier_pts) >= 3:
            centroid = np.mean(inlier_pts, axis=0)
            # 공분산 행렬의 SVD를 통해 법선 벡터 정밀 계산
            _, _, vh = np.linalg.svd(inlier_pts - centroid)
            normal = vh[2, :]  # 가장 작은 특이값에 해당하는 고유벡터가 법선 벡터
            p0 = centroid

        # 법선 벡터의 Z 방향을 항상 위쪽(양수)으로 통일
        if normal[2] < 0:
            normal = -normal

        return normal, p0

    # =================================================================

    # [💡 모듈화 - 격리된 전용 파이어베이스 업데이트 함수 스크립트]
    # =================================================================
    
    def _update_live_scan(self, workstation_id, section_name, capture_id, timestamp_str, public_url, live_screws_data):
        """1단계: 실시간 동기화만을 추구하는 독립 live_scan 노드를 업데이트합니다 (ZYZ 오리엔테이션 각도 매핑, 3D 배경 포함)."""
        try:
            live_scan_ref = get_db_reference('live_scan')
            if live_scan_ref:
                live_scan_ref.child('workstations').child(workstation_id).set({
                    'section_name': section_name,
                    'capture_id': capture_id,
                    'timestamp': timestamp_str,
                    'background_url': public_url,
                    'screws': live_screws_data
                })
                self.get_logger().info(f'📱 [live_scan] {workstation_id} 오리엔테이션 및 3D 배경 실시간 동기화 성공')
        except Exception as e:
            self.get_logger().error(f'❌ live_scan 업데이트 실패: {e}')

    def _update_inspection_records(self, capture_id, workstation_id, section_name, timestamp_str, time_display_str, now_ms, screw_idx, normal_count, defect_count, markers_data, blob_path, public_url, tf_pc, tf_color, pc_msg, img_msg):
        """2단계: 마스터 원본 대용량 검사 아카이브 기록을 영구 누적 저장합니다 (중복 scan_history는 완전 삭제)."""
        capture_data = {
            'metadata': {
                'capture_id': capture_id, 'session_id': self.session_id,
                'company_id': settings.COMPANY_ID, 'company_name': settings.COMPANY_NAME,
                'site_id': settings.SITE_ID, 'site_name': settings.SITE_NAME,
                'line_id': settings.LINE_ID, 'line_name': settings.LINE_NAME,
                'workstation_id': workstation_id, 'workstation_name': section_name,
                'robot_id': settings.ROBOT_ID, 'camera_id': settings.CAMERA_ID,
                'schema_version': '1.0.0', 'created_at': now_ms,
                'timestamp': timestamp_str, 'time_display': time_display_str,
            },
            'summary': {
                'total_count': int(screw_idx), 'normal_count': int(normal_count),
                'defect_count': int(defect_count), 'inspection_status': 'completed',
            },
            'markers': markers_data,
            'transform_snapshot': {
                'pointcloud_frame_to_base': {
                    'source_frame': pc_msg.header.frame_id, 'target_frame': settings.BASE_FRAME,
                    'matrix_4x4': np.round(tf_pc, 8).tolist(),
                },
                'color_frame_to_base': {
                    'source_frame': img_msg.header.frame_id, 'target_frame': settings.BASE_FRAME,
                    'matrix_4x4': np.round(tf_color, 8).tolist(),
                },
            },
            'storage_refs': {'background_js': blob_path, 'background_url': public_url},
        }

        # 세션 하위 영구 아카이브 구조 저장
        get_db_reference(db_paths.session_capture_path(self.session_id, workstation_id, capture_id)).set(capture_data)

        # 구형 GUI 연동 대시보드 미러 노드 유지
        if self.inspections_ref:
            self.inspections_ref.child(capture_id).set(capture_data)

    def _update_twin_states(self, capture_id, workstation_id, now_ms, screw_idx, normal_count, defect_count):
        """3단계: 디지털 트윈 환경 모델 시뮬레이션 및 로봇팔 상호 작용 제어 상태 노드를 업데이트합니다."""
        if self.twin_state_ref:
            self.twin_state_ref.child('current_inspection').set({
                'session_id': self.session_id, 'workstation_id': workstation_id,
                'capture_id': capture_id, 'status': 'completed',
                'total_count': int(screw_idx), 'normal_count': int(normal_count),
                'defect_count': int(defect_count), 'updated_at': now_ms,
            })
            self.twin_state_ref.child('current_session').update({
                'session_id': self.session_id, 'status': 'running',
                'latest_capture_id': capture_id, 'latest_workstation_id': workstation_id,
                'total_workstations': self.capture_count, 'total_captures': self.session_total_captures,
                'total_markers': self.session_total_markers, 'normal_count': self.session_normal_count,
                'defect_count': self.session_defect_count, 'updated_at': now_ms,
            })
            self.twin_state_ref.child('robots').child(settings.ROBOT_ID).update({
                'status': 'ready', 'mode': 'inspection_completed',
                'current_session_id': self.session_id, 'current_workstation_id': workstation_id,
                'current_capture_id': capture_id, 'updated_at': now_ms,
            })
        if self.site_ref:
            self.site_ref.update({
                'latest_session_id': self.session_id, 'latest_capture_id': capture_id, 'updated_at': now_ms,
            })
        if self.robot_ref:
            self.robot_ref.update({
                'status': 'ready', 'current_session_id': self.session_id,
                'current_workstation_id': workstation_id, 'current_capture_id': capture_id, 'updated_at': now_ms,
            })

    def _update_database_indexes(self, capture_id, workstation_id, timestamp_str, now_ms, screw_idx, normal_count, defect_count, markers_data):
        """4단계: 관리 대시보드 조회 조건 필터링에 핵심적인 다차원 조건 검색형 가벼운 인덱싱을 빌드합니다."""
        capture_index_data = {
            'session_id': self.session_id, 'workstation_id': workstation_id,
            'capture_id': capture_id, 'created_at': now_ms,
            'total_count': int(screw_idx), 'normal_count': int(normal_count), 'defect_count': int(defect_count),
        }
        get_db_reference(db_paths.index_latest_path()).update({
            'latest_session_id': self.session_id, 'latest_workstation_id': workstation_id,
            'latest_capture_id': capture_id, 'updated_at': now_ms,
        })
        get_db_reference(db_paths.index_capture_lookup_path(capture_id)).set(capture_index_data)
        get_db_reference(db_paths.index_captures_by_date_path(timestamp_str[:8], capture_id)).set(capture_index_data)
        get_db_reference(db_paths.index_captures_by_workstation_path(workstation_id, capture_id)).set(capture_index_data)

        # 불량 추적 탐색용 인덱스 매핑
        for marker_id, marker in markers_data.items():
            if marker.get('status') in ['defect', 'failed']:
                get_db_reference(db_paths.index_unresolved_defect_path(f'{capture_id}_{marker_id}')).set({
                    'session_id': self.session_id, 'workstation_id': workstation_id,
                    'capture_id': capture_id, 'marker_id': marker_id,
                    'status': marker.get('status'), 'defect_type': marker.get('defect_type'),
                    'delta_z_mm': marker.get('delta_z_mm'), 'created_at': now_ms,
                })

    def _update_events(self, capture_id, workstation_id, now_ms, screw_idx, normal_count, defect_count):
        """5단계: 중앙 서버 이벤트 위험 등급 로그를 남깁니다."""
        session_ref = get_db_reference(db_paths.inspection_session_path(self.session_id))
        if session_ref:
            session_ref.child('summary').update({
                'latest_capture_id': capture_id, 'latest_workstation_id': workstation_id,
                'total_workstations': self.capture_count, 'updated_at': now_ms,
            })

        get_db_reference(db_paths.events_path()).child(db_paths.now_event_id('inspection_completed')).set({
            'event_type': 'inspection_completed', 'company_id': settings.COMPANY_ID,
            'site_id': settings.SITE_ID, 'robot_id': settings.ROBOT_ID, 'capture_id': capture_id,
            'severity': 'info' if defect_count == 0 else 'warning',
            'summary': {'total_count': int(screw_idx), 'normal_count': int(normal_count), 'defect_count': int(defect_count)},
            'created_at': now_ms, 'schema_version': '1.0.0',
        })

    # =================================================================
    # [서비스 콜백 (메인 검사 핸들러)]
    # =================================================================
    def inspect_callback(self, request, response):
        missing = []
        if self.latest_pc_msg is None:
            missing.append('pointcloud')
        if self.latest_img_msg is None:
            missing.append('color_image')
        if self.latest_depth_msg is None:
            missing.append('aligned_depth_or_depth')
        if self.cam_info is None:
            missing.append('camera_info')

        if missing:
            self.get_logger().warn(
                f'❌ 데이터 수신 오류: missing={missing}, '
                f'used_topics={{pc:{self.pc_topic_used}, color:{self.img_topic_used}, depth:{self.depth_topic_used}, info:{self.info_topic_used}}}'
            )
            response.success = False
            response.message = f"데이터 수신 오류: {missing}"
            return response

        self.get_logger().info(
            f'✅ 검사 입력 토픽 확인: pc={self.pc_topic_used}, color={self.img_topic_used}, '
            f'depth={self.depth_topic_used}, info={self.info_topic_used}'
        )

        cv_image = self.bridge.imgmsg_to_cv2(self.latest_img_msg, desired_encoding='bgr8')
        cv_depth = self.bridge.imgmsg_to_cv2(self.latest_depth_msg, desired_encoding='passthrough')

        if cv_image.shape[:2] != cv_depth.shape[:2]:
            self.get_logger().warn(
                f'⚠️ Color/Depth 해상도 불일치: color={cv_image.shape[:2]}, depth={cv_depth.shape[:2]}, '
                f'depth_topic={self.depth_topic_used}. aligned_depth_to_color 토픽 사용을 권장합니다.'
            )

        yolo_results = self.yolo_model(cv_image, verbose=False)[0]

        if len(yolo_results.boxes) == 0:
            self.get_logger().info('⏩ 나사 미발견: 검사를 생략하고 다음으로 넘어갑니다.')
            response.success = True; response.message = "SKIPPED"
            return response

        self.capture_count += 1
        section_name = f'작업대 {self.capture_count}'
        workstation_id = f'workstation_{self.capture_count:02d}'
        self.get_logger().info(f'📸 나사 발견! [{section_name}] 정밀 분석 및 업로드 시작...')

        capture_id = db_paths.now_capture_id()
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        time_display_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        now_ms = int(time.time() * 1000)
        
        # 새 세션 시작 시 live_scan 데이터 자동 초기화 트리거
        self.ensure_session_started(now_ms)

        pc_msg = self.latest_pc_msg
        img_msg = self.latest_img_msg

        tf_pc = self.get_tf_matrix(settings.BASE_FRAME, pc_msg.header.frame_id)
        tf_color = self.get_tf_matrix(settings.BASE_FRAME, img_msg.header.frame_id)
        tf_depth = self.get_tf_matrix(settings.BASE_FRAME, self.latest_depth_msg.header.frame_id)
        
        if tf_pc is None or tf_color is None or tf_depth is None:
            response.success = False; response.message = "TF 변환 실패"
            return response

        fx, fy, cx, cy = self.cam_info.k[0], self.cam_info.k[4], self.cam_info.k[2], self.cam_info.k[5]
        
        # 1. 3D 포인트 클라우드 배경 맵핑 및 전역 평면 피팅
        self.get_logger().info('☁️ 3D 포인트 클라우드 배경 맵핑 및 전역 평면 피팅 중...')
        pc_data = list(pc2.read_points(pc_msg, field_names=('x', 'y', 'z', 'rgb'), skip_nans=True))
        pts = np.array([[p[0], p[1], p[2]] for p in pc_data])
        colors = np.array([[(struct.unpack('I', struct.pack('f', p[3]))[0] >> 16 & 0xFF) / 255.0,
                            (struct.unpack('I', struct.pack('f', p[3]))[0] >> 8 & 0xFF) / 255.0,
                            (struct.unpack('I', struct.pack('f', p[3]))[0] & 0xFF) / 255.0] for p in pc_data])
        distances = np.linalg.norm(pts, axis=1)
        mask = (distances >= 0.25) & (distances <= 1.0)
        pts_filtered, colors_filtered = pts[mask], colors[mask]
        pts_transformed = (tf_pc @ np.hstack([pts_filtered, np.ones((len(pts_filtered), 1))]).T).T[:, :3]

        # RANSAC을 이용해 전역 작업대 평면 피팅
        self.get_logger().info('📐 전역 작업대 평면 피팅 중 (RANSAC)...')
        global_plane_normal, global_plane_point = self.fit_plane_ransac(pts_transformed, max_iterations=100, threshold=0.01)

        markers_data = {}
        live_screws_data = {}  # 🔄 신규 라이브 뷰 전용 데이터 딕셔너리
        
        normal_count = 0
        defect_count = 0
        screw_idx = 0

        # 1개의 나사에 대해 중복 검출(바운딩 박스 여러 개) 방지를 위한 Class-agnostic NMS
        boxes_list = []
        for box in yolo_results.boxes:
            b = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy()) if box.conf is not None else 0.0
            cls = int(box.cls[0].cpu().numpy()) if box.cls is not None else -1
            boxes_list.append((conf, b, cls, box))
        
        # 신뢰도 내림차순 정렬
        boxes_list.sort(key=lambda x: x[0], reverse=True)
        
        filtered_boxes = []
        for conf, b, cls, box in boxes_list:
            overlap = False
            for f_conf, f_b, f_cls, f_box in filtered_boxes:
                x1_max = max(b[0], f_b[0])
                y1_max = max(b[1], f_b[1])
                x2_min = min(b[2], f_b[2])
                y2_min = min(b[3], f_b[3])
                
                inter_w = max(0.0, x2_min - x1_max)
                inter_h = max(0.0, y2_min - y1_max)
                inter_area = inter_w * inter_h
                
                area1 = (b[2] - b[0]) * (b[3] - b[1])
                area2 = (f_b[2] - f_b[0]) * (f_b[3] - f_b[1])
                union_area = area1 + area2 - inter_area
                
                iou = inter_area / union_area if union_area > 0 else 0.0
                min_area = min(area1, area2)
                iomin = inter_area / min_area if min_area > 0 else 0.0
                
                # 중복 감지 판단 기준 (IoU가 0.4를 초과하거나, 한 박스가 다른 박스에 60% 이상 포함될 때)
                if iou > 0.4 or iomin > 0.6:
                    overlap = True
                    self.get_logger().info(
                        f"⚠️ 중복 감지 필터링: 클래스 {cls}(신뢰도 {conf:.2f})가 클래스 {f_cls}(신뢰도 {f_conf:.2f})와 중복되어 제외됨 (IoU: {iou:.2f}, IoMin: {iomin:.2f})"
                    )
                    break
            
            if not overlap:
                filtered_boxes.append((conf, b, cls, box))

        for conf, b, cls, box in filtered_boxes:
            confidence = conf
            # 전역 평면 정보를 주입하여 나사 분석 호출
            normal, center_3d, plane_point, max_height_m = self.analyze_screw_with_retry(
                b, cv_depth, cv_image, fx, fy, cx, cy, tf_depth, global_plane_point, global_plane_normal
            )

            if normal is not None and center_3d is not None:
                # 1. 로봇팔 좌표계 타겟 매핑용 ZYZ 회전 각도 획득
                pose_angles = self.calculate_target_pose(normal)
                rx_val, ry_val, rz_val = pose_angles if pose_angles else (0.0, 0.0, 0.0)
                
                # 2. 직교 평면 투영 정밀 단차(mm) 도출 (군집 내 최고높이가 유효하면 그것을 쓰고, 없으면 무게중심 기준 높이 계산)
                if max_height_m is not None:
                    delta_z_mm = max_height_m * 1000.0
                else:
                    vec_to_center = center_3d - plane_point
                    delta_z_m = abs(np.dot(vec_to_center, normal))
                    delta_z_mm = delta_z_m * 1000.0
                
                # 3. 임계치 비교 합격 판정 및 mm 단위 위치 스케일링
                status_bool = bool(delta_z_mm <= 10.0)
                pt_base_mm = center_3d * 1000.0

                # 4. 마스터 이력 레코드 보관용 데이터 축적
                marker_data = self.build_marker_data(screw_idx, status_bool, delta_z_mm, confidence, b, pt_base_mm, pose_angles, now_ms)
                marker_key = f'screw_{screw_idx}'
                markers_data[marker_key] = marker_data

                # 5. 🔄 [구조화] live_scan 하위에도 ZYZ 회전 각도를 오리엔테이션 규격으로 일치 주입
                live_screws_data[f'screw_{screw_idx:02d}'] = {
                    'status': 'normal' if status_bool else 'defect',
                    'defect_type': 'none' if status_bool else 'height_defect',
                    'position': {
                        'x': float(pt_base_mm[0]),
                        'y': float(pt_base_mm[1]),
                        'z': float(pt_base_mm[2])
                    },
                    'orientation': {
                        'rx': float(rx_val),
                        'ry': float(ry_val),
                        'rz': float(rz_val)
                    }
                }

                if status_bool:
                    normal_count += 1
                else:
                    defect_count += 1
                self.get_logger().info(f'🎯 나사 {screw_idx} -> {"✅ 정상" if status_bool else "❌ 불량"} (단차: {delta_z_mm:.1f}mm)')
                screw_idx += 1
            else:
                self.get_logger().warn(f'나사 {screw_idx} 데이터 추출 실패. 무시합니다.')

        if not markers_data:
            self.get_logger().info('⏩ 유효한 나사가 검출되지 않았습니다. 검사를 생략하고 다음으로 넘어갑니다.')
            response.success = True
            response.message = "SKIPPED"
            return response

        # 나사 위치 정보 수집 (m 단위)
        detected_screws = []
        for marker_key, marker in markers_data.items():
            pos = marker['position']
            # position은 mm 단위로 저장되어 있으므로 m 단위로 변환
            detected_screws.append(np.array([pos['x'], pos['y'], pos['z']]) / 1000.0)
            
        # 평평화 보정 수행 (나사 제외 영역)
        plane_dist_threshold = 0.02      # 평면으로부터 2cm 이내인 점만 평평하게 보정
        screw_radius_threshold = 0.02    # 나사 중심으로부터 2cm 반경 내는 나사의 형상을 보존하기 위해 보정 제외
        
        pts_flattened = pts_transformed.copy()
        
        # 각 포인트가 평면에 사영될 때의 거리 계산
        dists_to_plane = np.dot(pts_transformed - global_plane_point, global_plane_normal)
        
        # 평면에 매우 인접한 점들만 평평화 보정 대상 마스크로 지정 (그 외 공중에 떠 있거나 너무 아래인 점들은 그대로 둠)
        flatten_mask = np.abs(dists_to_plane) < plane_dist_threshold
        
        if len(detected_screws) > 0 and len(pts_transformed) > 0:
            for screw_center in detected_screws:
                # 점 p에서 나사 중심 c로의 벡터
                vec_to_screw = pts_transformed - screw_center
                # 평면 법선 방향 성분 크기
                proj_len = np.dot(vec_to_screw, global_plane_normal)
                # 법선 방향에 수직인 성분 (방출 방사형 벡터)
                radial_vecs = vec_to_screw - np.outer(proj_len, global_plane_normal)
                # 방사형 거리 계산
                radial_dists = np.linalg.norm(radial_vecs, axis=1)
                
                # 나사 중심축에서 2cm 이내인 영역은 보정 대상에서 제외
                flatten_mask = flatten_mask & (radial_dists >= screw_radius_threshold)
                
        # 마스크에 해당하는 포인트들을 평면 법선 방향으로 투영시켜 평평하게 조절
        pts_flattened[flatten_mask] = pts_transformed[flatten_mask] - np.outer(dists_to_plane[flatten_mask], global_plane_normal)

        bg_dict = {
            'x': np.round(pts_flattened[:, 0], 4).tolist(),
            'y': np.round(pts_flattened[:, 1], 4).tolist(),
            'z': np.round(pts_flattened[:, 2], 4).tolist(),
            'colors': [f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' for c in colors_filtered],
        }
        js_filename = f'bg_{timestamp_str}.js'
        with open(js_filename, 'w', encoding='utf-8') as f:
            f.write(f'window.latestBackground = {json.dumps(bg_dict)};')

        self.get_logger().info('🚀 Firebase 실시간 데이터베이스 동기화 중...')
        blob_path = db_paths.storage_background_js_path(capture_id, self.session_id, workstation_id)
        public_url = self.upload_to_firebase_storage(js_filename, blob_path, 'application/javascript')

        if public_url and self.inspections_ref and self.twin_state_ref and self.site_ref and self.robot_ref:
            self.session_total_captures += 1
            self.session_total_markers += int(screw_idx)
            self.session_normal_count += int(normal_count)
            self.session_defect_count += int(defect_count)

            # =================================================================
            # [🔄 모듈화 실행 - 찢어놓은 하위 데이터 갱신 메서드 전격 순차 실행]
            # =================================================================
            
            # 1. 라이브 뷰 실시간 연동용 노드 업데이트 (ZYZ 포함)
            self._update_live_scan(workstation_id, section_name, capture_id, timestamp_str, public_url, live_screws_data)
            
            # 2. 메인 마스터 아카이브 영구 기록방 백업 저장 (scan_history 중복 코드 완전 배제)
            self._update_inspection_records(
                capture_id, workstation_id, section_name, timestamp_str, time_display_str, now_ms,
                screw_idx, normal_count, defect_count, markers_data, blob_path, public_url,
                tf_pc, tf_color, pc_msg, img_msg
            )
            
            # 3. 디지털 트윈 상태 동기화 관리
            self._update_twin_states(capture_id, workstation_id, now_ms, screw_idx, normal_count, defect_count)
            
            # 4. 빠른 데이터 조회를 위한 다차원 쿼리 색인 생성
            self._update_database_indexes(capture_id, workstation_id, timestamp_str, now_ms, screw_idx, normal_count, defect_count, markers_data)
            
            # 5. 상위 레벨 시스템 로그 발행
            self._update_events(
                capture_id, workstation_id, now_ms, screw_idx, normal_count, defect_count
            )

            self.get_logger().info(f'🎉 [{section_name}] 다이어트 리팩토링 데이터 분산 저장 완료!')
        else:
            self.get_logger().warn('⚠️ 데이터 업로드 및 DB 동기화에 실패했습니다.')

        if os.path.exists(js_filename):
            os.remove(js_filename)

        response.success = True
        response.message = "UPLOADED"
        return response

def main(args=None):
    rclpy.init(args=args)
    node = VisionServerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
