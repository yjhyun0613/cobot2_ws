import rclpy
from rclpy.node import Node
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

        # 🌟 캘리브레이션 데이터 로드 (TF 설정보다 먼저 실행)
        calib_path = str(ROOT_DIR / 'YJH' / 'resource' / 'T_gripper2camera.npy')
        try:
            self.gripper2cam_mat = np.load(calib_path)
            self.get_logger().info(f'✅ 카메라 캘리브레이션 데이터 로딩 완료! ({calib_path})')
        except Exception as e:
            self.get_logger().error(f'❌ 캘리브레이션 데이터 로드 실패. 수동 입력값을 대체 사용합니다: {e}')
            # 로드 실패 시 기존 수동 값을 행렬 형태로 임시 생성
            self.gripper2cam_mat = np.eye(4)
            self.gripper2cam_mat[:3, 3] = [0.0, 0.075, 0.03991]
            self.gripper2cam_mat[:3, :3] = R.from_quat([-0.5, -0.5, -0.5, 0.5]).as_matrix()

        # TF 설정
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        self.publish_static_transform()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # YOLO 로딩
        self.bridge = CvBridge()
        model_path = str(ROOT_DIR / 'YJH' / 'resource' / 'hyupdong2_yolo11x_img960_best.pt')
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
            self.legacy_ref = get_db_reference(db_paths.legacy_linestatus_path())
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
            self.legacy_ref = None
            self.capture_count = 0

        # 데이터 구독
        self.latest_pc_msg = self.latest_img_msg = self.latest_depth_msg = self.cam_info = None
        self.create_subscription(PointCloud2, '/camera/camera/depth/color/points', self.pc_callback, 1)
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.img_callback, 1)
        self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, 1)
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.info_callback, 1)

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
            'screw_id': f'screw_{screw_idx}', 'status': status_text,
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

    def build_legacy_marker_data(self, marker_data, time_display_str):
        return {
            'status': bool(marker_data.get('status_bool_legacy', False)),
            'position': {
                'x': marker_data['position']['x'],
                'y': marker_data['position']['y'],
                'z': marker_data['position']['z'],
            },
            'time': time_display_str,
        }

    def publish_static_transform(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'link_6'
        t.child_frame_id = settings.CAMERA_FRAME
        
        # 🌟 로드된 4x4 행렬에서 위치 및 회전값 추출 적용
        trans = self.gripper2cam_mat[:3, 3]
        rot_mat = self.gripper2cam_mat[:3, :3]
        quat = R.from_matrix(rot_mat).as_quat() # [x, y, z, w]

        t.transform.translation.x = float(trans[0])
        t.transform.translation.y = float(trans[1])
        t.transform.translation.z = float(trans[2])
        t.transform.rotation.x = float(quat[0])
        t.transform.rotation.y = float(quat[1])
        t.transform.rotation.z = float(quat[2])
        t.transform.rotation.w = float(quat[3])
        
        self.tf_static_broadcaster.sendTransform(t)

    def pc_callback(self, msg): self.latest_pc_msg = msg
    def img_callback(self, msg): self.latest_img_msg = msg
    def depth_callback(self, msg): self.latest_depth_msg = msg
    def info_callback(self, msg): self.cam_info = msg

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

    def analyze_screw_with_retry(self, bbox, depth_img, fx, fy, cx, cy, tf_mat):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = depth_img.shape[:2]
        cx_img, cy_img = (x1 + x2) // 2, (y1 + y2) // 2
        
        for attempt in range(5):
            offset = 10 + (attempt * 5)
            radius = 3 + attempt
            
            b_pixel = (x1 - offset, y1 - offset)
            a_pixel = (x1 - offset, y2 + offset)
            c_pixel = (x2 + offset, y1 - offset)
            
            if any(p[0] < 5 or p[0] >= w - 5 or p[1] < 5 or p[1] >= h - 5 for p in [b_pixel, a_pixel, c_pixel]):
                continue
                
            b_3d = self.get_surface_point(b_pixel[0], b_pixel[1], depth_img, fx, fy, cx, cy, tf_mat, radius=radius)
            a_3d = self.get_surface_point(a_pixel[0], a_pixel[1], depth_img, fx, fy, cx, cy, tf_mat, radius=radius)
            c_3d = self.get_surface_point(c_pixel[0], c_pixel[1], depth_img, fx, fy, cx, cy, tf_mat, radius=radius)
            center_3d = self.get_surface_point(cx_img, cy_img, depth_img, fx, fy, cx, cy, tf_mat, radius=radius)
            
            if any(p is None for p in [b_3d, a_3d, c_3d, center_3d]):
                continue
                
            A = a_3d - b_3d
            B = c_3d - b_3d
            normal = np.cross(A, B)
            norm = np.linalg.norm(normal)
            
            if norm > 0:
                normal = normal / norm
                return normal, center_3d, b_3d
                
        return None, None, None

    # =================================================================
    # [💡 모듈화 - 격리된 전용 파이어베이스 업데이트 함수 스크립트]
    # =================================================================
    
    def _update_live_scan(self, workstation_id, live_screws_data):
        """1단계: 실시간 동기화만을 추구하는 독립 live_scan 노드를 업데이트합니다 (ZYZ 오리엔테이션 각도 매핑 완료)."""
        try:
            live_scan_ref = get_db_reference('live_scan')
            if live_scan_ref:
                live_scan_ref.child('workstations').child(workstation_id).set({
                    'screws': live_screws_data
                })
                self.get_logger().info(f'📱 [live_scan] {workstation_id} 오리엔테이션 실시간 동기화 성공')
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

    def _update_legacy_and_events(self, capture_id, workstation_id, section_name, timestamp_str, time_display_str, public_url, legacy_markers_data, now_ms, screw_idx, normal_count, defect_count):
        """5단계: 하위 유산 레거시 미러링 통신 및 중앙 서버 이벤트 위험 등급 로그를 남깁니다."""
        if self.legacy_ref:
            self.legacy_ref.child(section_name).set({
                'capture_index': self.capture_count, 'capture_id': capture_id,
                'timestamp': timestamp_str, 'background_url': public_url, 'markers': legacy_markers_data,
            })

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
        if not all([self.latest_pc_msg, self.latest_img_msg, self.latest_depth_msg, self.cam_info]):
            response.success = False; response.message = "데이터 수신 오류"
            return response

        cv_image = self.bridge.imgmsg_to_cv2(self.latest_img_msg, desired_encoding='bgr8')
        cv_depth = self.bridge.imgmsg_to_cv2(self.latest_depth_msg, desired_encoding='passthrough')
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
        
        markers_data = {}
        legacy_markers_data = {}
        live_screws_data = {}  # 🔄 신규 라이브 뷰 전용 데이터 딕셔너리
        
        normal_count = 0
        defect_count = 0
        screw_idx = 0

        for box in yolo_results.boxes:
            b = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0].cpu().numpy()) if box.conf is not None else 0.0

            # 점 3개 기하학 분석 기반 나사의 법선 및 위치 연산 호출
            normal, center_3d, plane_point = self.analyze_screw_with_retry(b, cv_depth, fx, fy, cx, cy, tf_depth)

            if normal is not None and center_3d is not None:
                # 1. 로봇팔 좌표계 타겟 매핑용 ZYZ 회전 각도 획득
                pose_angles = self.calculate_target_pose(normal)
                rx_val, ry_val, rz_val = pose_angles if pose_angles else (0.0, 0.0, 0.0)
                
                # 2. 직교 평면 투영 정밀 단차(mm) 도출
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
                legacy_markers_data[str(screw_idx)] = self.build_legacy_marker_data(marker_data, time_display_str)

                # 5. 🔄 [구조화] live_scan 하위에도 ZYZ 회전 각도를 오리엔테이션 규격으로 일치 주입
                live_screws_data[f'screw_{screw_idx:02d}'] = {
                    'status': 'normal' if status_bool else 'defective',
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
                self.get_logger().warn(f'나사 {screw_idx} 데이터 추출 실패 (재시도 초과). 무시합니다.')

        if not markers_data:
            self.get_logger().warn('모든 나사의 3D 계산이 실패했습니다. 로봇에게 재촬영을 요청합니다.')
            response.success = False
            response.message = "RETRY_VISION"
            self.capture_count -= 1
            return response

        # 3D 포인트 클라우드 배경 맵핑
        self.get_logger().info('☁️ 3D 포인트 클라우드 배경 맵핑 중...')
        pc_data = list(pc2.read_points(pc_msg, field_names=('x', 'y', 'z', 'rgb'), skip_nans=True))
        pts = np.array([[p[0], p[1], p[2]] for p in pc_data])
        colors = np.array([[(struct.unpack('I', struct.pack('f', p[3]))[0] >> 16 & 0xFF) / 255.0,
                            (struct.unpack('I', struct.pack('f', p[3]))[0] >> 8 & 0xFF) / 255.0,
                            (struct.unpack('I', struct.pack('f', p[3]))[0] & 0xFF) / 255.0] for p in pc_data])
        distances = np.linalg.norm(pts, axis=1)
        mask = (distances >= 0.25) & (distances <= 1.0)
        pts_filtered, colors_filtered = pts[mask], colors[mask]
        pts_transformed = (tf_pc @ np.hstack([pts_filtered, np.ones((len(pts_filtered), 1))]).T).T[:, :3]

        bg_dict = {
            'x': np.round(pts_transformed[:, 0], 4).tolist(),
            'y': np.round(pts_transformed[:, 1], 4).tolist(),
            'z': np.round(pts_transformed[:, 2], 4).tolist(),
            'colors': [f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' for c in colors_filtered],
        }
        js_filename = f'bg_{timestamp_str}.js'
        with open(js_filename, 'w', encoding='utf-8') as f:
            f.write(f'window.latestBackground = {json.dumps(bg_dict)};')

        self.get_logger().info('🚀 Firebase 실시간 데이터베이스 동기화 중...')
        blob_path = db_paths.storage_background_js_path(capture_id, self.session_id, workstation_id)
        public_url = self.upload_to_firebase_storage(js_filename, blob_path, 'application/javascript')

        if public_url and self.inspections_ref and self.twin_state_ref and self.site_ref and self.robot_ref and self.legacy_ref:
            self.session_total_captures += 1
            self.session_total_markers += int(screw_idx)
            self.session_normal_count += int(normal_count)
            self.session_defect_count += int(defect_count)

            # =================================================================
            # [🔄 모듈화 실행 - 찢어놓은 하위 데이터 갱신 메서드 전격 순차 실행]
            # =================================================================
            
            # 1. 라이브 뷰 실시간 연동용 노드 업데이트 (ZYZ 포함)
            self._update_live_scan(workstation_id, live_screws_data)
            
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
            
            # 5. 레거시 호환 및 상위 레벨 시스템 로그 발행
            self._update_legacy_and_events(
                capture_id, workstation_id, section_name, timestamp_str, time_display_str,
                public_url, legacy_markers_data, now_ms, screw_idx, normal_count, defect_count
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