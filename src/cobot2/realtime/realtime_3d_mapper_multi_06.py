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

    def build_marker_data(self, screw_idx, status_bool, delta_z_mm, confidence, bbox, pt_base, time_ms):
        status_text = 'normal' if status_bool else 'defect'
        defect_type = 'none' if status_bool else 'height_defect'
        return {
            'screw_id': f'screw_{screw_idx}', 'status': status_text,
            'status_bool_legacy': status_bool, 'defect_type': defect_type,
            'confidence': float(confidence), 'delta_z_mm': float(delta_z_mm),
            'position': {
                'x': float(pt_base[0]), 'y': float(pt_base[1]),
                'z': float(pt_base[2]), 'frame_id': settings.BASE_FRAME,
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
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.075
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

    def get_tf_matrix(self, target, source):
        try:
            trans = self.tf_buffer.lookup_transform(target, source, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0))
            rot = R.from_quat([trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w]).as_matrix()
            mat = np.eye(4); mat[:3, :3] = rot; mat[:3, 3] = [trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z]
            return mat
        except: return None

    # 🌟 핵심: 로봇이 호출할 때 실행되는 함수
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
        self.ensure_session_started(now_ms)

        pc_msg = self.latest_pc_msg
        img_msg = self.latest_img_msg

        tf_pc = self.get_tf_matrix(settings.BASE_FRAME, pc_msg.header.frame_id)
        tf_color = self.get_tf_matrix(settings.BASE_FRAME, img_msg.header.frame_id)
        if tf_pc is None or tf_color is None:
            response.success = False; response.message = "TF 변환 실패"
            return response

        fx, fy, cx, cy = self.cam_info.k[0], self.cam_info.k[4], self.cam_info.k[2], self.cam_info.k[5]
        markers_data = {}
        legacy_markers_data = {}
        normal_count = 0
        defect_count = 0
        screw_idx = 0

        for box in yolo_results.boxes:
            b = box.xyxy[0].cpu().numpy()
            xmin, ymin, xmax, ymax = b[0], b[1], b[2], b[3]
            confidence = float(box.conf[0].cpu().numpy()) if box.conf is not None else 0.0
            cx_img, cy_img = int((xmin + xmax) / 2), int((ymin + ymax) / 2)
            bw, bh = int(xmax - xmin), int(ymax - ymin)
            s_pts, surf_pts = [], []
            h, w = cv_depth.shape

            for dy in range(-int(bh * 0.75), int(bh * 0.75) + 1):
                for dx in range(-int(bw * 0.75), int(bw * 0.75) + 1):
                    nx, ny = cx_img + dx, cy_img + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        z_mm = float(cv_depth[ny, nx])
                        if z_mm > 0:
                            if abs(dx) < bw * 0.1 and abs(dy) < bh * 0.1:
                                s_pts.append(z_mm)
                            elif abs(dx) > bw * 0.5 or abs(dy) > bh * 0.5:
                                surf_pts.append(z_mm)
            if not s_pts or not surf_pts:
                continue

            cam_z = min(s_pts) / 1000.0
            if cam_z < 0.25 or cam_z > 1.0:
                continue

            delta_z_mm = np.median(surf_pts) - min(s_pts)
            status_bool = not (delta_z_mm > 10.0)
            pt_base = tf_color @ np.array([(cx_img - cx) * cam_z / fx, (cy_img - cy) * cam_z / fy, cam_z, 1.0])

            marker_data = self.build_marker_data(screw_idx, status_bool, delta_z_mm, confidence, [xmin, ymin, xmax, ymax], pt_base, now_ms)
            marker_key = f'screw_{screw_idx}'
            markers_data[marker_key] = marker_data
            legacy_markers_data[str(screw_idx)] = self.build_legacy_marker_data(marker_data, time_display_str)

            if status_bool:
                normal_count += 1
            else:
                defect_count += 1
            self.get_logger().info(f'🎯 나사 {screw_idx} -> {"✅ 정상" if status_bool else "❌ 불량"} (단차: {delta_z_mm:.1f}mm)')
            screw_idx += 1

        # 3D 포인트 클라우드 배경
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

        # Firebase Storage 업로드
        self.get_logger().info('🚀 Firebase 실시간 데이터베이스 동기화 중...')
        blob_path = db_paths.storage_background_js_path(capture_id, self.session_id, workstation_id)
        public_url = self.upload_to_firebase_storage(js_filename, blob_path, 'application/javascript')

        if public_url and self.inspections_ref and self.twin_state_ref and self.site_ref and self.robot_ref and self.legacy_ref:
            self.session_total_captures += 1
            self.session_total_markers += int(screw_idx)
            self.session_normal_count += int(normal_count)
            self.session_defect_count += int(defect_count)

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

            # 세션별 캡처 저장
            get_db_reference(db_paths.session_capture_path(self.session_id, workstation_id, capture_id)).set(capture_data)

            # 기존 GUI/API 호환용 mirror
            self.inspections_ref.child(capture_id).set(capture_data)

            # twin_state 업데이트
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

            # 세션 메타데이터/요약 업데이트
            session_ref = get_db_reference(db_paths.inspection_session_path(self.session_id))
            session_ref.child('metadata').update({
                'session_id': self.session_id, 'company_id': settings.COMPANY_ID,
                'company_name': settings.COMPANY_NAME, 'site_id': settings.SITE_ID,
                'site_name': settings.SITE_NAME, 'robot_id': settings.ROBOT_ID,
                'status': 'running', 'started_at': now_ms, 'updated_at': now_ms, 'schema_version': '1.0.0',
            })
            session_ref.child('summary').update({
                'latest_capture_id': capture_id, 'latest_workstation_id': workstation_id,
                'total_workstations': self.capture_count, 'updated_at': now_ms,
            })

            # 로봇 상태 업데이트
            self.twin_state_ref.child('robots').child(settings.ROBOT_ID).update({
                'status': 'ready', 'mode': 'inspection_completed',
                'current_session_id': self.session_id, 'current_workstation_id': workstation_id,
                'current_capture_id': capture_id, 'updated_at': now_ms,
            })
            self.site_ref.update({
                'latest_session_id': self.session_id, 'latest_capture_id': capture_id, 'updated_at': now_ms,
            })
            self.robot_ref.update({
                'status': 'ready', 'current_session_id': self.session_id,
                'current_workstation_id': workstation_id, 'current_capture_id': capture_id, 'updated_at': now_ms,
            })

            # 인덱스 데이터
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

            # 불량 인덱스
            for marker_id, marker in markers_data.items():
                if marker.get('status') in ['defect', 'failed']:
                    get_db_reference(db_paths.index_unresolved_defect_path(f'{capture_id}_{marker_id}')).set({
                        'session_id': self.session_id, 'workstation_id': workstation_id,
                        'capture_id': capture_id, 'marker_id': marker_id,
                        'status': marker.get('status'), 'defect_type': marker.get('defect_type'),
                        'delta_z_mm': marker.get('delta_z_mm'), 'created_at': now_ms,
                    })

            # 레거시 호환
            self.legacy_ref.child(section_name).set({
                'capture_index': self.capture_count, 'capture_id': capture_id,
                'timestamp': timestamp_str, 'background_url': public_url, 'markers': legacy_markers_data,
            })

            # 이벤트 로그
            get_db_reference(db_paths.events_path()).child(db_paths.now_event_id('inspection_completed')).set({
                'event_type': 'inspection_completed', 'company_id': settings.COMPANY_ID,
                'site_id': settings.SITE_ID, 'robot_id': settings.ROBOT_ID, 'capture_id': capture_id,
                'severity': 'info' if defect_count == 0 else 'warning',
                'summary': {'total_count': int(screw_idx), 'normal_count': int(normal_count), 'defect_count': int(defect_count)},
                'created_at': now_ms, 'schema_version': '1.0.0',
            })

            self.get_logger().info(f'🎉 [{section_name}] 데이터 저장 완료! capture_id={capture_id}, 총 {screw_idx}개 나사')
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