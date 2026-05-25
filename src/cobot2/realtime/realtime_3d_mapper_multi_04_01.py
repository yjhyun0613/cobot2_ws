import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
import sensor_msgs_py.point_cloud2 as pc2
from tf2_ros import Buffer, TransformListener, StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
import numpy as np
from scipy.spatial.transform import Rotation as R
import threading
import sys
import termios
import tty
import select
import struct
import cv2
from cv_bridge import CvBridge
from ultralytics import YOLO
import os
import json
import time
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from common import settings
from common import db_paths
from common.firebase_client import get_db_reference, get_storage_bucket


class Yolo3DMapperNode(Node):
    def __init__(self):
        super().__init__('yolo_3d_mapper_node')

        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        self.publish_static_transform()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.bridge = CvBridge()
        model_path = str(ROOT_DIR / 'YJH' / 'resource' / 'hyupdong2_yolo11x_img960_best.pt')
        self.get_logger().info(f'YOLO 모델 로딩 중... ({model_path})')
        self.yolo_model = YOLO(model_path)
        self.get_logger().info('✅ YOLO 모델 로딩 완료!')

        self.session_id = None
        self.session_started_at_ms = None
        self.session_total_captures = 0
        self.session_total_markers = 0
        self.session_normal_count = 0
        self.session_defect_count = 0

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

        self.latest_pc_msg = None
        self.latest_img_msg = None
        self.latest_depth_msg = None
        self.cam_info = None

        self.sub_pc = self.create_subscription(PointCloud2, '/camera/camera/depth/color/points', self.pc_callback, 10)
        self.sub_img = self.create_subscription(Image, '/camera/camera/color/image_raw', self.img_callback, 10)
        self.sub_depth = self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        self.sub_info = self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.info_callback, 10)

        self.get_logger().info('★ 데이터 수신 대기 중... "s" 키를 누르면 [자동 검사 및 누적]이 시작됩니다. ★')

        self.key_thread = threading.Thread(target=self.keyboard_listener)
        self.key_thread.daemon = True
        self.key_thread.start()


    def ensure_session_started(self, now_ms):
        if self.session_id:
            return

        self.session_id = db_paths.now_session_id()
        self.session_started_at_ms = now_ms

        self.get_logger().info(f'🗂️ 검사 세션 시작: {self.session_id}')

        if self.site_ref:
            self.site_ref.update({
                'latest_session_id': self.session_id,
                'updated_at': now_ms,
            })

        if self.robot_ref:
            self.robot_ref.update({
                'current_session_id': self.session_id,
                'updated_at': now_ms,
            })

        if self.twin_state_ref:
            self.twin_state_ref.child('current_session').set({
                'session_id': self.session_id,
                'status': 'running',
                'site_id': settings.SITE_ID,
                'robot_id': settings.ROBOT_ID,
                'started_at': now_ms,
                'updated_at': now_ms,
            })

        session_ref = get_db_reference(
            db_paths.inspection_session_path(self.session_id)
        )

        session_ref.child('metadata').set({
            'session_id': self.session_id,
            'company_id': settings.COMPANY_ID,
            'company_name': settings.COMPANY_NAME,
            'site_id': settings.SITE_ID,
            'site_name': settings.SITE_NAME,
            'robot_id': settings.ROBOT_ID,
            'status': 'running',
            'started_at': now_ms,
            'updated_at': now_ms,
            'schema_version': '1.0.0',
        })

        session_ref.child('summary').set({
            'total_workstations': 0,
            'total_captures': 0,
            'total_markers': 0,
            'normal_count': 0,
            'defect_count': 0,
            'resolved_count': 0,
            'failed_count': 0,
            'updated_at': now_ms,
        })

    def init_base_database_nodes(self):
        now_ms = int(time.time() * 1000)

        get_db_reference(db_paths.system_path()).update({
            'project_name': settings.COMPANY_NAME,
            'schema_version': '1.0.0',
            'updated_at': now_ms,
        })

        get_db_reference(db_paths.company_path()).update({
            'name': settings.COMPANY_NAME,
            'created_at': now_ms,
            'schema_version': '1.0.0',
        })

        self.site_ref.update({
            'company_id': settings.COMPANY_ID,
            'site_name': settings.SITE_NAME,
            'active_robot_id': settings.ROBOT_ID,
            'updated_at': now_ms,
        })

        self.robot_ref.update({
            'company_id': settings.COMPANY_ID,
            'site_id': settings.SITE_ID,
            'robot_name': settings.ROBOT_NAME,
            'robot_model': 'Doosan M0609',
            'gripper': 'OnRobot RG2',
            'status': 'ready',
            'updated_at': now_ms,
        })

        get_db_reference(db_paths.twin_static_path()).update({
            'company_id': settings.COMPANY_ID,
            'site_id': settings.SITE_ID,
            'robots': {
                settings.ROBOT_ID: {
                    'robot_name': settings.ROBOT_NAME,
                    'robot_model': 'Doosan M0609',
                    'gripper': 'OnRobot RG2',
                    'base_frame': settings.BASE_FRAME,
                    'tool_frame': 'tool0',
                    'camera_frame': settings.CAMERA_FRAME,
                }
            },
            'cameras': {
                settings.CAMERA_ID: {
                    'camera_name': settings.CAMERA_NAME,
                    'type': 'depth_camera',
                    'frame_id': settings.CAMERA_FRAME,
                    'parent_frame': 'tool0',
                }
            },
            'workstations': {
                settings.WORKSTATION_ID: {
                    'display_name': settings.WORKSTATION_NAME,
                    'line_id': settings.LINE_ID,
                    'base_frame': settings.BASE_FRAME,
                }
            },
            'coordinate_frames': {
                settings.BASE_FRAME: {
                    'type': 'root'
                },
                settings.CAMERA_FRAME: {
                    'parent': 'tool0'
                }
            },
            'updated_at': now_ms,
        })

    def init_capture_count(self):
        self.capture_count = 0
        self.get_logger().info(
            '🗂️ 새 검사 세션 기준으로 작업대 번호를 1부터 시작합니다. session_id는 첫 검사 시작 시 생성됩니다.'
        )

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

    def pc_callback(self, msg):
        self.latest_pc_msg = msg

    def img_callback(self, msg):
        self.latest_img_msg = msg

    def depth_callback(self, msg):
        self.latest_depth_msg = msg

    def info_callback(self, msg):
        self.cam_info = msg

    def keyboard_listener(self):
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while rclpy.ok():
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)
                    if key.lower() == 's':
                        self.process_capture()
                    elif key == '\x03':
                        rclpy.shutdown()
                        break
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def get_tf_matrix(self, target_frame, source_frame):
        try:
            trans = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0),
            )
            rot = R.from_quat([
                trans.transform.rotation.x,
                trans.transform.rotation.y,
                trans.transform.rotation.z,
                trans.transform.rotation.w,
            ]).as_matrix()
            mat = np.eye(4)
            mat[:3, :3] = rot
            mat[:3, 3] = [
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z,
            ]
            return mat
        except Exception:
            return None

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
            'screw_id': f'screw_{screw_idx}',
            'status': status_text,
            'status_bool_legacy': status_bool,
            'defect_type': defect_type,
            'confidence': float(confidence),
            'delta_z_mm': float(delta_z_mm),
            'position': {
                'x': float(pt_base[0]),
                'y': float(pt_base[1]),
                'z': float(pt_base[2]),
                'frame_id': settings.BASE_FRAME,
            },
            'bbox': {
                'x1': int(bbox[0]),
                'y1': int(bbox[1]),
                'x2': int(bbox[2]),
                'y2': int(bbox[3]),
            },
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

    def process_capture(self):
        if not all([self.latest_pc_msg, self.latest_img_msg, self.latest_depth_msg, self.cam_info]):
            self.get_logger().warn('카메라 데이터를 받지 못했습니다. 잠시 후 다시 시도해주세요.')
            return

        self.capture_count += 1
        section_name = f'작업대 {self.capture_count}'
        workstation_id = f'workstation_{self.capture_count:02d}'

        self.get_logger().info(f'\n{"="*50}\n📸 [검사 진행] {section_name} 영역 스캔 및 분석 시작...\n{"="*50}')

        pc_msg = self.latest_pc_msg
        img_msg = self.latest_img_msg
        depth_msg = self.latest_depth_msg
        cam_info = self.cam_info

        capture_id = db_paths.now_capture_id()
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        time_display_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        now_ms = int(time.time() * 1000)
        self.ensure_session_started(now_ms)

        tf_pc = self.get_tf_matrix(settings.BASE_FRAME, pc_msg.header.frame_id)
        tf_color = self.get_tf_matrix(settings.BASE_FRAME, img_msg.header.frame_id)

        if tf_pc is None or tf_color is None:
            self.get_logger().error('TF 좌표 변환에 실패했습니다. 로봇 연결 상태를 확인하세요.')
            return

        fx, fy, cx, cy = cam_info.k[0], cam_info.k[4], cam_info.k[2], cam_info.k[5]
        cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        yolo_results = self.yolo_model(cv_image, verbose=False)[0]
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
            bw_img, bh_img = int(xmax - xmin), int(ymax - ymin)

            screw_pts, surf_pts = [], []
            h, w = cv_depth.shape
            search_rx, search_ry = int(bw_img * 0.75), int(bh_img * 0.75)

            for dy in range(-search_ry, search_ry + 1):
                for dx in range(-search_rx, search_rx + 1):
                    nx, ny = cx_img + dx, cy_img + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        z_mm = float(cv_depth[ny, nx])
                        if z_mm > 0:
                            if abs(dx) < bw_img * 0.1 and abs(dy) < bh_img * 0.1:
                                screw_pts.append(z_mm)
                            elif abs(dx) > bw_img * 0.5 or abs(dy) > bh_img * 0.5:
                                surf_pts.append(z_mm)

            if not screw_pts or not surf_pts:
                continue

            z_screw = min(screw_pts)
            cam_z = z_screw / 1000.0

            if cam_z < 0.25 or cam_z > 1.0:
                continue

            z_surf = np.median(surf_pts)
            delta_z_mm = z_surf - z_screw
            is_defective = delta_z_mm > 10.0
            status_bool = not is_defective

            pt_base = tf_color @ np.array([
                (cx_img - cx) * cam_z / fx,
                (cy_img - cy) * cam_z / fy,
                cam_z,
                1.0,
            ])

            marker_data = self.build_marker_data(
                screw_idx=screw_idx,
                status_bool=status_bool,
                delta_z_mm=delta_z_mm,
                confidence=confidence,
                bbox=[xmin, ymin, xmax, ymax],
                pt_base=pt_base,
                time_ms=now_ms,
            )

            marker_key = f'screw_{screw_idx}'
            markers_data[marker_key] = marker_data
            legacy_markers_data[str(screw_idx)] = self.build_legacy_marker_data(marker_data, time_display_str)

            if status_bool:
                normal_count += 1
            else:
                defect_count += 1

            self.get_logger().info(f'🎯 나사 {screw_idx} 결과 -> {"✅ 정상" if status_bool else "❌ 불량(돌출)"} (단차: {delta_z_mm:.1f}mm)')
            screw_idx += 1

        self.get_logger().info('☁️ 3D 포인트 클라우드 배경 맵핑 중...')
        pc_data = list(pc2.read_points(pc_msg, field_names=('x', 'y', 'z', 'rgb'), skip_nans=True))
        pts = np.array([[p[0], p[1], p[2]] for p in pc_data])
        colors = np.array([
            [
                (struct.unpack('I', struct.pack('f', p[3]))[0] >> 16 & 0xFF) / 255.0,
                (struct.unpack('I', struct.pack('f', p[3]))[0] >> 8 & 0xFF) / 255.0,
                (struct.unpack('I', struct.pack('f', p[3]))[0] & 0xFF) / 255.0,
            ]
            for p in pc_data
        ])

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

            capture_data = {
                'metadata': {
                    'capture_id': capture_id,
                    'session_id': self.session_id,
                    'company_id': settings.COMPANY_ID,
                    'company_name': settings.COMPANY_NAME,
                    'site_id': settings.SITE_ID,
                    'site_name': settings.SITE_NAME,
                    'line_id': settings.LINE_ID,
                    'line_name': settings.LINE_NAME,
                    'workstation_id': workstation_id,
                    'workstation_name': section_name,
                    'robot_id': settings.ROBOT_ID,
                    'camera_id': settings.CAMERA_ID,
                    'schema_version': '1.0.0',
                    'created_at': now_ms,
                    'timestamp': timestamp_str,
                    'time_display': time_display_str,
                },
                'summary': {
                    'total_count': int(screw_idx),
                    'normal_count': int(normal_count),
                    'defect_count': int(defect_count),
                    'inspection_status': 'completed',
                },
                'markers': markers_data,
                'transform_snapshot': {
                    'pointcloud_frame_to_base': {
                        'source_frame': pc_msg.header.frame_id,
                        'target_frame': settings.BASE_FRAME,
                        'matrix_4x4': np.round(tf_pc, 8).tolist(),
                    },
                    'color_frame_to_base': {
                        'source_frame': img_msg.header.frame_id,
                        'target_frame': settings.BASE_FRAME,
                        'matrix_4x4': np.round(tf_color, 8).tolist(),
                    },
                },
                'storage_refs': {
                    'background_js': blob_path,
                    'background_url': public_url,
                },
            }

            get_db_reference(
                db_paths.session_capture_path(
                    self.session_id,
                    workstation_id,
                    capture_id,
                )
            ).set(capture_data)

            # 기존 GUI/API/로봇 코드 호환용 mirror 저장
            self.inspections_ref.child(capture_id).set(capture_data)

            self.twin_state_ref.child('current_inspection').set({
                'session_id': self.session_id,
                'workstation_id': workstation_id,
                'capture_id': capture_id,
                'status': 'completed',
                'total_count': int(screw_idx),
                'normal_count': int(normal_count),
                'defect_count': int(defect_count),
                'updated_at': now_ms,
            })

            self.twin_state_ref.child('current_session').update({
                'session_id': self.session_id,
                'status': 'running',
                'latest_capture_id': capture_id,
                'latest_workstation_id': workstation_id,
                'total_workstations': self.capture_count,
                'total_captures': self.session_total_captures,
                'total_markers': self.session_total_markers,
                'normal_count': self.session_normal_count,
                'defect_count': self.session_defect_count,
                'updated_at': now_ms,
            })

            get_db_reference(
                db_paths.inspection_session_path(self.session_id)
            ).child('metadata').update({
                'session_id': self.session_id,
                'company_id': settings.COMPANY_ID,
                'company_name': settings.COMPANY_NAME,
                'site_id': settings.SITE_ID,
                'site_name': settings.SITE_NAME,
                'robot_id': settings.ROBOT_ID,
                'status': 'running',
                'started_at': now_ms,
                'updated_at': now_ms,
                'schema_version': '1.0.0',
            })

            get_db_reference(
                db_paths.inspection_session_path(self.session_id)
            ).child('summary').update({
                'latest_capture_id': capture_id,
                'latest_workstation_id': workstation_id,
                'total_workstations': self.capture_count,
                'updated_at': now_ms,
            })

            self.twin_state_ref.child('robots').child(settings.ROBOT_ID).update({
                'status': 'ready',
                'mode': 'inspection_completed',
                'current_session_id': self.session_id,
                'current_workstation_id': workstation_id,
                'current_capture_id': capture_id,
                'updated_at': now_ms,
            })

            self.site_ref.update({
                'latest_session_id': self.session_id,
                'latest_capture_id': capture_id,
                'updated_at': now_ms,
            })

            self.robot_ref.update({
                'status': 'ready',
                'current_session_id': self.session_id,
                'current_workstation_id': workstation_id,
                'current_capture_id': capture_id,
                'updated_at': now_ms,
            })


            index_base = f"indexes/{settings.SITE_ID}"
            capture_index_data = {
                'session_id': self.session_id,
                'workstation_id': workstation_id,
                'capture_id': capture_id,
                'created_at': now_ms,
                'total_count': int(screw_idx),
                'normal_count': int(normal_count),
                'defect_count': int(defect_count),
            }

            get_db_reference(f"{index_base}/latest").update({
                'latest_session_id': self.session_id,
                'latest_workstation_id': workstation_id,
                'latest_capture_id': capture_id,
                'updated_at': now_ms,
            })

            get_db_reference(
                db_paths.index_capture_lookup_path(capture_id)
            ).set(capture_index_data)

            get_db_reference(
                db_paths.index_captures_by_date_path(timestamp_str[:8], capture_id)
            ).set(capture_index_data)

            get_db_reference(
                db_paths.index_captures_by_workstation_path(workstation_id, capture_id)
            ).set(capture_index_data)

            for marker_id, marker in markers_data.items():
                if marker.get('status') in ['defect', 'failed']:
                    get_db_reference(
                        db_paths.index_unresolved_defect_path(f"{capture_id}_{marker_id}")
                    ).set({
                        'session_id': self.session_id,
                        'workstation_id': workstation_id,
                        'capture_id': capture_id,
                        'marker_id': marker_id,
                        'status': marker.get('status'),
                        'defect_type': marker.get('defect_type'),
                        'delta_z_mm': marker.get('delta_z_mm'),
                        'created_at': now_ms,
                    })

            self.legacy_ref.child(section_name).set({
                'capture_index': self.capture_count,
                'capture_id': capture_id,
                'timestamp': timestamp_str,
                'background_url': public_url,
                'markers': legacy_markers_data,
            })

            get_db_reference(db_paths.events_path()).child(db_paths.now_event_id('inspection_completed')).set({
                'event_type': 'inspection_completed',
                'company_id': settings.COMPANY_ID,
                'site_id': settings.SITE_ID,
                'robot_id': settings.ROBOT_ID,
                'capture_id': capture_id,
                'severity': 'info' if defect_count == 0 else 'warning',
                'summary': {
                    'total_count': int(screw_idx),
                    'normal_count': int(normal_count),
                    'defect_count': int(defect_count),
                },
                'created_at': now_ms,
                'schema_version': '1.0.0',
            })

            self.get_logger().info(f'🎉 [{section_name}] 데이터 저장 완료! capture_id={capture_id}, 총 {screw_idx}개 나사 감지됨')
        else:
            self.get_logger().warn('⚠️ 데이터 업로드 및 DB 동기화에 실패했습니다.')

        if os.path.exists(js_filename):
            os.remove(js_filename)


def main(args=None):
    rclpy.init(args=args)
    node = Yolo3DMapperNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
