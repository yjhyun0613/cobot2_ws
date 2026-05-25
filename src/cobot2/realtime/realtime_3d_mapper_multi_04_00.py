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
from datetime import datetime

# --- Firebase 라이브러리 추가 ---
import firebase_admin
from firebase_admin import credentials, storage, db

# --- Firebase 설정 변수 ---
SERVICE_ACCOUNT_KEY_PATH = '/home/yoon/YJH/resource/rokey-d-2-4c32a-firebase-adminsdk-fbsvc-3c31d8ba34.json'
STORAGE_BUCKET_NAME = 'rokey-d-2-4c32a.firebasestorage.app'
DATABASE_URL = 'https://rokey-d-2-4c32a-default-rtdb.asia-southeast1.firebasedatabase.app'

class Yolo3DMapperNode(Node):
    def __init__(self):
        super().__init__('yolo_3d_mapper_node')

        # 1. TF 설정
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        self.publish_static_transform()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # 2. YOLO 및 OpenCV 설정
        self.bridge = CvBridge()
        model_path = '/home/yoon/YJH/resource/hyupdong2_yolo11x_img960_best.pt'
        self.get_logger().info(f'YOLO 모델 로딩 중... ({model_path})')
        self.yolo_model = YOLO(model_path)
        self.get_logger().info('✅ YOLO 모델 로딩 완료!')
        
        # --- 2-1. Firebase 초기화 ---
        try:
            cred = credentials.Certificate(SERVICE_ACCOUNT_KEY_PATH)
            if not firebase_admin._apps:
                firebase_admin.initialize_app(cred, {
                    'storageBucket': STORAGE_BUCKET_NAME,
                    'databaseURL': DATABASE_URL
                })
            self.bucket = storage.bucket()
            self.db_ref = db.reference('linestatus')
            self.get_logger().info('✅ Firebase Storage & Realtime Database 연결 완료!')
            
            self.init_capture_count()
        except Exception as e:
            self.get_logger().error(f'❌ Firebase 초기화 실패: {e}')
            self.bucket = None
            self.db_ref = None
            self.capture_count = 0

        # 3. 데이터 구독
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

    def init_capture_count(self):
        try:
            snapshot = self.db_ref.get()
            if snapshot:
                self.capture_count = len(snapshot.keys())
                self.get_logger().info(f'🔄 기존 데이터 확인됨. 다음 작업은 "작업대 {self.capture_count + 1}"로 저장됩니다.')
                return
            self.capture_count = 0
        except Exception as e:
            self.get_logger().error(f'기존 회차 조회 실패로 1부터 시작합니다: {e}')
            self.capture_count = 0

    def publish_static_transform(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'link_6'
        t.child_frame_id = 'camera_link'
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

    def keyboard_listener(self):
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while rclpy.ok():
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)
                    if key.lower() == 's':
                        # 터미널 스위칭(입력 대기) 제거 -> 누르자마자 즉각 실행
                        self.process_capture()
                    elif key == '\x03':
                        rclpy.shutdown()
                        break
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def get_tf_matrix(self, target_frame, source_frame):
        try:
            trans = self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0))
            rot = R.from_quat([trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w]).as_matrix()
            mat = np.eye(4)
            mat[:3, :3] = rot
            mat[:3, 3] = [trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z]
            return mat
        except: return None
        
    def upload_to_firebase_storage(self, file_path, destination_blob_name, content_type):
        if not self.bucket: return None
        try:
            blob = self.bucket.blob(destination_blob_name)
            blob.upload_from_filename(file_path, content_type=content_type)
            blob.make_public()
            return blob.public_url
        except Exception as e:
            self.get_logger().error(f'❌ Storage 업로드 실패: {e}')
            return None

    def process_capture(self):
        if not all([self.latest_pc_msg, self.latest_img_msg, self.latest_depth_msg, self.cam_info]):
            self.get_logger().warn('카메라 데이터를 받지 못했습니다. 잠시 후 다시 시도해주세요.')
            return

        self.capture_count += 1
        # 🌟 자동으로 "작업대 1", "작업대 2" 형태로 이름 지정
        section_name = f"작업대 {self.capture_count}"
        
        self.get_logger().info(f'\n{"="*50}\n📸 [검사 진행] {section_name} 영역 스캔 및 분석 시작...\n{"="*50}')

        pc_msg, img_msg, depth_msg, cam_info = self.latest_pc_msg, self.latest_img_msg, self.latest_depth_msg, self.cam_info
        
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        time_display_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        tf_pc = self.get_tf_matrix('base_link', pc_msg.header.frame_id)
        tf_color = self.get_tf_matrix('base_link', img_msg.header.frame_id)
        
        if tf_pc is None or tf_color is None:
            self.get_logger().error('TF 좌표 변환에 실패했습니다. 로봇 연결 상태를 확인하세요.')
            return
            
        fx, fy, cx, cy = cam_info.k[0], cam_info.k[4], cam_info.k[2], cam_info.k[5]
        cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        img_h, img_w = cv_image.shape[:2]
        cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        
        yolo_results = self.yolo_model(cv_image, verbose=False)[0]
        markers_data = {}

        # 1. 나사 돌출 검사 및 구조화 (나사 0, 1, 2... 자동 넘버링)
        screw_idx = 0
        for box in yolo_results.boxes:
            b = box.xyxy[0].cpu().numpy()
            xmin, ymin, xmax, ymax = b[0], b[1], b[2], b[3]
            
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
                            if abs(dx) < bw_img * 0.1 and abs(dy) < bh_img * 0.1: screw_pts.append(z_mm)
                            elif abs(dx) > bw_img * 0.5 or abs(dy) > bh_img * 0.5: surf_pts.append(z_mm)

            if not screw_pts or not surf_pts: continue
            
            z_screw = min(screw_pts) 
            cam_z = z_screw / 1000.0
            
            if cam_z < 0.25 or cam_z > 1.0: continue

            z_surf = np.median(surf_pts) 
            delta_z_mm = z_surf - z_screw
            is_defective = delta_z_mm > 10.0  # 단차 10mm 기준 불량 판별
            
            status_bool = not is_defective
            pt_base = tf_color @ np.array([(cx_img - cx) * cam_z / fx, (cy_img - cy) * cam_z / fy, cam_z, 1.0])
            
            # YOLO 탐지 순서대로 나사 번호(0, 1, 2...) 부여
            markers_data[str(screw_idx)] = {
                "status": status_bool,
                "position": {
                    "x": float(pt_base[0]),
                    "y": float(pt_base[1]),
                    "z": float(pt_base[2])
                },
                "time": time_display_str
            }
            self.get_logger().info(f'🎯 나사 {screw_idx} 결과 -> {"✅ 정상" if status_bool else "❌ 불량(돌출)"} (단차: {delta_z_mm:.1f}mm)')
            screw_idx += 1

        # 2. 배경 포인트 클라우드를 .js 스크립트 파일 형태로 변환 및 로컬 저장
        self.get_logger().info('☁️ 3D 포인트 클라우드 배경 맵핑 중...')
        pc_data = list(pc2.read_points(pc_msg, field_names=("x", "y", "z", "rgb"), skip_nans=True))
        pts = np.array([[p[0], p[1], p[2]] for p in pc_data])
        colors = np.array([[(struct.unpack('I', struct.pack('f', p[3]))[0] >> 16 & 0xFF)/255.0, 
                            (struct.unpack('I', struct.pack('f', p[3]))[0] >> 8 & 0xFF)/255.0, 
                            (struct.unpack('I', struct.pack('f', p[3]))[0] & 0xFF)/255.0] for p in pc_data])
        
        distances = np.linalg.norm(pts, axis=1)
        mask = (distances >= 0.25) & (distances <= 1.0)
        
        pts_filtered, colors_filtered = pts[mask], colors[mask]
        pts_transformed = (tf_pc @ np.hstack([pts_filtered, np.ones((len(pts_filtered), 1))]).T).T[:, :3]
        
        bg_dict = {
            "x": np.round(pts_transformed[:, 0], 4).tolist(),
            "y": np.round(pts_transformed[:, 1], 4).tolist(),
            "z": np.round(pts_transformed[:, 2], 4).tolist(),
            "colors": [f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' for c in colors_filtered]
        }

        js_filename = f'bg_{timestamp_str}.js'
        with open(js_filename, 'w') as f:
            f.write(f"window.latestBackground = {json.dumps(bg_dict)};")

        # 3. Firebase 업로드 및 실시간 DB 구조 반영
        self.get_logger().info('🚀 Firebase 실시간 데이터베이스 동기화 중...')
        blob_path = f'backgrounds/{js_filename}'
        public_url = self.upload_to_firebase_storage(js_filename, blob_path, 'application/javascript')

        if public_url and self.db_ref:
            # 🌟 수동 입력 대신, 자동 생성된 section_name ("작업대 1", "작업대 2"...)으로 DB에 덮어쓰기
            self.db_ref.child(section_name).set({
                "capture_index": self.capture_count,
                "timestamp": timestamp_str,
                "background_url": public_url,
                "markers": markers_data
            })
            self.get_logger().info(f'🎉 [{section_name}] 데이터 저장 완료! (총 {screw_idx}개 나사 감지됨)')
        else:
            self.get_logger().warn('⚠️ 데이터 업로드 및 DB 동기화에 실패했습니다.')

        if os.path.exists(js_filename): os.remove(js_filename)

def main(args=None):
    rclpy.init(args=args)
    node = Yolo3DMapperNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()