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

# Firebase 관련 라이브러리
import firebase_admin
from firebase_admin import credentials, storage, db

SERVICE_ACCOUNT_KEY_PATH = '/home/rokey/yjh/rokey-d-2-4c32a-firebase-adminsdk-fbsvc-7f5d874f48.json'
STORAGE_BUCKET_NAME = 'rokey-d-2-4c32a.firebasestorage.app'
DATABASE_URL = 'https://rokey-d-2-4c32a-default-rtdb.asia-southeast1.firebasedatabase.app'
MODEL_PATH = '/home/rokey/yjh/hyupdong2_yolo11x_img960_best.pt'

class Yolo3DNormalNode(Node):
    def __init__(self):
        super().__init__('yolo_3d_normal_node')

        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        self.publish_static_transform()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.bridge = CvBridge()
        
        # YOLO 모델 상태 변수 및 백그라운드 로딩 스레드 시작
        self.yolo_model = None
        self.is_model_loaded = False
        self.get_logger().info('⏳ YOLO 모델을 백그라운드에서 로딩합니다. 카메라 데이터 수신은 정상 진행됩니다.')
        
        self.model_thread = threading.Thread(target=self.load_yolo_model)
        self.model_thread.daemon = True
        self.model_thread.start()
        
        # Firebase 초기화
        try:
            cred = credentials.Certificate(SERVICE_ACCOUNT_KEY_PATH)
            if not firebase_admin._apps:
                firebase_admin.initialize_app(cred, {'storageBucket': STORAGE_BUCKET_NAME, 'databaseURL': DATABASE_URL})
            self.bucket = storage.bucket()
            self.db_ref = db.reference('linestatus')
        except Exception as e:
            self.get_logger().error(f'❌ Firebase 초기화 실패: {e}')
            self.bucket = None

        self.sub_pc = self.create_subscription(PointCloud2, '/camera/camera/depth/color/points', self.pc_callback, 10)
        self.sub_img = self.create_subscription(Image, '/camera/camera/color/image_raw', self.img_callback, 10)
        self.sub_depth = self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        self.sub_info = self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.info_callback, 10)

        self.latest_pc_msg = None
        self.latest_img_msg = None
        self.latest_depth_msg = None
        self.cam_info = None

        self.key_thread = threading.Thread(target=self.keyboard_listener)
        self.key_thread.daemon = True
        self.key_thread.start()

    def load_yolo_model(self):
        try:
            self.yolo_model = YOLO(MODEL_PATH)
            self.is_model_loaded = True
            self.get_logger().info('✅ YOLO 모델 로딩 완료! 이제 "s" 키를 눌러 검사를 진행할 수 있습니다.')
        except Exception as e:
            self.get_logger().error(f'❌ YOLO 모델 로딩 중 에러 발생: {e}')

    def pc_callback(self, msg): self.latest_pc_msg = msg
    def img_callback(self, msg): self.latest_img_msg = msg
    def depth_callback(self, msg): self.latest_depth_msg = msg
    def info_callback(self, msg): self.cam_info = msg

    def get_tf_matrix(self, target_frame, source_frame):
        try:
            trans = self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0))
            rot = R.from_quat([trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w]).as_matrix()
            mat = np.eye(4)
            mat[:3, :3] = rot
            mat[:3, 3] = [trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z]
            return mat
        except: return None

    def get_3d_point(self, u, v, depth_map, fx, fy, cx, cy, tf_matrix):
        """특정 픽셀(u, v)의 3D 좌표를 base_link 기준으로 반환"""
        h, w = depth_map.shape
        if v < 0 or v >= h or u < 0 or u >= w:
            return None

        z = float(depth_map[v, u]) / 1000.0
        if z <= 0.1 or z > 2.0: return None
        
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        pt_cam = np.array([x, y, z, 1.0])
        pt_base = tf_matrix @ pt_cam
        return pt_base[:3]

    def get_stable_point(self, u, v, depth_map, fx, fy, cx, cy, tf_matrix):
        """주변 5x5 픽셀을 평균내어 안정적인 3D 좌표 반환 (결측치 방어)"""
        pts = []
        # 주변 5x5 픽셀 탐색 (노이즈에 훨씬 강해짐)
        for dv in range(-2, 3):
            for du in range(-2, 3):
                pt = self.get_3d_point(u + du, v + dv, depth_map, fx, fy, cx, cy, tf_matrix)
                if pt is not None: 
                    pts.append(pt)
        
        if not pts: return None
        return np.mean(pts, axis=0)

    def calculate_normal_vector(self, p1, p2, p3):
        """3개의 점으로 평면의 법선 벡터 계산"""
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm == 0: return np.array([0.0, 0.0, 1.0])
        return normal / norm

    def process_capture(self):
        if not self.is_model_loaded:
            self.get_logger().warn('⚠️ YOLO 모델이 아직 로딩 중입니다. 완료 메시지가 뜰 때까지 잠시만 기다려주세요.')
            return

        if not all([self.latest_depth_msg, self.latest_img_msg, self.cam_info]):
            self.get_logger().warn('⚠️ 카메라 데이터가 아직 충분히 수신되지 않았습니다.')
            return

        depth_img = self.bridge.imgmsg_to_cv2(self.latest_depth_msg, desired_encoding='passthrough')
        cam_info = self.cam_info
        fx, fy, cx, cy = cam_info.k[0], cam_info.k[4], cam_info.k[2], cam_info.k[5]
        
        tf_mat = self.get_tf_matrix('base_link', self.latest_depth_msg.header.frame_id)
        if tf_mat is None: 
            self.get_logger().warn('⚠️ TF 변환 매트릭스를 구할 수 없습니다.')
            return

        # YOLO 추론
        img = self.bridge.imgmsg_to_cv2(self.latest_img_msg, desired_encoding='bgr8')
        results = self.yolo_model(img, verbose=False)[0]

        if len(results.boxes) == 0:
            self.get_logger().info('객체가 화면에 없습니다.')

        for i, box in enumerate(results.boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cx_img, cy_img = int((x1 + x2) / 2), int((y1 + y2) / 2)
            
            # 여백 설정: 바운딩 박스에서 30픽셀만큼 완전히 떨어트림
            margin = 30
            
            # 1. 샘플링 지점 3개 설정 (바운딩 박스 외부)
            # p_left: 박스 왼쪽 바깥, p_right: 박스 오른쪽 바깥, p_top: 박스 위쪽 바깥
            p_left = self.get_stable_point(x1 - margin, cy_img, depth_img, fx, fy, cx, cy, tf_mat)
            p_right = self.get_stable_point(x2 + margin, cy_img, depth_img, fx, fy, cx, cy, tf_mat)
            p_top = self.get_stable_point(cx_img, y1 - margin, depth_img, fx, fy, cx, cy, tf_mat)

            if p_left is not None and p_right is not None and p_top is not None:
                # 2. 법선 벡터 계산
                normal = self.calculate_normal_vector(p_left, p_right, p_top)
                self.get_logger().info(f'🎯 객체 [{i+1}] 주변 배경 평면! -> 법선 벡터(Normal): {np.round(normal, 3)}')
            else:
                self.get_logger().warn(f'⚠️ 객체 [{i+1}] 바운딩 박스 외부 영역의 깊이 데이터가 부족합니다. (화면 밖이거나 측정 불가 지역)')

    def keyboard_listener(self):
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while rclpy.ok():
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)
                    if key.lower() == 's': 
                        self.process_capture()
                    elif key == '\x03': # Ctrl+C
                        rclpy.shutdown()
                        break
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

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

def main(args=None):
    rclpy.init(args=args)
    node = Yolo3DNormalNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()