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
import cv2
from cv_bridge import CvBridge
from ultralytics import YOLO
import os
from datetime import datetime

# Firebase 관련 라이브러리
import firebase_admin
from firebase_admin import credentials, storage, db

SERVICE_ACCOUNT_KEY_PATH = '/home/yoon/YJH/resource/rokey-d-2-4c32a-firebase-adminsdk-fbsvc-3c31d8ba34.json'
STORAGE_BUCKET_NAME = 'rokey-d-2-4c32a.firebasestorage.app'
DATABASE_URL = 'https://rokey-d-2-4c32a-default-rtdb.asia-southeast1.firebasedatabase.app'

class Yolo3DNormalNode(Node):
    def __init__(self):
        super().__init__('yolo_3d_normal_node')

        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        self.publish_static_transform()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.bridge = CvBridge()
        self.model_path = '/home/yoon/YJH/resource/hyupdong2_yolo11x_img960_best.pt'
        self.yolo_model = None  # 초기에는 None으로 설정
        
        # 모델 로딩을 위한 타이머 생성 (노드 시작 후 0.1초 뒤 실행)
        self.load_timer = self.create_timer(0.1, self.load_model_timer_callback)

        # Firebase 초기화
        try:
            cred = credentials.Certificate(SERVICE_ACCOUNT_KEY_PATH)
            if not firebase_admin._apps:
                firebase_admin.initialize_app(cred, {'storageBucket': STORAGE_BUCKET_NAME, 'databaseURL': DATABASE_URL})
            self.bucket = storage.bucket()
            self.db_ref = db.reference('linestatus')
        except Exception as e:
            self.get_logger().error(f'Firebase 초기화 실패: {e}')
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

    def load_model_timer_callback(self):
        """모델을 비동기적으로 로드하기 위한 타이머 콜백"""
        self.get_logger().info('YOLO 모델 로딩 시작...')
        try:
            self.yolo_model = YOLO(self.model_path)
            self.get_logger().info('YOLO 모델 로딩 완료!')
        except Exception as e:
            self.get_logger().error(f'모델 로딩 실패: {e}')
        
        # 로딩 작업은 1회만 수행하므로 타이머 종료
        self.load_timer.cancel()

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
        z = float(depth_map[v, u]) / 1000.0
        if z <= 0.1 or z > 2.0: return None
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        pt_cam = np.array([x, y, z, 1.0])
        pt_base = tf_matrix @ pt_cam
        return pt_base[:3]

    def get_stable_point(self, u, v, depth_map, fx, fy, cx, cy, tf_matrix):
        pts = []
        for du, dv in [(0,0), (1,0), (0,1), (1,1)]:
            pt = self.get_3d_point(u + du, v + dv, depth_map, fx, fy, cx, cy, tf_matrix)
            if pt is not None: pts.append(pt)
        if not pts: return None
        return np.mean(pts, axis=0)

    def calculate_normal_vector(self, p1, p2, p3):
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm == 0: return np.array([0.0, 0.0, 1.0])
        return normal / norm

    def calculate_normal_from_points(self, points):
        """여러 점으로부터 SVD 기반 평면 피팅으로 법선 벡터 계산"""
        pts = np.array(points)
        centroid = pts.mean(axis=0)
        centered = pts - centroid
        # SVD: 가장 작은 singular value에 해당하는 벡터가 법선
        _, _, Vt = np.linalg.svd(centered)
        normal = Vt[-1]  # 마지막 행 = 최소 분산 방향 = 법선
        # 법선이 z+ 방향(위쪽)을 가리키도록 보정
        if normal[2] < 0:
            normal = -normal
        return normal

    def is_inside_any_bbox(self, u, v, bboxes, margin=20):
        """픽셀 (u,v)가 어떤 바운딩 박스 안에 있는지 확인 (margin 포함)"""
        for (x1, y1, x2, y2) in bboxes:
            if x1 - margin <= u <= x2 + margin and y1 - margin <= v <= y2 + margin:
                return True
        return False

    def get_surface_point(self, u, v, depth_map, fx, fy, cx, cy, tf_matrix, radius=5):
        """주변 (2*radius+1)x(2*radius+1) 영역을 평균내어 안정적인 표면 3D 좌표 반환"""
        h, w = depth_map.shape[:2]
        pts = []
        for du in range(-radius, radius + 1, 2):
            for dv in range(-radius, radius + 1, 2):
                nu, nv = u + du, v + dv
                if 0 <= nu < w and 0 <= nv < h:
                    pt = self.get_3d_point(nu, nv, depth_map, fx, fy, cx, cy, tf_matrix)
                    if pt is not None:
                        pts.append(pt)
        if not pts:
            return None
        return np.mean(pts, axis=0)

    def get_screw_normal(self, bbox, depth_img, fx, fy, cx, cy, tf_mat):
        """개별 나사 주변 표면에서 L자 형태 3점으로 법선 벡터 계산
        
        이미지 픽셀 좌표 기준:
            b ───B───→ c     (B = c - b, 오른쪽)
            │
            A
            │
            ↓
            a                (A = a - b, 아래쪽)
        
        법선 = A × B → 항상 카메라를 향하는 방향
        """
        x1, y1, x2, y2 = bbox
        h, w = depth_img.shape[:2]
        offset = 10  # bbox 바깥으로 10px

        # b = bbox 좌상단 바깥, a = bbox 좌하단 바깥, c = bbox 우상단 바깥
        b_pixel = (x1 - offset, y1 - offset)   # 좌상단
        a_pixel = (x1 - offset, y2 + offset)   # 좌하단
        c_pixel = (x2 + offset, y1 - offset)   # 우상단

        # 이미지 범위 체크
        for pu, pv in [b_pixel, a_pixel, c_pixel]:
            if pu < 5 or pu >= w - 5 or pv < 5 or pv >= h - 5:
                return None

        # 3D 좌표 변환 (주변 영역 평균으로 안정화)
        b_3d = self.get_surface_point(b_pixel[0], b_pixel[1], depth_img, fx, fy, cx, cy, tf_mat, radius=3)
        a_3d = self.get_surface_point(a_pixel[0], a_pixel[1], depth_img, fx, fy, cx, cy, tf_mat, radius=3)
        c_3d = self.get_surface_point(c_pixel[0], c_pixel[1], depth_img, fx, fy, cx, cy, tf_mat, radius=3)

        if any(p is None for p in [b_3d, a_3d, c_3d]):
            return None

        # A = a - b (이미지에서 아래 방향), B = c - b (이미지에서 오른쪽 방향)
        A = a_3d - b_3d
        B = c_3d - b_3d
        normal = np.cross(A, B)
        norm = np.linalg.norm(normal)
        if norm == 0:
            return None
        return normal / norm

    def process_capture(self):
        # 1. 모델 로드 여부 확인
        if self.yolo_model is None:
            self.get_logger().warn('모델이 아직 로딩 중입니다. 잠시만 기다려주세요.')
            return

        if not all([self.latest_depth_msg, self.cam_info, self.latest_img_msg]):
            self.get_logger().warn('센서 데이터가 준비되지 않았습니다.')
            return

        depth_img = self.bridge.imgmsg_to_cv2(self.latest_depth_msg, desired_encoding='passthrough')
        cam_info = self.cam_info
        fx, fy, cx, cy = cam_info.k[0], cam_info.k[4], cam_info.k[2], cam_info.k[5]

        tf_mat = self.get_tf_matrix('base_link', self.latest_depth_msg.header.frame_id)
        if tf_mat is None:
            self.get_logger().warn('TF 변환을 가져올 수 없습니다.')
            return

        # YOLO 추론
        img = self.bridge.imgmsg_to_cv2(self.latest_img_msg, desired_encoding='bgr8')
        results = self.yolo_model(img)[0]

        # 바운딩 박스 목록 수집
        bboxes = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            bboxes.append((x1, y1, x2, y2))

        self.get_logger().info(f'검출된 나사 수: {len(bboxes)}')

        if len(bboxes) == 0:
            self.get_logger().warn('나사가 검출되지 않았습니다.')
            return

        # 각 나사별로 법선 벡터 계산
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox
            cx_img, cy_img = (x1 + x2) // 2, (y1 + y2) // 2

            normal = self.get_screw_normal(bbox, depth_img, fx, fy, cx, cy, tf_mat)
            if normal is not None:
                self.get_logger().info(
                    f'나사 #{i+1} bbox=({x1},{y1})-({x2},{y2}) '
                    f'법선벡터: [{normal[0]:.4f}, {normal[1]:.4f}, {normal[2]:.4f}]'
                )
            else:
                self.get_logger().warn(f'나사 #{i+1} bbox=({x1},{y1})-({x2},{y2}) 법선벡터 계산 실패 (주변 샘플 부족)')

    def keyboard_listener(self):
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while rclpy.ok():
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    if sys.stdin.read(1) == 's': self.process_capture()
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
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()