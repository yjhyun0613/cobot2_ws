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
import math
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
        self.yolo_model = None
        
        self.load_timer = self.create_timer(0.1, self.load_model_timer_callback)

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
        self.get_logger().info('YOLO 모델 로딩 시작...')
        try:
            self.yolo_model = YOLO(self.model_path)
            self.get_logger().info('YOLO 모델 로딩 완료!')
        except Exception as e:
            self.get_logger().error(f'모델 로딩 실패: {e}')
        self.load_timer.cancel()

    def pc_callback(self, msg): self.latest_pc_msg = msg
    def img_callback(self, msg): self.latest_img_msg = msg
    def depth_callback(self, msg): self.latest_depth_msg = msg
    def info_callback(self, msg): self.cam_info = msg

    # =================================================================
    # [수학 및 기하학 보조 함수 (motion_04에서 이식됨)]
    # =================================================================
    def normalize_vec(self, v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-6 else v

    def rot_to_zyz(self, R):
        beta = math.acos(max(min(R[2, 2], 1.0), -1.0))
        if abs(beta) < 1e-6:
            alpha, gamma = 0.0, math.atan2(R[1, 0], R[0, 0])
        else:
            alpha = math.atan2(R[1, 2], R[0, 2])
            gamma = math.atan2(R[2, 1], -R[2, 0])
        return [math.degrees(alpha), math.degrees(beta), math.degrees(gamma)]

    def wrap_angle(self, angle):
        while angle > 180.0: angle -= 360.0
        while angle < -180.0: angle += 360.0
        return angle

    def calculate_target_pose(self, normal_vec):
        """법선 벡터를 기반으로 로봇 TCP가 나사를 정면으로 바라보는 자세(rx, ry, rz) 계산"""
        if np.linalg.norm(normal_vec) < 1e-6:
            return None
            
        n = self.normalize_vec(normal_vec)
        # 툴의 Z축이 나사 방향(-n)을 향하도록 설정
        z_axis_final = -n
        
        # 툴의 Y축이 항상 글로벌 Y축을 바라보도록 설정하여 손목 꼬임 방지
        global_y = np.array([0.0, 1.0, 0.0])
        x_axis = np.cross(global_y, z_axis_final)
        
        if np.linalg.norm(x_axis) < 1e-6:
            global_x = np.array([1.0, 0.0, 0.0])
            y_axis = np.cross(z_axis_final, global_x)
            y_axis = self.normalize_vec(y_axis)
            x_axis = self.normalize_vec(np.cross(y_axis, z_axis_final))
        else:
            x_axis = self.normalize_vec(x_axis)
            y_axis = self.normalize_vec(np.cross(z_axis_final, x_axis))
            
        rot_matrix = np.column_stack((x_axis, y_axis, z_axis_final))
        rx, ry, rz = self.rot_to_zyz(rot_matrix)
        
        return self.wrap_angle(rx), self.wrap_angle(ry), self.wrap_angle(rz)

    # =================================================================
    # [비전 처리 및 3D 계산 로직]
    # =================================================================
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

    def get_surface_point(self, u, v, depth_map, fx, fy, cx, cy, tf_matrix, radius=5):
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
        x1, y1, x2, y2 = bbox
        h, w = depth_img.shape[:2]
        offset = 10 

        b_pixel = (x1 - offset, y1 - offset)
        a_pixel = (x1 - offset, y2 + offset)
        c_pixel = (x2 + offset, y1 - offset)

        for pu, pv in [b_pixel, a_pixel, c_pixel]:
            if pu < 5 or pu >= w - 5 or pv < 5 or pv >= h - 5:
                return None

        b_3d = self.get_surface_point(b_pixel[0], b_pixel[1], depth_img, fx, fy, cx, cy, tf_mat, radius=3)
        a_3d = self.get_surface_point(a_pixel[0], a_pixel[1], depth_img, fx, fy, cx, cy, tf_mat, radius=3)
        c_3d = self.get_surface_point(c_pixel[0], c_pixel[1], depth_img, fx, fy, cx, cy, tf_mat, radius=3)

        if any(p is None for p in [b_3d, a_3d, c_3d]):
            return None

        A = a_3d - b_3d
        B = c_3d - b_3d
        normal = np.cross(A, B)
        norm = np.linalg.norm(normal)
        if norm == 0:
            return None
        return normal / norm

    def process_capture(self):
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

        img = self.bridge.imgmsg_to_cv2(self.latest_img_msg, desired_encoding='bgr8')
        results = self.yolo_model(img)[0]

        bboxes = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            bboxes.append((x1, y1, x2, y2))

        self.get_logger().info(f'검출된 나사 수: {len(bboxes)}')

        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox
            cx_img, cy_img = (x1 + x2) // 2, (y1 + y2) // 2

            # 1. 나사의 중심 3D 좌표 추출 (단위: m)
            center_3d = self.get_surface_point(cx_img, cy_img, depth_img, fx, fy, cx, cy, tf_mat, radius=3)
            # 2. 나사 주변 평면의 법선 벡터 계산
            normal = self.get_screw_normal(bbox, depth_img, fx, fy, cx, cy, tf_mat)

            if normal is not None and center_3d is not None:
                # 3. 로봇 자세 계산 (ZYZ 오일러 각도)
                pose_angles = self.calculate_target_pose(normal)
                
                if pose_angles:
                    rx, ry, rz = pose_angles
                    # 로봇 제어 규격(posx)에 맞게 단위 변환 (m -> mm)
                    target_x = center_3d[0] * 1000.0
                    target_y = center_3d[1] * 1000.0
                    target_z = center_3d[2] * 1000.0

                    # 최종 검증용 로그 출력
                    self.get_logger().info(
                        f'🎯 나사 #{i+1} 타겟 정보 -> '
                        f'위치(mm): [{target_x:.1f}, {target_y:.1f}, {target_z:.1f}], '
                        f'자세(posx): [{rx:.1f}, {ry:.1f}, {rz:.1f}]'
                    )
            else:
                self.get_logger().warn(f'나사 #{i+1} 3D 데이터 또는 법선 벡터 계산 실패')

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