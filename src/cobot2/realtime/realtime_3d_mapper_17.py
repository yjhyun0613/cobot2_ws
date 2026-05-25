import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
import sensor_msgs_py.point_cloud2 as pc2
from tf2_ros import Buffer, TransformListener, StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
import open3d as o3d
import numpy as np
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R
import threading
import sys
import termios
import tty
import select
import struct
import cv2
import math
from cv_bridge import CvBridge
from ultralytics import YOLO
import os

# --- Firebase 라이브러리 추가 ---
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage

# --- Firebase 설정 변수 (본인 환경에 맞게 수정!) ---
SERVICE_ACCOUNT_KEY_PATH = '/home/rokey/yjh/rokey-d-2-4c32a-firebase-adminsdk-fbsvc-7f5d874f48.json'  # 다운받은 JSON 키 파일 경로
STORAGE_BUCKET_NAME = 'rokey-d-2-4c32a.firebasestorage.app'         # gs:// 뒷부분 주소

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
        model_path = '/home/rokey/yjh/hyupdong2_final_clean_best.pt'
        self.get_logger().info(f'YOLO 모델 로딩 중... ({model_path})')
        self.yolo_model = YOLO(model_path)
        self.get_logger().info('✅ YOLO 모델 로딩 완료!')
        
        # --- 2-1. Firebase 초기화 ---
        try:
            cred = credentials.Certificate(SERVICE_ACCOUNT_KEY_PATH)
            # 이미 초기화된 앱이 있는지 확인 (중복 에러 방지)
            if not firebase_admin._apps:
                firebase_admin.initialize_app(cred, {
                    'storageBucket': STORAGE_BUCKET_NAME
                })
            self.bucket = storage.bucket()
            self.get_logger().info('✅ Firebase Storage 연결 완료!')
        except Exception as e:
            self.get_logger().error(f'❌ Firebase 초기화 실패: {e}')
            self.bucket = None

        # 3. 데이터 구독
        self.latest_pc_msg = None
        self.latest_img_msg = None
        self.latest_depth_msg = None
        self.cam_info = None
        
        self.sub_pc = self.create_subscription(PointCloud2, '/camera/camera/depth/color/points', self.pc_callback, 10)
        self.sub_img = self.create_subscription(Image, '/camera/camera/color/image_raw', self.img_callback, 10)
        self.sub_depth = self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        self.sub_info = self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.info_callback, 10)

        self.capture_count = 0
        self.get_logger().info('★ 모든 데이터 수신 대기 중... "s" 키를 누르면 [10mm 기준 단차 검사]가 시작됩니다. ★')

        # 4. 키보드 스레드
        self.key_thread = threading.Thread(target=self.keyboard_listener)
        self.key_thread.daemon = True
        self.key_thread.start()

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
                        self.get_logger().info('📸 [검사 시작] 나사가 10mm 이상 튀어나왔는지 정밀 분석합니다...')
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
        
    def upload_to_firebase(self, file_path, destination_blob_name, content_type):
        """로컬 파일을 Firebase Storage에 업로드하는 헬퍼 함수"""
        if not self.bucket:
            self.get_logger().error('Firebase가 초기화되지 않아 업로드를 건너뜁니다.')
            return None

        try:
            blob = self.bucket.blob(destination_blob_name)
            blob.upload_from_filename(file_path, content_type=content_type)
            # 업로드된 파일의 공개 URL 생성
            blob.make_public()
            return blob.public_url
        except Exception as e:
            self.get_logger().error(f'❌ 파일 업로드 실패 ({file_path}): {e}')
            return None

    def process_capture(self):
        if not all([self.latest_pc_msg, self.latest_img_msg, self.latest_depth_msg, self.cam_info]):
            self.get_logger().warn('카메라 데이터를 받지 못했습니다.')
            return

        pc_msg, img_msg, depth_msg, cam_info = self.latest_pc_msg, self.latest_img_msg, self.latest_depth_msg, self.cam_info
        self.capture_count += 1
        
        tf_pc = self.get_tf_matrix('base_link', pc_msg.header.frame_id)
        tf_color = self.get_tf_matrix('base_link', img_msg.header.frame_id)
        
        fx, fy, cx, cy = cam_info.k[0], cam_info.k[4], cam_info.k[2], cam_info.k[5]
        cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        img_h, img_w = cv_image.shape[:2]
        cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        
        yolo_results = self.yolo_model(cv_image, verbose=False)[0]
        detected_markers = []

        for box in yolo_results.boxes:
            b = box.xyxy[0].cpu().numpy()
            xmin, ymin, xmax, ymax = b[0], b[1], b[2], b[3]
            
            cx_img = int((xmin + xmax) / 2)
            cy_img = int((ymin + ymax) / 2)
            bw_img = int(xmax - xmin)
            bh_img = int(ymax - ymin)

            screw_pts, surf_pts = [], []
            h, w = cv_depth.shape
            
            search_rx = int(bw_img * 0.75) 
            search_ry = int(bh_img * 0.75)

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
            z_surf = np.median(surf_pts) 
            delta_z_mm = z_surf - z_screw
            is_defective = delta_z_mm > 10.0 
            
            status = "불량(돌출)" if is_defective else "정상(체결)"
            color = 'red' if is_defective else 'green'
            
            cam_z = z_screw / 1000.0
            pt_base = tf_color @ np.array([(cx_img - cx) * cam_z / fx, (cy_img - cy) * cam_z / fy, cam_z, 1.0])
            
            self.get_logger().info(f'🎯 [{status}] 돌출 높이: {delta_z_mm:.1f}mm | Z_나사: {z_screw:.1f}mm, Z_바닥: {z_surf:.1f}mm')
            detected_markers.append({'name': f"Screw ({status})", 'x': pt_base[0], 'y': pt_base[1], 'z': pt_base[2], 'color': color})

        # 2. 결과 시각화 및 HTML 파일 생성 (JPG 관련 로직 제거)
        pc_data = list(pc2.read_points(pc_msg, field_names=("x", "y", "z", "rgb"), skip_nans=True))
        pts = np.array([[p[0], p[1], p[2]] for p in pc_data])
        colors = np.array([[(struct.unpack('I', struct.pack('f', p[3]))[0] >> 16 & 0xFF)/255.0, 
                            (struct.unpack('I', struct.pack('f', p[3]))[0] >> 8 & 0xFF)/255.0, 
                            (struct.unpack('I', struct.pack('f', p[3]))[0] & 0xFF)/255.0] for p in pc_data])
        
        pts_transformed = (tf_pc @ np.hstack([pts, np.ones((len(pts), 1))]).T).T[:, :3]
        
        plot_data = [go.Scatter3d(x=pts_transformed[:, 0], y=pts_transformed[:, 1], z=pts_transformed[:, 2], 
                                  mode='markers', marker=dict(size=1.5, color=[f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' for c in colors]))]

        for m in detected_markers:
            plot_data.append(go.Scatter3d(x=[m['x']], y=[m['y']], z=[m['z']], mode='markers+text',
                                          marker=dict(size=30, color=m['color'], symbol='circle', line=dict(color='black', width=2)),
                                          text=[f"★ {m['name']}"], textfont=dict(size=18, color=m['color'])))

        fig = go.Figure(data=plot_data)
        fig.update_layout(scene=dict(aspectmode='data'), title=f"10mm 기준 단차 검사 결과: {self.capture_count}")
        
        # HTML 파일 로컬 저장
        html_filename = f'inspection_{self.capture_count}.html'
        fig.write_html(html_filename)
        self.get_logger().info(f'✅ 3D 렌더링 로컬 저장 완료: {html_filename}')

        # --- Firebase 업로드 실행 (HTML만) ---
        self.get_logger().info('☁️ Firebase Storage로 3D 결과(HTML) 업로드를 시작합니다...')
        
        html_blob_path = f'inspections/{html_filename}'
        html_url = self.upload_to_firebase(html_filename, html_blob_path, 'text/html')

        if html_url:
            self.get_logger().info(f'🎉 Firebase 업로드 성공!')
            self.get_logger().info(f'🌐 3D 뷰어 링크(HTML): {html_url}')
        else:
            self.get_logger().warn('⚠️ Firebase 업로드에 문제가 발생했습니다.')


def main(args=None):
    rclpy.init(args=args)
    node = Yolo3DMapperNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()