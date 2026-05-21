import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
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
import math
import cv2
from cv_bridge import CvBridge

# YOLO 라이브러리 임포트
from ultralytics import YOLO

class Yolo3DMapperNode(Node):
    def __init__(self):
        super().__init__('yolo_3d_mapper_node')

        # 1. TF 설정 (축 보정 완료된 값)
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        self.publish_static_transform()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # 2. YOLO 및 OpenCV 설정
        self.bridge = CvBridge()
        # 알려주신 YOLO 모델 경로
        model_path = 'yolov8n_tools_0122.pt'
        self.get_logger().info(f'YOLO 모델 로딩 중... ({model_path})')
        self.yolo_model = YOLO(model_path)
        self.get_logger().info('✅ YOLO 모델 로딩 완료!')

        # 3. Subscriber 설정 (3D 점구름 + 2D 사진)
        self.latest_pc_msg = None
        self.latest_img_msg = None
        
        self.sub_pc = self.create_subscription(
            PointCloud2, '/camera/camera/depth/color/points', self.pc_callback, 10)
        self.sub_img = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.img_callback, 10)

        self.capture_count = 0
        self.get_logger().info('★ 모든 데이터 수신 대기 중... "s" 키를 누르면 캡처+YOLO가 실행됩니다. (Ctrl+C로 종료) ★')

        # 4. 키보드 스레드
        self.key_thread = threading.Thread(target=self.keyboard_listener)
        self.key_thread.daemon = True
        self.key_thread.start()

    def publish_static_transform(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'link_6'
        t.child_frame_id = 'camera_link'

        # 물리적 오프셋 및 축 보정 (Roll -90, Pitch -90)
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

    def keyboard_listener(self):
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while rclpy.ok():
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)
                    if key.lower() == 's':
                        self.get_logger().info('---------------------------------------')
                        self.get_logger().info('📸 [캡처 찰칵] 2D/3D 데이터를 결합합니다...')
                        self.process_capture()
                    elif key == '\x03':
                        rclpy.shutdown()
                        break
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def process_capture(self):
        if self.latest_pc_msg is None or self.latest_img_msg is None:
            self.get_logger().warn('아직 카메라로부터 데이터를 받지 못했습니다.')
            return

        # 찰칵 순간의 데이터를 변수에 고정
        pc_msg = self.latest_pc_msg
        img_msg = self.latest_img_msg
        self.capture_count += 1
        filename_base = f'yolo_capture_{self.capture_count}'

        try:
            # 베이스 링크 기준 변환 행렬 가져오기
            trans = self.tf_buffer.lookup_transform(
                'base_link', pc_msg.header.frame_id, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0))
        except Exception as e:
            self.get_logger().error(f'좌표계 변환 실패: {e}')
            return

        tx, ty, tz = trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z
        rx, ry, rz, rw = trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w
        rot = R.from_quat([rx, ry, rz, rw]).as_matrix()
        tf_matrix = np.eye(4)
        tf_matrix[:3, :3] = rot
        tf_matrix[:3, 3] = [tx, ty, tz]

        # 1. 2D 이미지 YOLO 추론
        cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        img_h, img_w = cv_image.shape[:2]  # 2D 사진의 진짜 해상도 (예: 1280x720)
        
        yolo_results = self.yolo_model(cv_image, verbose=False)[0]
        
        # 2. 3D 포인트 클라우드 파싱
        pc_w = pc_msg.width
        pc_h = pc_msg.height
        pc_data_generator = pc2.read_points(pc_msg, field_names=("x", "y", "z", "rgb"), skip_nans=False)
        pc_data = list(pc_data_generator)

        # 3D 데이터가 1차원(Unordered)으로 들어올 경우를 대비한 가로세로 역산
        if pc_h == 1: 
            pc_w_real = int(np.sqrt(len(pc_data) * (img_w / img_h)))
            pc_h_real = int(pc_w_real * (img_h / img_w))
        else:
            pc_w_real = pc_w
            pc_h_real = pc_h

        # 마커 저장용 리스트
        detected_markers = []

        # YOLO가 찾은 물체들 처리
        for box in yolo_results.boxes:
            b = box.xyxy[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())
            class_name = self.yolo_model.names[class_id]

            # 2D 사진 기준 바운딩 박스 중심 픽셀 (u, v)
            cx_img = int((b[0] + b[2]) / 2)
            cy_img = int((b[1] + b[3]) / 2)

            # [핵심 해결책] 2D 해상도와 3D 해상도의 비율을 곱해서 3D 좌표 인덱스로 변환 (스케일링)
            cx_pc = int(cx_img * (pc_w_real / img_w))
            cy_pc = int(cy_img * (pc_h_real / img_h))

            # 픽셀 좌표를 1차원 인덱스로 변환
            idx = cy_pc * pc_w_real + cx_pc
            
            # 인덱스 범위 초과 방지 및 원인 로깅 (이제 무시하지 않고 경고창을 띄움)
            if idx >= len(pc_data):
                self.get_logger().warn(f'[{class_name}] 3D 데이터 범위를 벗어났습니다! (해상도 매핑 오류)')
                continue

            p = pc_data[idx]
            cam_x = float(p[0])
            cam_y = float(p[1])
            cam_z = float(p[2])

            # 뎁스 값이 비어있는(NaN) 반사 표면 등은 제외
            if math.isnan(cam_x) or math.isnan(cam_z):
                self.get_logger().warn(f'[{class_name}] 중심점 뎁스 데이터가 유실되었습니다. (표면 반사 또는 측정 불가 거리)')
                continue

            # 카메라 기준 좌표를 base_link 기준 절대 좌표로 변환
            pt_homogeneous = np.array([float(cam_x), float(cam_y), float(cam_z), 1.0])
            pt_base_link = tf_matrix @ pt_homogeneous
            target_x, target_y, target_z = pt_base_link[:3]

            # 드디어 터미널에 출력!
            self.get_logger().info(f'🎯 욜로 감지! [{class_name}] -> base_link 좌표: X={target_x:.3f}, Y={target_y:.3f}, Z={target_z:.3f}')
            
            detected_markers.append({
                'name': class_name,
                'x': target_x, 'y': target_y, 'z': target_z
            })

        # 3. HTML 배경이 될 3D 맵 구성 (NaN 값을 제거하여 유효한 점만 추출)
        points, colors = [], []
        for p in pc_data:
            if math.isnan(p[0]): continue
            x, y, z = float(p[0]), float(p[1]), float(p[2])
            packed = struct.pack('f', p[3])
            i = struct.unpack('I', packed)[0]
            r, g, b = (i >> 16) & 0xFF, (i >> 8) & 0xFF, i & 0xFF
            points.append([x, y, z])
            colors.append([r / 255.0, g / 255.0, b / 255.0])

        points_np = np.array(points)
        colors_np = np.array(colors)
        ones = np.ones((points_np.shape[0], 1))
        points_homogeneous = np.hstack([points_np, ones])
        points_transformed = (tf_matrix @ points_homogeneous.T).T[:, :3]

        # Voxel Downsampling (성능을 위해 5mm)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_transformed)
        pcd.colors = o3d.utility.Vector3dVector(colors_np)
        downpcd = pcd.voxel_down_sample(voxel_size=0.005)
        
        down_pts = np.asarray(downpcd.points)
        down_colors = np.asarray(downpcd.colors) * 255

        # 4. Plotly HTML 시각화 (3D 맵 + YOLO 마커)
        if len(down_pts) > 0:
            html_colors = [f'rgb({int(r)}, {int(g)}, {int(b)})' for r, g, b in down_colors]
            
            # 배경 3D 점구름 레이어
            plot_data = [go.Scatter3d(
                x=down_pts[:, 0], y=down_pts[:, 1], z=down_pts[:, 2],
                mode='markers', marker=dict(size=2, color=html_colors), name='Environment'
            )]

            # YOLO 물체 마커 레이어 추가 (커다란 구슬)
            for m in detected_markers:
                plot_data.append(go.Scatter3d(
                    x=[m['x']], y=[m['y']], z=[m['z']],
                    mode='markers+text',
                    marker=dict(size=30, color='red', symbol='circle', line=dict(color='black', width=2)),
                    text=[f"★ {m['name']}"],
                    textposition="top center",
                    textfont=dict(size=15, color="red"),
                    name=m['name']
                ))

            fig = go.Figure(data=plot_data)
            fig.update_layout(
                scene=dict(
                    aspectmode='data',
                    xaxis_title='X (base_link 기준)',
                    yaxis_title='Y (base_link 기준)',
                    zaxis_title='Z (base_link 기준)'
                ), 
                title=f"YOLO + 3D Map: {filename_base}"
            )
            fig.write_html(f'{filename_base}.html')
            
            # 2D 욜로 박스쳐진 이미지도 함께 저장 (확인용)
            annotated_img = yolo_results.plot()
            cv2.imwrite(f'{filename_base}_2d.jpg', annotated_img)
            
            self.get_logger().info(f'✅ {filename_base}.html 및 2D 캡처본 저장 완료!')

def main(args=None):
    rclpy.init(args=args)
    node = Yolo3DMapperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()