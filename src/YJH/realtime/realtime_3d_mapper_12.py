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
from cv_bridge import CvBridge

# YOLO 라이브러리 임포트
from ultralytics import YOLO

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
        model_path = 'yolov8n_tools_0122.pt'
        self.get_logger().info(f'YOLO 모델 로딩 중... ({model_path})')
        self.yolo_model = YOLO(model_path)
        self.get_logger().info('✅ YOLO 모델 로딩 완료!')

        # 3. 4개의 핵심 데이터 구독 (배경맵, 2D사진, 정렬된 깊이, 카메라 내부 정보)
        self.latest_pc_msg = None
        self.latest_img_msg = None
        self.latest_depth_msg = None
        self.cam_info = None
        
        self.sub_pc = self.create_subscription(PointCloud2, '/camera/camera/depth/color/points', self.pc_callback, 10)
        self.sub_img = self.create_subscription(Image, '/camera/camera/color/image_raw', self.img_callback, 10)
        self.sub_depth = self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        self.sub_info = self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.info_callback, 10)

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
                        self.get_logger().info('---------------------------------------')
                        self.get_logger().info('📸 [캡처 찰칵] 정밀 수학 계산을 통한 좌표 매핑 시작...')
                        self.process_capture()
                    elif key == '\x03':
                        self.get_logger().info('종료합니다.')
                        rclpy.shutdown()
                        break
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def get_tf_matrix(self, target_frame, source_frame):
        try:
            trans = self.tf_buffer.lookup_transform(
                target_frame, source_frame, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0))
            tx, ty, tz = trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z
            rx, ry, rz, rw = trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w
            rot = R.from_quat([rx, ry, rz, rw]).as_matrix()
            mat = np.eye(4)
            mat[:3, :3] = rot
            mat[:3, 3] = [tx, ty, tz]
            return mat
        except Exception as e:
            self.get_logger().error(f'좌표계 변환 실패 ({source_frame}): {e}')
            return None

    def process_capture(self):
        if not all([self.latest_pc_msg, self.latest_img_msg, self.latest_depth_msg, self.cam_info]):
            self.get_logger().warn('데이터를 모두 수신하지 못했습니다. 카메라 토픽이 정상인지 확인해주세요.')
            return

        pc_msg, img_msg, depth_msg, cam_info = self.latest_pc_msg, self.latest_img_msg, self.latest_depth_msg, self.cam_info
        self.capture_count += 1
        filename_base = f'yolo_capture_{self.capture_count}'

        # 배경 맵(Depth 렌즈)과 욜로 마커(Color 렌즈)의 변환 행렬을 각각 다르게 가져와서 시차 완벽 보정
        tf_pc = self.get_tf_matrix('base_link', pc_msg.header.frame_id)
        tf_color = self.get_tf_matrix('base_link', img_msg.header.frame_id)
        
        if tf_pc is None or tf_color is None: return

        # 카메라 내부 파라미터 (Pinhole Math)
        fx, fy = cam_info.k[0], cam_info.k[4]
        cx, cy = cam_info.k[2], cam_info.k[5]

        cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough') # 16-bit 뎁스 (mm 단위)
        
        yolo_results = self.yolo_model(cv_image, verbose=False)[0]
        detected_markers = []

        # 1. YOLO 물체 처리 및 정밀 좌표 추출
        for box in yolo_results.boxes:
            b = box.xyxy[0].cpu().numpy()
            class_name = self.yolo_model.names[int(box.cls[0].cpu().numpy())]

            cx_img = int((b[0] + b[2]) / 2)
            cy_img = int((b[1] + b[3]) / 2)

            # 스마트 주변 탐색: 중앙 픽셀 주변 11x11을 뒤져서 빈 공간(도넛 현상)을 거르고 표면 Z값 획득
            search_radius = 5
            valid_z = []
            h, w = cv_depth.shape
            
            for dy in range(-search_radius, search_radius + 1):
                for dx in range(-search_radius, search_radius + 1):
                    nx, ny = cx_img + dx, cy_img + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        z_mm = cv_depth[ny, nx]
                        if z_mm > 0: # 0은 거리 측정 불가(NaN)를 의미함
                            valid_z.append(z_mm)

            if not valid_z:
                self.get_logger().warn(f'[{class_name}] 표면에서 깊이 값을 찾을 수 없습니다.')
                continue

            # 카메라에 가장 가까운 최상단 표면점의 깊이 (m 단위 변환)
            z_m = min(valid_z) / 1000.0

            # 실제 렌즈 굴절률 수학 공식을 이용한 완벽한 3D 복원
            x_m = (cx_img - cx) * z_m / fx
            y_m = (cy_img - cy) * z_m / fy

            # Color 렌즈 기준 좌표를 로봇 base_link 절대 좌표로 변환
            pt_color_frame = np.array([x_m, y_m, z_m, 1.0])
            pt_base_link = tf_color @ pt_color_frame
            target_x, target_y, target_z = pt_base_link[:3]

            self.get_logger().info(f'🎯 욜로 감지! [{class_name}] -> base_link 좌표: X={target_x:.3f}, Y={target_y:.3f}, Z={target_z:.3f}')
            detected_markers.append({'name': class_name, 'x': target_x, 'y': target_y, 'z': target_z})

        # 2. HTML 배경이 될 3D 맵 구성 (인덱스 매칭 불필요로 속도 대폭 향상)
        self.get_logger().info('배경 3D 맵을 생성 중입니다...')
        pc_data_generator = pc2.read_points(pc_msg, field_names=("x", "y", "z", "rgb"), skip_nans=True)
        points, colors = [], []
        
        for p in pc_data_generator:
            points.append([float(p[0]), float(p[1]), float(p[2])])
            packed = struct.pack('f', p[3])
            i = struct.unpack('I', packed)[0]
            colors.append([(i >> 16 & 0xFF) / 255.0, (i >> 8 & 0xFF) / 255.0, (i & 0xFF) / 255.0])

        points_np, colors_np = np.array(points), np.array(colors)
        if len(points_np) > 0:
            ones = np.ones((points_np.shape[0], 1))
            points_transformed = (tf_pc @ np.hstack([points_np, ones]).T).T[:, :3]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_transformed)
            pcd.colors = o3d.utility.Vector3dVector(colors_np)
            downpcd = pcd.voxel_down_sample(voxel_size=0.005)
            
            down_pts = np.asarray(downpcd.points)
            down_colors = np.asarray(downpcd.colors) * 255

            # 3. Plotly HTML 시각화
            if len(down_pts) > 0:
                html_colors = [f'rgb({int(r)}, {int(g)}, {int(b)})' for r, g, b in down_colors]
                plot_data = [go.Scatter3d(
                    x=down_pts[:, 0], y=down_pts[:, 1], z=down_pts[:, 2],
                    mode='markers', marker=dict(size=2, color=html_colors), name='Environment'
                )]

                for m in detected_markers:
                    plot_data.append(go.Scatter3d(
                        x=[m['x']], y=[m['y']], z=[m['z']],
                        mode='markers+text',
                        marker=dict(size=30, color='red', symbol='circle', line=dict(color='black', width=2)),
                        text=[f"★ {m['name']}"], textposition="top center",
                        textfont=dict(size=18, color="red", family="Arial Black"), name=m['name']
                    ))

                fig = go.Figure(data=plot_data)
                fig.update_layout(scene=dict(aspectmode='data', xaxis_title='X (base_link)', yaxis_title='Y (base_link)', zaxis_title='Z (base_link)'), title=f"YOLO + 3D Map: {filename_base}")
                fig.write_html(f'{filename_base}.html')
                
                annotated_img = yolo_results.plot()
                cv2.imwrite(f'{filename_base}_2d.jpg', annotated_img)
                self.get_logger().info(f'✅ {filename_base}.html 저장 완료! 좌표가 완벽히 정렬되었습니다.')

def main(args=None):
    rclpy.init(args=args)
    node = Yolo3DMapperNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()