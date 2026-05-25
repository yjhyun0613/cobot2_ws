import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
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

class PointCloudCaptureNode(Node):
    def __init__(self):
        super().__init__('pointcloud_capture_node')

        # 1. Static TF Broadcaster (link_6 -> camera_link 연결)
        # 터미널에서 치지 않아도 코드 실행 시 자동으로 위치 정보를 퍼블리시합니다.
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        self.publish_static_transform()

        # 2. TF Listener (base_link부터 카메라 렌즈까지의 전체 뼈대 변환을 추적)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # 3. Point Cloud Subscriber
        self.latest_pc_msg = None
        self.subscription = self.create_subscription(
            PointCloud2,
            '/camera/camera/depth/color/points',
            self.pc_callback,
            10)

        self.capture_count = 0
        self.get_logger().info('노드가 초기화되었습니다. 포인트 클라우드 데이터를 기다리는 중...')
        self.get_logger().info('★ 이 터미널 창을 선택한 상태에서 "s" 키를 누르면 캡처됩니다. (Ctrl+C로 종료) ★')

        # 4. 키보드 입력을 백그라운드에서 감지하는 스레드 실행
        self.key_thread = threading.Thread(target=self.keyboard_listener)
        self.key_thread.daemon = True
        self.key_thread.start()

    def publish_static_transform(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'link_6'
        t.child_frame_id = 'camera_link'

        # 알려주신 카메라 위치 오프셋 (단위: m)
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.075    # 75mm
        t.transform.translation.z = 0.03991  # 39.91mm

        # 기본 방향 유지
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        self.tf_static_broadcaster.sendTransform(t)

    def pc_callback(self, msg):
        # 가장 최근에 들어온 포인트 클라우드 데이터를 계속 업데이트
        self.latest_pc_msg = msg

    def keyboard_listener(self):
        # 터미널에서 엔터 없이 's' 키만 눌러도 즉시 반응하도록 설정
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while rclpy.ok():
                # 0.1초마다 키보드 입력이 있는지 확인 (논블로킹)
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)
                    if key.lower() == 's':
                        self.get_logger().info('---------------------------------------')
                        self.get_logger().info('[캡처 신호 수신] 데이터 변환 및 저장을 시작합니다...')
                        self.process_capture()
                    elif key == '\x03': # Ctrl+C 입력 시 안전하게 종료
                        self.get_logger().info('종료합니다.')
                        rclpy.shutdown()
                        break
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def process_capture(self):
        if self.latest_pc_msg is None:
            self.get_logger().warn('아직 카메라로부터 포인트 클라우드 데이터를 받지 못했습니다.')
            return

        msg = self.latest_pc_msg
        target_frame = 'base_link'
        source_frame = msg.header.frame_id  # camera_depth_optical_frame

        try:
            # 현재 시점의 base_link -> 카메라 간의 4x4 변환 행렬(TF) 가져오기
            trans = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(), # 최신 시간의 변환 가져오기
                timeout=rclpy.duration.Duration(seconds=1.0))
        except Exception as e:
            self.get_logger().error(f'좌표계 변환 실패 (TF 트리를 확인해주세요): {e}')
            return

        self.capture_count += 1
        filename_base = f'capture_{self.capture_count}'

        # 이동 및 회전 변환 행렬 구성
        tx = trans.transform.translation.x
        ty = trans.transform.translation.y
        tz = trans.transform.translation.z
        rx = trans.transform.rotation.x
        ry = trans.transform.rotation.y
        rz = trans.transform.rotation.z
        rw = trans.transform.rotation.w

        rot = R.from_quat([rx, ry, rz, rw]).as_matrix()
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rot
        transform_matrix[:3, 3] = [tx, ty, tz]

        points = []
        colors = []

        # ROS 메시지에서 X, Y, Z 및 RGB 데이터 추출
        for p in pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True):
            # 수정된 부분: 데이터를 인덱스로 하나씩 직접 꺼냅니다.
            x = float(p[0])
            y = float(p[1])
            z = float(p[2])
            rgb_float = p[3]
            
            # ROS는 RGB를 float32 하나에 압축해서 보내므로 이를 분리하는 작업
            packed = struct.pack('f', rgb_float)
            i = struct.unpack('I', packed)[0]
            r = (i >> 16) & 0x000000FF
            g = (i >> 8) & 0x000000FF
            b = (i) & 0x000000FF

            points.append([x, y, z])
            colors.append([r / 255.0, g / 255.0, b / 255.0])

        if not points:
            self.get_logger().warn('데이터에 유효한 점(Point)이 없습니다.')
            return

        points_np = np.array(points)
        colors_np = np.array(colors)

        # 4x4 변환 행렬을 모든 점의 좌표에 곱하여 base_link 기준으로 매핑
        ones = np.ones((points_np.shape[0], 1))
        points_homogeneous = np.hstack([points_np, ones])
        points_transformed = (transform_matrix @ points_homogeneous.T).T[:, :3]

        # 1. PCD 파일 저장 (원본 해상도)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_transformed)
        pcd.colors = o3d.utility.Vector3dVector(colors_np)
        o3d.io.write_point_cloud(f'{filename_base}.pcd', pcd)
        self.get_logger().info(f'✅ {filename_base}.pcd 저장 완료 (원본 데이터)')

        # 2. HTML 파일 저장 (웹브라우저 성능을 위해 Voxel Downsampling 진행)
        # 점이 수십만 개면 브라우저가 멈추므로 1cm 간격으로 점을 줄여서 저장합니다.
        downpcd = pcd.voxel_down_sample(voxel_size=0.001)
        down_pts = np.asarray(downpcd.points)
        down_colors = np.asarray(downpcd.colors) * 255

        if len(down_pts) > 0:
            html_colors = [f'rgb({int(r)}, {int(g)}, {int(b)})' for r, g, b in down_colors]
            fig = go.Figure(data=[go.Scatter3d(
                x=down_pts[:, 0],
                y=down_pts[:, 1],
                z=down_pts[:, 2],
                mode='markers',
                marker=dict(size=2, color=html_colors)
            )])
            fig.update_layout(
                scene=dict(
                    aspectmode='data',
                    xaxis_title='X (base_link 기준)',
                    yaxis_title='Y (base_link 기준)',
                    zaxis_title='Z (base_link 기준)'
                ), 
                title=f"3D Capture - {filename_base}"
            )
            fig.write_html(f'{filename_base}.html')
            self.get_logger().info(f'✅ {filename_base}.html 저장 완료 (웹 뷰어용)')

def main(args=None):
    rclpy.init(args=args)
    node = PointCloudCaptureNode()
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