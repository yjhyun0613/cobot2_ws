#1초에 3작씩 pointcloud 가져오기


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
from datetime import datetime # 🌟 시간 저장을 위한 라이브러리 추가

class MapStitcherNode(Node):
    def __init__(self):
        super().__init__('map_stitcher_node')

        # 1. TF 설정
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        self.publish_static_transform()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # 2. 3D 데이터 구독
        self.latest_pc_msg = None
        self.sub_pc = self.create_subscription(
            PointCloud2, '/camera/camera/depth/color/points', self.pc_callback, 10)

        # 3. 맵 데이터 누적 변수
        self.accumulated_points = []
        self.accumulated_colors = []
        self.capture_count = 0
        
        # 연속 스캔을 위한 상태 변수 (초당 3번)
        self.is_recording = False
        self.capture_timer = self.create_timer(1.0 / 3.0, self.timer_callback)

        self.get_logger().info('=========================================')
        self.get_logger().info('🎥 [연속 맵핑 모드] 준비 완료! (시간 자동 저장)')
        self.get_logger().info(' [r] 키: 녹화 시작 / 일시정지 (Toggle)')
        self.get_logger().info(' [f] 키: 녹화 종료 및 3D 맵 저장 (Finish)')
        self.get_logger().info(' [Ctrl+C]: 종료')
        self.get_logger().info('=========================================')

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

    def pc_callback(self, msg):
        self.latest_pc_msg = msg

    def keyboard_listener(self):
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while rclpy.ok():
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1).lower()
                    if key == 'r':
                        self.toggle_recording()
                    elif key == 'f':
                        self.finish_and_save_map()
                    elif key == '\x03': # Ctrl+C
                        rclpy.shutdown()
                        break
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def toggle_recording(self):
        self.is_recording = not self.is_recording
        if self.is_recording:
            self.get_logger().info('▶️ [녹화 시작] 찰칵... 찰칵... 로봇을 천천히 움직여주세요!')
        else:
            self.get_logger().info('⏸️ [일시 정지] 녹화가 잠시 중단되었습니다. (다시 r을 누르면 재개)')

    def timer_callback(self):
        if not self.is_recording or self.latest_pc_msg is None:
            return

        pc_msg = self.latest_pc_msg
        tf_pc = self.get_tf_matrix('base_link', pc_msg.header.frame_id)
        if tf_pc is None: return

        pc_data = list(pc2.read_points(pc_msg, field_names=("x", "y", "z", "rgb"), skip_nans=True))
        
        pts, colors = [], []
        MAX_DISTANCE_M = 1.0 # 1m 이내만 저장

        for p in pc_data:
            cam_z = float(p[2])
            if cam_z > MAX_DISTANCE_M or cam_z <= 0:
                continue
                
            pts.append([float(p[0]), float(p[1]), cam_z])
            packed = struct.pack('f', p[3])
            i = struct.unpack('I', packed)[0]
            colors.append([(i >> 16 & 0xFF)/255.0, (i >> 8 & 0xFF)/255.0, (i & 0xFF)/255.0])

        pts_np = np.array(pts)
        if len(pts_np) > 0:
            ones = np.ones((pts_np.shape[0], 1))
            pts_transformed = (tf_pc @ np.hstack([pts_np, ones]).T).T[:, :3]

            self.accumulated_points.extend(pts_transformed.tolist())
            self.accumulated_colors.extend(colors)
            
            self.capture_count += 1
            print(f'\r📷 연속 스캔 중... 현재 {self.capture_count}장 누적 완료', end='', flush=True)

    def get_tf_matrix(self, target_frame, source_frame):
        try:
            trans = self.tf_buffer.lookup_transform(
                target_frame, source_frame, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0))
            rot = R.from_quat([trans.transform.rotation.x, trans.transform.rotation.y, 
                               trans.transform.rotation.z, trans.transform.rotation.w]).as_matrix()
            mat = np.eye(4)
            mat[:3, :3] = rot
            mat[:3, 3] = [trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z]
            return mat
        except:
            return None

    def finish_and_save_map(self):
        self.is_recording = False
        print() 
        
        if not self.accumulated_points:
            self.get_logger().warn('저장할 데이터가 없습니다. r을 눌러 먼저 스캔하세요.')
            return

        self.get_logger().info('=========================================')
        self.get_logger().info('🛠️ 방대한 맵 데이터를 최적화하고 병합합니다...')
        
        pts_np = np.array(self.accumulated_points)
        colors_np = np.array(self.accumulated_colors)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_np)
        pcd.colors = o3d.utility.Vector3dVector(colors_np)
        
        # 1cm 간격 압축
        downpcd = pcd.voxel_down_sample(voxel_size=0.01)
        
        down_pts = np.asarray(downpcd.points)
        down_colors = np.asarray(downpcd.colors) * 255

        self.get_logger().info(f'📉 압축 완료: 원본 점 {len(pts_np):,}개 -> 최종 맵 {len(down_pts):,}개')
        self.get_logger().info('🌐 HTML 3D 파일 생성 중...')

        # 🌟 현재 시간 가져오기
        now = datetime.now()
        time_str_file = now.strftime("%Y%m%d_%H%M%S")  # 파일명용 (예: 20260519_210843)
        time_str_title = now.strftime("%Y-%m-%d %H:%M:%S") # 제목용 (예: 2026-05-19 21:08:43)

        html_colors = [f'rgb({int(c[0])},{int(c[1])},{int(c[2])})' for c in down_colors]
        plot_data = [go.Scatter3d(
            x=down_pts[:, 0], y=down_pts[:, 1], z=down_pts[:, 2],
            mode='markers', marker=dict(size=2, color=html_colors)
        )]

        fig = go.Figure(data=plot_data)
        # HTML 내부 제목에도 시간 추가
        fig.update_layout(scene=dict(aspectmode='data', 
                                     xaxis_title='X (base_link)', 
                                     yaxis_title='Y (base_link)', 
                                     zaxis_title='Z (base_link)'), 
                          title=f"Continuous 3D Map ({self.capture_count} Scans) - {time_str_title}")
        
        # 🌟 파일명에 시간 추가
        filename = f'continuous_map_{time_str_file}.html'
        fig.write_html(filename)
        self.get_logger().info(f'✅ 완벽합니다! [{filename}] 파일이 저장되었습니다.')
        self.get_logger().info('=========================================')
        
        # 메모리 초기화
        self.accumulated_points = []
        self.accumulated_colors = []
        self.capture_count = 0

def main(args=None):
    rclpy.init(args=args)
    node = MapStitcherNode()
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