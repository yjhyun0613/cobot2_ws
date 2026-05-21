#ICP 기능 추가


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
from datetime import datetime

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

        # 3. 🌟 Open3D 전용 글로벌 맵 변수 (ICP를 위해 구조 변경)
        self.global_pcd = None
        self.capture_count = 0
        
        self.is_recording = False
        self.capture_timer = self.create_timer(1.0 / 3.0, self.timer_callback)

        self.get_logger().info('=========================================')
        self.get_logger().info('🎥 [정밀 ICP 맵핑 모드] 준비 완료!')
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
            self.get_logger().info('▶️ [녹화 시작] 로봇을 천천히 움직여주세요... ICP 퍼즐을 맞춥니다!')
        else:
            self.get_logger().info('⏸️ [일시 정지] 녹화가 잠시 중단되었습니다.')

    def timer_callback(self):
        if not self.is_recording or self.latest_pc_msg is None:
            return

        pc_msg = self.latest_pc_msg
        tf_pc = self.get_tf_matrix('base_link', pc_msg.header.frame_id)
        if tf_pc is None: return

        # 데이터 파싱 및 1m 필터링
        pc_data = list(pc2.read_points(pc_msg, field_names=("x", "y", "z", "rgb"), skip_nans=True))
        pts, colors = [], []
        for p in pc_data:
            cam_z = float(p[2])
            if cam_z > 1.0 or cam_z <= 0: continue
            pts.append([float(p[0]), float(p[1]), cam_z])
            packed = struct.pack('f', p[3])
            i = struct.unpack('I', packed)[0]
            colors.append([(i >> 16 & 0xFF)/255.0, (i >> 8 & 0xFF)/255.0, (i & 0xFF)/255.0])

        if not pts: return

        pts_np = np.array(pts)
        ones = np.ones((pts_np.shape[0], 1))
        pts_transformed = (tf_pc @ np.hstack([pts_np, ones]).T).T[:, :3]

        # 🌟 현재 화면(프레임)을 Open3D 포인트 클라우드로 변환
        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(pts_transformed)
        source_pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
        
        # 연산 속도를 위해 1cm 단위로 미리 압축
        source_pcd = source_pcd.voxel_down_sample(voxel_size=0.01)

        # 🌟 [핵심] ICP 매칭 알고리즘 적용
        if self.global_pcd is None:
            # 첫 번째 사진은 무조건 기준이 됨
            self.global_pcd = source_pcd
        else:
            # 두 번째 사진부터는 기존 맵과 모양을 비교해서 끼워 맞춤
            threshold = 0.05 # 최대 5cm 이내의 오차만 찾아서 맞춤 (너무 멀면 엉뚱한데 붙음 방지)
            trans_init = np.eye(4) # TF가 이미 적용되었으므로 초기 변환은 0
            
            reg_p2p = o3d.pipelines.registration.registration_icp(
                source_pcd, self.global_pcd, threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
            )
            
            # ICP가 찾아낸 미세 오차 각도만큼 현재 사진을 비틀어서 교정!
            source_pcd.transform(reg_p2p.transformation)
            
            # 교정된 사진을 글로벌 맵에 병합
            self.global_pcd += source_pcd
            
            # 병합 후 맵이 무거워지지 않도록 다시 1cm 단위로 깔끔하게 정리
            self.global_pcd = self.global_pcd.voxel_down_sample(voxel_size=0.01)

        self.capture_count += 1
        print(f'\r📷 ICP 매칭 중... 현재 {self.capture_count}장 누적 완료', end='', flush=True)

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
        
        if self.global_pcd is None or len(self.global_pcd.points) == 0:
            self.get_logger().warn('저장할 맵 데이터가 없습니다.')
            return

        self.get_logger().info('=========================================')
        self.get_logger().info('🌐 정밀하게 교정된 HTML 3D 파일을 생성합니다...')

        down_pts = np.asarray(self.global_pcd.points)
        down_colors = np.asarray(self.global_pcd.colors) * 255

        now = datetime.now()
        time_str_file = now.strftime("%Y%m%d_%H%M%S")
        time_str_title = now.strftime("%Y-%m-%d %H:%M:%S")

        html_colors = [f'rgb({int(c[0])},{int(c[1])},{int(c[2])})' for c in down_colors]
        plot_data = [go.Scatter3d(
            x=down_pts[:, 0], y=down_pts[:, 1], z=down_pts[:, 2],
            mode='markers', marker=dict(size=2, color=html_colors)
        )]

        fig = go.Figure(data=plot_data)
        fig.update_layout(scene=dict(aspectmode='data', 
                                     xaxis_title='X (base_link)', 
                                     yaxis_title='Y (base_link)', 
                                     zaxis_title='Z (base_link)'), 
                          title=f"ICP Stitched 3D Map ({self.capture_count} Scans) - {time_str_title}")
        
        filename = f'icp_map_{time_str_file}.html'
        fig.write_html(filename)
        self.get_logger().info(f'✅ 성공! [{filename}] 파일이 저장되었습니다.')
        self.get_logger().info('=========================================')
        
        self.global_pcd = None
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