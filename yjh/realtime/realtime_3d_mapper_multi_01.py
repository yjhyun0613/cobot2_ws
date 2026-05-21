# r키를 누를떄마다 데이터 가져와 기록하기
# f키를 누르면 모든 점들 저장
# 단 1m 이하의 point만 기록하기


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

class MapStitcherNode(Node):
    def __init__(self):
        super().__init__('map_stitcher_node')

        # 1. TF 설정 (카메라 축 보정)
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        self.publish_static_transform()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # 2. 3D 데이터 구독
        self.latest_pc_msg = None
        self.sub_pc = self.create_subscription(
            PointCloud2, '/camera/camera/depth/color/points', self.pc_callback, 10)

        # 3. 맵 데이터 누적용 리스트
        self.accumulated_points = []
        self.accumulated_colors = []
        self.capture_count = 0

        self.get_logger().info('=========================================')
        self.get_logger().info('🗺️ 3D 맵 스티칭 모드 (1m 거리 제한 적용)')
        self.get_logger().info(' [s] 키: 현재 화면을 맵에 추가 (Scan)')
        self.get_logger().info(' [f] 키: 누적된 맵 병합 및 HTML 저장 (Finish)')
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
                    if key == 's':
                        self.capture_scan()
                    elif key == 'f':
                        self.finish_and_save_map()
                    elif key == '\x03': # Ctrl+C
                        rclpy.shutdown()
                        break
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

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
        except Exception as e:
            self.get_logger().error(f'좌표계 변환 실패: {e}')
            return None

    def capture_scan(self):
        if self.latest_pc_msg is None:
            self.get_logger().warn('아직 카메라로부터 3D 데이터를 받지 못했습니다.')
            return

        pc_msg = self.latest_pc_msg
        tf_pc = self.get_tf_matrix('base_link', pc_msg.header.frame_id)
        if tf_pc is None: return

        self.get_logger().info('점구름 데이터를 파싱하는 중...')
        pc_data = list(pc2.read_points(pc_msg, field_names=("x", "y", "z", "rgb"), skip_nans=True))
        
        pts, colors = [], []
        
        # 필터링 설정: 최대 측정 거리 (미터 단위)
        MAX_DISTANCE_M = 1.0 

        for p in pc_data:
            cam_z = float(p[2])
            
            # 🌟 [핵심 변경점] 카메라 렌즈 기준 Z값이 1m를 초과하거나 0 이하(에러)면 무시
            if cam_z > MAX_DISTANCE_M or cam_z <= 0:
                continue
                
            pts.append([float(p[0]), float(p[1]), cam_z])
            
            packed = struct.pack('f', p[3])
            i = struct.unpack('I', packed)[0]
            colors.append([(i >> 16 & 0xFF)/255.0, (i >> 8 & 0xFF)/255.0, (i & 0xFF)/255.0])

        pts_np = np.array(pts)
        if len(pts_np) > 0:
            # base_link 좌표계로 변환
            ones = np.ones((pts_np.shape[0], 1))
            pts_transformed = (tf_pc @ np.hstack([pts_np, ones]).T).T[:, :3]

            # 메모리에 누적
            self.accumulated_points.extend(pts_transformed.tolist())
            self.accumulated_colors.extend(colors)
            
            self.capture_count += 1
            self.get_logger().info(f'📸 [스캔 완료] {len(pts_transformed)}개의 점이 맵에 추가되었습니다. (현재 총 {self.capture_count}장 누적)')
        else:
            self.get_logger().warn('1m 이내에 잡힌 물체가 없습니다!')

    def finish_and_save_map(self):
        if not self.accumulated_points:
            self.get_logger().warn('누적된 맵 데이터가 없습니다. 먼저 "s"를 눌러 스캔해주세요.')
            return

        self.get_logger().info('=========================================')
        self.get_logger().info('🛠️ 누적된 조각들을 하나의 3D 맵으로 병합합니다...')
        
        pts_np = np.array(self.accumulated_points)
        colors_np = np.array(self.accumulated_colors)

        # Open3D를 이용한 겹치는 점 제거 및 최적화 (Voxel Downsampling)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_np)
        pcd.colors = o3d.utility.Vector3dVector(colors_np)
        
        # 0.005 = 5mm 간격으로 점들을 깔끔하게 정리
        downpcd = pcd.voxel_down_sample(voxel_size=0.005)
        
        down_pts = np.asarray(downpcd.points)
        down_colors = np.asarray(downpcd.colors) * 255

        self.get_logger().info(f'📉 최적화 완료: 원본 점 {len(pts_np)}개 -> 병합 후 {len(down_pts)}개')
        self.get_logger().info('🌐 HTML 3D 파일 생성 중...')

        # Plotly를 이용해 3D HTML 그리기
        html_colors = [f'rgb({int(c[0])},{int(c[1])},{int(c[2])})' for c in down_colors]
        plot_data = [go.Scatter3d(
            x=down_pts[:, 0], y=down_pts[:, 1], z=down_pts[:, 2],
            mode='markers', marker=dict(size=1.5, color=html_colors)
        )]

        fig = go.Figure(data=plot_data)
        fig.update_layout(scene=dict(aspectmode='data', 
                                     xaxis_title='X (base_link)', 
                                     yaxis_title='Y (base_link)', 
                                     zaxis_title='Z (base_link)'), 
                          title=f"Stitched 3D Map (1m Limit, Total {self.capture_count} scans)")
        
        filename = 'stitched_3d_map.html'
        fig.write_html(filename)
        self.get_logger().info(f'✅ 성공! [{filename}] 파일이 저장되었습니다.')
        self.get_logger().info('=========================================')
        
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