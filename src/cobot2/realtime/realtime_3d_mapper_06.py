import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from rclpy.qos import qos_profile_sensor_data
from tf2_ros import Buffer, TransformListener # [핵심 추가] ROS 2 공식 좌표 변환기

import numpy as np
from scipy.spatial.transform import Rotation as R
import struct
import datetime
import plotly.graph_objects as go
import sys
import select
import termios
import tty

class CumulativeSnapshotMapper(Node):
    def __init__(self):
        super().__init__('cumulative_snapshot_mapper')
        
        # 초정밀 전처리 설정
        self.voxel_size = 0.003
        self.max_distance = 1.2
        self.min_distance = 0.2
        
        # 전역 마스터 도화지
        self.global_points = np.empty((0, 3), dtype=np.float32)
        self.global_colors = np.empty((0, 3), dtype=np.float32)
        
        # 최신 카메라 데이터 보관용
        self.latest_msg = None
        
        # [핵심 추가] 수동 조인트 수식을 버리고, ROS 2의 공식 TF(좌표계) 수신기를 장착합니다.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # [수정됨] 조인트 구독자(JointState)는 더 이상 필요 없으므로 삭제하고 카메라만 구독합니다.
        self.pc_sub = self.create_subscription(
            PointCloud2, '/camera/camera/depth/color/points', self.pointcloud_callback, qos_profile_sensor_data
        )
        
        self.get_logger().info('🚀 TF2 기반 정밀 누적 매핑 노드가 시작되었습니다! (축 왜곡/비율 문제 해결)')
        self.get_logger().info('📌 [사용 방법]')
        self.get_logger().info('  1. 로봇을 수평 이동 후 완전히 멈춥니다.')
        self.get_logger().info('  2. 터미널 창에서 소문자 [s]를 누르면 완벽하게 정렬된 데이터가 누적됩니다.')
        self.get_logger().info('  3. 스캔이 끝나면 [Ctrl + C]를 눌러 저장합니다.')

    def pointcloud_callback(self, msg):
        # 배경에서 항상 점들을 파싱하면 컴퓨터가 느려집니다.
        # 가장 최신 메시지 '원본'만 쥐고 있다가, 사용자가 's'를 누를 때만 파싱하도록 최적화했습니다.
        self.latest_msg = msg

    def accumulate_current_snapshot(self):
        """'s'를 눌렀을 때, ROS 2 공식 TF 좌표를 가져와 한 치의 오차 없이 마스터 도화지에 누적"""
        if self.latest_msg is None:
            self.get_logger().warn("⚠️ 아직 카메라 데이터가 수신되지 않았습니다.")
            return

        msg = self.latest_msg
        
        # 로봇 베이스 프레임 (두산 로봇은 보통 'base_0'을 사용합니다. 혹시 에러가 나면 'base_link'로 바꿔보세요)
        target_frame = 'base_link' 
        # 카메라 렌즈 프레임 (메시지에서 자동으로 읽어옵니다)
        source_frame = msg.header.frame_id

        try:
            # 1. ROS 2 시스템에 "지금 당장 카메라 렌즈부터 로봇 베이스까지의 거리와 각도 정답을 내놔!" 라고 요청
            trans = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time() # 가장 최신 시간
            )
        except Exception as e:
            self.get_logger().warn(f"⚠️ 로봇 좌표(TF)를 기다리는 중입니다... ({e})")
            return

        self.get_logger().info("✅ 정확한 로봇 3D 좌표를 획득했습니다. 데이터 파싱 중...")

        # 2. 받아온 정답지(Quaternion)를 4x4 변환 행렬로 만들기
        trans_matrix = np.eye(4)
        trans_matrix[0:3, 0:3] = R.from_quat([
            trans.transform.rotation.x, trans.transform.rotation.y,
            trans.transform.rotation.z, trans.transform.rotation.w
        ]).as_matrix()
        trans_matrix[0, 3] = trans.transform.translation.x
        trans_matrix[1, 3] = trans.transform.translation.y
        trans_matrix[2, 3] = trans.transform.translation.z

        # 3. 그제야 점 데이터를 꺼내서 필터링 (속도 대폭 향상)
        points, colors = [], []
        for p in pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True):
            if self.min_distance <= p[2] <= self.max_distance:
                points.append([p[0], p[1], p[2], 1.0])
                rgb_float = p[3]
                rgb_bytes = struct.pack('f', rgb_float)
                r, g, b, _ = struct.unpack('BBBB', rgb_bytes)
                colors.append([r, g, b])
                
        if not points: 
            self.get_logger().warn("⚠️ 거리 필터링 후 남은 점이 없습니다.")
            return

        pts_np = np.array(points, dtype=np.float32)
        colors_np = np.array(colors, dtype=np.float32)

        # 4. 카메라 시점의 점들을 완벽한 행렬을 이용해 로봇 베이스 절대 좌표계로 이동
        transformed_pts = (trans_matrix @ pts_np.T).T[:, :3]

        # 5. 마스터 도화지에 누적 및 복셀 압축
        self.global_points = np.vstack((self.global_points, transformed_pts))
        self.global_colors = np.vstack((self.global_colors, colors_np))

        voxels = np.round(self.global_points / self.voxel_size).astype(np.int32)
        _, unique_indices = np.unique(voxels, axis=0, return_index=True)
        
        self.global_points = self.global_points[unique_indices]
        self.global_colors = self.global_colors[unique_indices]

        self.get_logger().info(f"📥 [누적 완료] 축이 완벽하게 정렬되었습니다! (총 누적 점 개수: {len(self.global_points)}개)")

    def save_final_master_map(self):
        """Ctrl + C 클릭 시 누적된 모든 데이터를 저장"""
        if len(self.global_points) == 0:
            self.get_logger().warn("⚠️ 저장할 누적 데이터가 존재하지 않습니다.")
            return

        self.get_logger().info("⏳ [최종 마스터 맵 생성] 데이터를 가공하여 파일로 변환 중입니다...")

        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        pcd_filename = f"master_map_TF_{now_str}.pcd"
        html_filename = f"master_map_TF_{now_str}.html"

        num_points = len(self.global_points)
        pcd_header = f"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z rgb
SIZE 4 4 4 4
TYPE F F F F
COUNT 1 1 1 1
WIDTH {num_points}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {num_points}
DATA ascii
"""
        with open(pcd_filename, 'w') as f:
            f.write(pcd_header)
            for p, c in zip(self.global_points, self.global_colors):
                rgb = struct.unpack('f', struct.pack('BBBB', int(c[2]), int(c[1]), int(c[0]), 255))[0]
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {rgb}\n")
        self.get_logger().info(f"💾 1. PCD 파일 저장 완료: {pcd_filename}")

        voxels_web = np.round(self.global_points / 0.01).astype(np.int32)
        _, web_indices = np.unique(voxels_web, axis=0, return_index=True)
        web_points = self.global_points[web_indices]
        web_colors = self.global_colors[web_indices]

        plotly_colors = [f'rgb({int(c[0])}, {int(c[1])}, {int(c[2])})' for c in web_colors]

        fig = go.Figure(data=[go.Scatter3d(
            x=web_points[:, 0], y=web_points[:, 1], z=web_points[:, 2],
            mode='markers',
            marker=dict(size=2.5, color=plotly_colors, opacity=0.9)
        )])

        fig.update_layout(
            title=f"Doosan Robot Perfect TF Map ({now_str})",
            scene=dict(
                aspectmode='data',
                xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)',
                xaxis=dict(backgroundcolor="rgb(20, 20, 20)", gridcolor="gray"),
                yaxis=dict(backgroundcolor="rgb(20, 20, 20)", gridcolor="gray"),
                zaxis=dict(backgroundcolor="rgb(20, 20, 20)", gridcolor="gray"),
            ),
            paper_bgcolor="black", font=dict(color="white")
        )

        fig.write_html(html_filename)
        self.get_logger().info(f"🎉 2. HTML 뷰어 생성 성공: {html_filename}")


def main(args=None):
    rclpy.init(args=args)
    mapper = CumulativeSnapshotMapper()
    
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        
        while rclpy.ok():
            rclpy.spin_once(mapper, timeout_sec=0.01)

            if select.select([sys.stdin], [], [], 0.0)[0]:
                key = sys.stdin.read(1)
                if key == 's' or key == 'S':
                    mapper.accumulate_current_snapshot()

    except KeyboardInterrupt:
        mapper.get_logger().info('⚠️ 종료 신호. 최종 맵 파일 작성을 시작합니다.')
        mapper.save_final_master_map()
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        mapper.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()