import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, JointState
import sensor_msgs_py.point_cloud2 as pc2
from rclpy.qos import qos_profile_sensor_data

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
        self.voxel_size = 0.003       # 3mm 간격으로 격자 압축 (엄청나게 촘촘하고 정밀함)
        self.max_distance = 1.2       # 카메라 기준 1.2m 너머의 배경 노이즈 제거
        self.min_distance = 0.2       # 카메라 바로 앞 20cm 이내 노이즈 제거
        
        # 전체 점들을 누적해서 저장할 전역 마스터 도화지
        self.global_points = np.empty((0, 3), dtype=np.float32)
        self.global_colors = np.empty((0, 3), dtype=np.float32)
        
        # 가장 최근에 들어온 '단 한 프레임'을 임시 보관할 변수
        self.latest_points = None
        self.latest_colors = None
        self.current_q = None
        
        self.joint_sub = self.create_subscription(
            JointState, '/dsr01/joint_states', self.joint_callback, 10
        )
        self.pc_sub = self.create_subscription(
            PointCloud2, '/camera/camera/depth/color/points', self.pointcloud_callback, qos_profile_sensor_data
        )
        
        self.get_logger().info('🚀 수평 정밀 누적 매핑 노드가 시작되었습니다.')
        self.get_logger().info('📌 [사용 방법]')
        self.get_logger().info('  1. 로봇을 수평 이동 후 완전히 멈춥니다.')
        self.get_logger().info('  2. 이 터미널 창에서 소문자 [s]를 누르면 현재 정밀 데이터가 누적됩니다.')
        self.get_logger().info('  3. 스캔이 끝나면 [Ctrl + C]를 눌러 하나의 HTML 통합 파일로 저장합니다.')

    def joint_callback(self, msg):
        joint_dict = dict(zip(msg.name, msg.position))
        try:
            ordered_q = [
                joint_dict['joint_1'], joint_dict['joint_2'], joint_dict['joint_3'],
                joint_dict['joint_4'], joint_dict['joint_5'], joint_dict['joint_6']
            ]
            self.current_q = ordered_q
        except KeyError:
            pass

    def get_transform_matrix(self, x, y, z, rx, ry, rz):
        r = R.from_euler('xyz', [rx, ry, rz], degrees=False)
        mat = np.eye(4)
        mat[0:3, 0:3] = r.as_matrix()
        mat[0:3, 3] = [x, y, z]
        return mat

    def calculate_camera_transform(self, q):
        T01 = self.get_transform_matrix(0, 0, 0.1345, 0, 0, q[0])
        T12 = self.get_transform_matrix(0, 0.0062, 0, 0, -1.571, -1.571) @ self.get_transform_matrix(0, 0, 0, 0, 0, q[1])
        T23 = self.get_transform_matrix(0.411, 0, 0, 0, 0, 1.571) @ self.get_transform_matrix(0, 0, 0, 0, 0, q[2])
        T34 = self.get_transform_matrix(0, -0.368, 0, 1.571, 0, 0) @ self.get_transform_matrix(0, 0, 0, 0, 0, q[3])
        T45 = self.get_transform_matrix(0, 0, 0, -1.571, 0, 0) @ self.get_transform_matrix(0, 0, 0, 0, 0, q[4])
        T56 = self.get_transform_matrix(0, -0.121, 0, 1.571, 0, 0) @ self.get_transform_matrix(0, 0, 0, 0, 0, q[5])
        
        T06 = T01 @ T12 @ T23 @ T34 @ T45 @ T56
        T_6_cam = self.get_transform_matrix(0, 0.07, 0.037, 0, 0, 0)
        return T06 @ T_6_cam

    def pointcloud_callback(self, msg):
        if self.current_q is None:
            return

        points, colors = [], []
        for p in pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True):
            # 거리 필터 전처리
            if self.min_distance <= p[2] <= self.max_distance:
                points.append([p[0], p[1], p[2], 1.0])
                
                rgb_float = p[3]
                rgb_bytes = struct.pack('f', rgb_float)
                r, g, b, _ = struct.unpack('BBBB', rgb_bytes)
                colors.append([r, g, b])
                
        if not points: 
            return
        
        # 덮어쓰기를 통해 항상 가장 신선한 멈춤 데이터 대기
        self.latest_points = np.array(points, dtype=np.float32)
        self.latest_colors = np.array(colors, dtype=np.float32)

    def accumulate_current_snapshot(self):
        """사용자가 's'를 눌렀을 때, 오차가 없는 찰나의 순간을 전역 마스터 도화지에 누적하는 함수"""
        if self.latest_points is None or self.current_q is None:
            self.get_logger().warn("⚠️ 아직 카메라나 로봇 데이터가 수신되지 않았습니다. 잠시 후 다시 누르세요.")
            return

        # 1. 정지한 순간의 로봇 기구학 위치 계산
        trans_matrix = self.calculate_camera_transform(self.current_q)
        
        # 2. 로봇 베이스 절대 좌표계로 변환
        transformed_pts = (trans_matrix @ self.latest_points.T).T[:, :3]
        colors_np = self.latest_colors

        # 3. 기존 마스터 도화지에 이어 붙이기 (누적)
        self.global_points = np.vstack((self.global_points, transformed_pts))
        self.global_colors = np.vstack((self.global_colors, colors_np))

        # 4. 누적될 때마다 복셀 다운샘플링을 수행하여 중복된 포인트 정리 (경량화 및 정밀화)
        voxels = np.round(self.global_points / self.voxel_size).astype(np.int32)
        _, unique_indices = np.unique(voxels, axis=0, return_index=True)
        
        self.global_points = self.global_points[unique_indices]
        self.global_colors = self.global_colors[unique_indices]

        self.get_logger().info(f"📥 [누적 성공] 현재 정밀 스냅샷이 병합되었습니다! (총 누적 점 개수: {len(self.global_points)}개)")

    def save_final_master_map(self):
        """Ctrl + C 클릭 시 누적된 모든 데이터를 단 하나의 PCD와 HTML 파일로 저장"""
        if len(self.global_points) == 0:
            self.get_logger().warn("⚠️ 저장할 누적 데이터가 존재하지 않습니다.")
            return

        self.get_logger().info("⏳ [최종 마스터 맵 생성] 누적된 모든 데이터를 가공하여 파일로 변환 중입니다...")

        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        pcd_filename = f"master_map_{now_str}.pcd"
        html_filename = f"master_map_{now_str}.html"

        # 1. 통합 PCD 원본 파일 작성
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
        self.get_logger().info(f"💾 1. 통합 원본 마스터 PCD 파일 저장 완료: {pcd_filename}")

        # 2. 통합 Plotly 3D 대화형 웹 HTML 파일 작성
        # 웹 브라우저 용량 최적화를 위해 1cm 간격으로 살짝 경량화하여 HTML 생성
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
            title=f"Doosan Robot Master 3D Map ({now_str})",
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
        self.get_logger().info(f"🎉 2. 대화형 통합 Plotly 3D HTML 뷰어 생성 성공: {html_filename}")


def main(args=None):
    rclpy.init(args=args)
    mapper = CumulativeSnapshotMapper()
    
    # 터미널 키보드 입력을 엔터 없이 즉시 낚아채기 위한 리눅스 환경 설정 (setcbreak)
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        
        while rclpy.ok():
            # ROS 2 토픽 주기적 수신 (0.01초 대기)
            rclpy.spin_once(mapper, timeout_sec=0.01)

            # 터미널 창에 입력된 키가 있는지 비동기로 확인
            if select.select([sys.stdin], [], [], 0.0)[0]:
                key = sys.stdin.read(1)
                if key == 's' or key == 'S':
                    # 사용자가 s를 누르면 누적 함수 실행!
                    mapper.accumulate_current_snapshot()

    except KeyboardInterrupt:
        # 사용자가 Ctrl + C를 누르면 종료 프로세스 전 최종 병합 파일 생성
        mapper.get_logger().info('⚠️ 사용자에 의해 노드가 정지되었습니다. 최종 맵 파일(PCD + HTML) 작성을 시작합니다.')
        mapper.save_final_master_map()
    finally:
        # 터미널 설정을 원래대로 깨끗하게 되돌림 (필수)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        mapper.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()