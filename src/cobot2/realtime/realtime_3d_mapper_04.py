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

class PrecisionSnapshotMapper(Node):
    def __init__(self):
        super().__init__('precision_snapshot_mapper')
        
        # 정밀 전처리 설정
        self.voxel_size = 0.003       # 3mm 간격으로 매우 정밀하게 압축 (기존 5mm~10mm보다 정밀)
        self.max_distance = 1.2       # 카메라 기준 1.2m 보다 먼 배경 노이즈는 전부 잘라버림 (전처리 핵심)
        self.min_distance = 0.2       # 카메라 너무 앞쪽(20cm 이내) 노이즈 제거
        
        # 실시간 누적이 아닌, 가장 최근의 '딱 한 프레임'만 저장할 변수
        self.latest_points = None
        self.latest_colors = None
        self.current_q = None
        
        self.joint_sub = self.create_subscription(
            JointState, 
            '/dsr01/joint_states', 
            self.joint_callback, 
            10
        )
        
        self.pc_sub = self.create_subscription(
            PointCloud2, 
            '/camera/camera/depth/color/points', 
            self.pointcloud_callback, 
            qos_profile_sensor_data
        )
        
        self.get_logger().info('🚀 정밀 3D 스냅샷 매퍼가 시작되었습니다. (Open3D 제거 완료)')
        self.get_logger().info('💡 로봇을 멈추고 터미널에서 [Ctrl + C]를 누르면 그 순간이 정교하게 HTML로 저장됩니다.')

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
            # [전처리 1] 카메라 기준 너무 멀거나 가까운 배경 노이즈 1차 필터링
            if self.min_distance <= p[2] <= self.max_distance:
                points.append([p[0], p[1], p[2], 1.0])
                
                rgb_float = p[3]
                rgb_bytes = struct.pack('f', rgb_float)
                r, g, b, _ = struct.unpack('BBBB', rgb_bytes)
                colors.append([r, g, b])
                
        if not points: 
            return
        
        # 누적이 아니라, 항상 가장 최신의 정교한 단일 프레임으로 덮어씌웁니다.
        self.latest_points = np.array(points, dtype=np.float32)
        self.latest_colors = np.array(colors, dtype=np.float32)

    def process_and_save_snapshot(self):
        """Ctrl+C가 감지되면 마지막 단 한 프레임을 정밀 가공하여 저장하는 함수"""
        if self.latest_points is None or self.current_q is None:
            self.get_logger().warn("⚠️ 수신된 데이터가 없어 스냅샷을 저장하지 못했습니다.")
            return

        self.get_logger().info("⏳ 찰나의 스냅샷 데이터를 기반으로 정밀 전처리를 시작합니다...")

        # 1. 기구학 수식을 딱 이 순간의 각도로 정밀 계산
        trans_matrix = self.calculate_camera_transform(self.current_q)
        
        # 2. 3D 점들을 로봇 베이스 절대 좌표계로 이동
        transformed_pts = (trans_matrix @ self.latest_points.T).T[:, :3]
        colors_np = self.latest_colors

        # 3. [전처리 2] NumPy 기반 정밀 복셀 다운샘플링 (중복된 점 제거 및 초정밀 고밀도화)
        voxels = np.round(transformed_pts / self.voxel_size).astype(np.int32)
        _, unique_indices = np.unique(voxels, axis=0, return_index=True)
        
        final_points = transformed_pts[unique_indices]
        final_colors = colors_np[unique_indices]

        # 4. 파일 이름 결정 (현재 시간 기준)
        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        pcd_filename = f"snapshot_{now_str}.pcd"
        html_filename = f"snapshot_{now_str}.html"

        # 5. 순수 파이썬 데이터 포맷으로 PCD 원본 저장
        num_points = len(final_points)
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
            for p, c in zip(final_points, final_colors):
                rgb = struct.unpack('f', struct.pack('BBBB', int(c[2]), int(c[1]), int(c[0]), 255))[0]
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {rgb}\n")
        self.get_logger().info(f"💾 1. 원본 정밀 PCD 파일 저장 완료: {pcd_filename}")

        # 6. Plotly를 활용해 화려한 3D 웹 HTML 문서로 굽기
        plotly_colors = [f'rgb({int(c[0])}, {int(c[1])}, {int(c[2])})' for c in final_colors]

        fig = go.Figure(data=[go.Scatter3d(
            x=final_points[:, 0], y=final_points[:, 1], z=final_points[:, 2],
            mode='markers',
            marker=dict(
                size=2.5,                 # 점 크기를 정밀하게 조절
                color=plotly_colors,
                opacity=0.9
            )
        )])

        fig.update_layout(
            title=f"Doosan Robot AI Scan Snapshot ({now_str})",
            scene=dict(
                aspectmode='data',
                xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)',
                xaxis=dict(backgroundcolor="rgb(20, 20, 20)", gridcolor="gray"),
                yaxis=dict(backgroundcolor="rgb(20, 20, 20)", gridcolor="gray"),
                zaxis=dict(backgroundcolor="rgb(20, 20, 20)", gridcolor="gray"),
            ),
            paper_bgcolor="black",
            font=dict(color="white")
        )

        fig.write_html(html_filename)
        self.get_logger().info(f"🎉 2. 대화형 Plotly 3D HTML 뷰어 생성 성공: {html_filename}")


def main(args=None):
    rclpy.init(args=args)
    mapper = PrecisionSnapshotMapper()
    
    try:
        rclpy.spin(mapper)
    except KeyboardInterrupt:
        # 사용자가 Ctrl + C를 누르면 딱 그 순간 멈춰있던 마지막 장면을 가공함
        mapper.process_and_save_snapshot()
    finally:
        mapper.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()