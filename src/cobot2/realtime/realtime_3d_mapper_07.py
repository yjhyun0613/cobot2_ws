import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from rclpy.qos import qos_profile_sensor_data

# [핵심] 두산 로봇 공식 서비스 패키지 임포트
from dsr_msgs2.srv import GetCurrentPosx

import numpy as np
from scipy.spatial.transform import Rotation as R
import struct
import datetime
import plotly.graph_objects as go
import sys
import select
import termios
import tty

class ServiceSnapshotMapper(Node):
    def __init__(self):
        super().__init__('service_snapshot_mapper')
        
        # 전처리 설정
        self.voxel_size = 0.003
        self.max_distance = 1.2
        self.min_distance = 0.2
        
        # 누적 도화지
        self.global_points = np.empty((0, 3), dtype=np.float32)
        self.global_colors = np.empty((0, 3), dtype=np.float32)
        
        # 최신 카메라 메시지 임시 보관
        self.latest_msg = None
        
        # 1. 두산 로봇 위치 요청 클라이언트 생성
        self.pos_client = self.create_client(GetCurrentPosx, '/dsr01/aux_control/get_current_posx')
        
        # 2. 리얼센스 카메라 구독
        self.pc_sub = self.create_subscription(
            PointCloud2, 
            '/camera/camera/depth/color/points', 
            self.pointcloud_callback, 
            qos_profile_sensor_data
        )
        
        self.get_logger().info('🚀 두산 TCP 서비스 기반 누적 매퍼가 시작되었습니다!')
        self.get_logger().info('💡 수평 이동 후 멈춰서 [s]를 누르면 정교하게 맵이 누적됩니다.')

    def pointcloud_callback(self, msg):
        self.latest_msg = msg

    def accumulate_current_snapshot(self):
        """'s'를 눌렀을 때 실행되며, 두산 제어기에 서비스 요청을 보냄"""
        if self.latest_msg is None:
            self.get_logger().warn("⚠️ 수신된 카메라 데이터가 없습니다.")
            return

        if not self.pos_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error("⚠️ 두산 위치 서비스(/dsr01/aux_control/get_current_posx)에 연결할 수 없습니다.")
            return

        # 비동기 통신 중 데이터가 바뀌지 않도록 찰나의 카메라 데이터를 고정
        snapshot_msg = self.latest_msg

        self.get_logger().info("⏳ 두산 제어기에 정밀 좌표를 요청 중...")
        req = GetCurrentPosx.Request()
        req.ref = 0  # 0: 로봇 Base 좌표계 기준

        # 비동기 요청 전송 (터미널 멈춤 방지)
        future = self.pos_client.call_async(req)
        # 응답이 오면 process_snapshot_future 함수를 실행하도록 연결
        future.add_done_callback(lambda f: self.process_snapshot_future(f, snapshot_msg))

    def process_snapshot_future(self, future, msg):
        """두산 제어기에서 X,Y,Z,A,B,C 정답이 도착하면 실행되는 융합 로직"""
        try:
            response = future.result()
            if not response.success:
                self.get_logger().warn("⚠️ 제어기에서 위치를 가져오는데 실패했습니다.")
                return

            # 두산 서비스 응답 데이터 파싱 [X, Y, Z, A, B, C, 솔루션]
            data = response.task_pos_info[0].data
            
            # mm 단위를 m 단위로 변환
            x, y, z = data[0] / 1000.0, data[1] / 1000.0, data[2] / 1000.0
            
            # A, B, C 각도 (Degree 유지)
            a, b, c = data[3], data[4], data[5]
            
        except Exception as e:
            self.get_logger().error(f"⚠️ 서비스 응답 처리 중 에러 발생: {e}")
            return

        # 1. 로봇 끝단(TCP)의 4x4 변환 행렬 생성
        T_base_tcp = np.eye(4)
        
        # [주의] 두산 로봇의 A, B, C는 기본적으로 x, y, z 축 회전(Roll-Pitch-Yaw)을 의미합니다.
        # 만약 스캔 시 회전이 이상하게 꼬인다면 'xyz' 부분을 'ZYZ' 또는 'ZYX'로 바꿔보세요.
        r_tcp = R.from_euler('xyz', [a, b, c], degrees=True)
        
        T_base_tcp[0:3, 0:3] = r_tcp.as_matrix()
        T_base_tcp[0:3, 3] = [x, y, z]

        # 2. 로봇 끝단 -> 카메라 렌즈 거리 오프셋 (수동 적용)
        T_tcp_cam = np.eye(4)
        T_tcp_cam[0:3, 3] = [0, 0.07, 0.037] # X, Y, Z (m)

        # 3. 카메라 광학 렌즈 회전 보정 (Y축 반전 및 Z축 앞보기 현상 완벽 해결)
        r_optical = R.from_euler('xyz', [np.pi, 0, 0], degrees=False)
        T_optical = np.eye(4)
        T_optical[0:3, 0:3] = r_optical.as_matrix()

        # 4. 세 가지 행렬을 모두 융합하여 최종 '카메라 렌즈 절대 위치 행렬' 완성
        trans_matrix = T_base_tcp @ T_tcp_cam @ T_optical

        # 5. 카메라 점구름 파싱 및 거리 필터링
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

        # 6. 최종 융합 행렬을 적용하여 점들을 로봇 베이스 절대 좌표로 던짐
        transformed_pts = (trans_matrix @ pts_np.T).T[:, :3]

        # 7. 마스터 도화지에 누적 후 복셀 압축 (중복 제거)
        self.global_points = np.vstack((self.global_points, transformed_pts))
        self.global_colors = np.vstack((self.global_colors, colors_np))

        voxels = np.round(self.global_points / self.voxel_size).astype(np.int32)
        _, unique_indices = np.unique(voxels, axis=0, return_index=True)
        self.global_points = self.global_points[unique_indices]
        self.global_colors = self.global_colors[unique_indices]

        self.get_logger().info(f"📥 [누적 완료] 서비스 정답 기반 데이터 누적됨 (총: {len(self.global_points)}개)")

    def save_final_master_map(self):
        """Ctrl + C 클릭 시 누적된 맵을 통합 파일로 저장"""
        if len(self.global_points) == 0:
            self.get_logger().warn("⚠️ 저장할 누적 데이터가 없습니다.")
            return

        self.get_logger().info("⏳ [최종 마스터 맵 생성] 데이터를 파일로 굽는 중...")
        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. PCD 저장
        pcd_filename = f"master_service_{now_str}.pcd"
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

        # 2. HTML 저장
        html_filename = f"master_service_{now_str}.html"
        
        # 웹용 경량화 (1cm 압축)
        voxels_web = np.round(self.global_points / 0.01).astype(np.int32)
        _, web_indices = np.unique(voxels_web, axis=0, return_index=True)
        web_points = self.global_points[web_indices]
        web_colors = self.global_colors[web_indices]

        plotly_colors = [f'rgb({int(c[0])}, {int(c[1])}, {int(c[2])})' for c in web_colors]
        fig = go.Figure(data=[go.Scatter3d(
            x=web_points[:, 0], y=web_points[:, 1], z=web_points[:, 2],
            mode='markers', marker=dict(size=2.5, color=plotly_colors, opacity=0.9)
        )])
        
        fig.update_layout(
            title=f"Doosan Robot Service Map ({now_str})",
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
        self.get_logger().info(f"🎉 2. 대화형 Plotly 3D HTML 뷰어 생성 성공: {html_filename}")


def main(args=None):
    rclpy.init(args=args)
    mapper = ServiceSnapshotMapper()
    
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
        mapper.get_logger().info('⚠️ 종료 신호 수신됨. 맵 파일을 저장합니다.')
        mapper.save_final_master_map()
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        mapper.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()