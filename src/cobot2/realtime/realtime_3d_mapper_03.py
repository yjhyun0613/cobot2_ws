import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField, JointState
import sensor_msgs_py.point_cloud2 as pc2
from rclpy.qos import qos_profile_sensor_data, QoSProfile
from std_msgs.msg import Header

import numpy as np
from scipy.spatial.transform import Rotation as R
import struct
import datetime

class PureRos3DMapper(Node):
    def __init__(self):
        super().__init__('pure_ros_3d_mapper')
        
        self.voxel_size = 0.01  # 연산 부하를 줄이기 위해 1cm 간격으로 설정
        
        # 전역 맵 데이터를 담을 순수 NumPy 배열
        self.global_points = np.empty((0, 3), dtype=np.float32)
        self.global_colors = np.empty((0, 3), dtype=np.float32)
        
        self.current_q = None
        
        self.joint_sub = self.create_subscription(JointState, '/dsr01/joint_states', self.joint_callback, 10)
        self.pc_sub = self.create_subscription(PointCloud2, '/camera/camera/depth/color/points', self.pointcloud_callback, qos_profile_sensor_data)
        
        # Foxglove나 RViz에서 볼 수 있도록 누적 맵을 퍼블리시
        self.map_pub = self.create_publisher(PointCloud2, '/accumulated_map', QoSProfile(depth=10))
        
        self.get_logger().info('🚀 순수 ROS 2 매핑 노드(Open3D 제거 버전)가 시작되었습니다.')
        self.get_logger().info('💡 Foxglove Studio에서 [/accumulated_map] 토픽을 띄워보세요!')

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

        # 1. 메시지에서 점과 색상 추출
        points, colors = [], []
        for p in pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True):
            points.append([p[0], p[1], p[2], 1.0]) # 4x4 행렬 곱을 위해 1.0 추가
            
            rgb_float = p[3]
            rgb_bytes = struct.pack('f', rgb_float)
            r, g, b, _ = struct.unpack('BBBB', rgb_bytes)
            colors.append([r, g, b])
            
        if not points: return
        
        pts_np = np.array(points, dtype=np.float32)
        colors_np = np.array(colors, dtype=np.float32)

        # 2. 카메라 뷰 좌표를 로봇 베이스 좌표계로 변환 (행렬 곱)
        trans_matrix = self.calculate_camera_transform(self.current_q)
        transformed_pts = (trans_matrix @ pts_np.T).T[:, :3]

        # 3. 전역 맵에 누적
        self.global_points = np.vstack((self.global_points, transformed_pts))
        self.global_colors = np.vstack((self.global_colors, colors_np))

        # 4. 순수 NumPy 기반 복셀 다운샘플링 (중복 점 제거 및 경량화)
        voxels = np.round(self.global_points / self.voxel_size).astype(np.int32)
        _, unique_indices = np.unique(voxels, axis=0, return_index=True)
        self.global_points = self.global_points[unique_indices]
        self.global_colors = self.global_colors[unique_indices]

        # 5. Foxglove/RViz용 토픽 발행
        self.publish_accumulated_map()

    def publish_accumulated_map(self):
        if len(self.global_points) == 0: return
        
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'base_link' # 두산 로봇 베이스 링크

        # 색상을 다시 float32 구조로 패킹
        packed_data = []
        for p, c in zip(self.global_points, self.global_colors):
            rgb = struct.unpack('f', struct.pack('BBBB', int(c[2]), int(c[1]), int(c[0]), 255))[0]
            packed_data.append([p[0], p[1], p[2], rgb])

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        
        pc2_msg = pc2.create_cloud(header, fields, packed_data)
        self.map_pub.publish(pc2_msg)

    def save_to_pcd(self, filename):
        """Open3D 없이 순수 파이썬으로 PCD 파일 작성"""
        if len(self.global_points) == 0:
            self.get_logger().warn("저장할 맵 데이터가 없습니다.")
            return
            
        num_points = len(self.global_points)
        header = f"""# .PCD v0.7 - Point Cloud Data file format
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
        with open(filename, 'w') as f:
            f.write(header)
            for p, c in zip(self.global_points, self.global_colors):
                rgb = struct.unpack('f', struct.pack('BBBB', int(c[2]), int(c[1]), int(c[0]), 255))[0]
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {rgb}\n")
                
        self.get_logger().info(f"💾 원본 PCD 저장 완료: {filename}")

def main(args=None):
    rclpy.init(args=args)
    mapper = PureRos3DMapper()
    
    try:
        rclpy.spin(mapper)
    except KeyboardInterrupt:
        mapper.get_logger().info('⚠️ 종료 신호 수신됨. 최종 맵(PCD) 저장을 시작합니다.')
    finally:
        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        mapper.save_to_pcd(f"pure_map_{now_str}.pcd")
        mapper.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()