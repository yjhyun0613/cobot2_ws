import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, JointState
import sensor_msgs_py.point_cloud2 as pc2
from rclpy.qos import qos_profile_sensor_data
#ㅁㄴㅇㄻㄴㅇㄻㄴㅇ
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
import struct
import datetime

class Realtime3DMapper(Node):
    def __init__(self):
        super().__init__('realtime_3d_mapper')
        
        self.voxel_size = 0.005 
        
        self.global_pcd = o3d.geometry.PointCloud()
        self.pcd_updated = False
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
        
        self.get_logger().info('🚀 3D 매핑 및 자동 저장 노드가 시작되었습니다.')
        self.get_logger().info('💡 3D 창을 클릭한 상태에서 [S] 키를 누르면 실시간 스냅샷이 저장됩니다.')

    def joint_callback(self, msg):
        joint_dict = dict(zip(msg.name, msg.position))
        try:
            ordered_q = [
                joint_dict['joint_1'],
                joint_dict['joint_2'],
                joint_dict['joint_3'],
                joint_dict['joint_4'],
                joint_dict['joint_5'],
                joint_dict['joint_6']
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

        trans_matrix = self.calculate_camera_transform(self.current_q)
        current_pcd = self.ros_to_o3d(msg)
        
        if current_pcd.is_empty():
            return

        current_pcd.transform(trans_matrix)
        
        self.global_pcd += current_pcd
        self.global_pcd = self.global_pcd.voxel_down_sample(self.voxel_size)
        self.pcd_updated = True

    def ros_to_o3d(self, ros_pc2):
        pcd = o3d.geometry.PointCloud()
        points, colors = [], []
        
        for p in pc2.read_points(ros_pc2, field_names=("x", "y", "z", "rgb"), skip_nans=True):
            points.append([p[0], p[1], p[2]])
            
            rgb_float = p[3]
            rgb_bytes = struct.pack('f', rgb_float)
            r, g, b, _ = struct.unpack('BBBB', rgb_bytes)
            colors.append([r / 255.0, g / 255.0, b / 255.0])
            
        if points:
            pcd.points = o3d.utility.Vector3dVector(np.array(points, dtype=np.float64))
            pcd.colors = o3d.utility.Vector3dVector(np.array(colors, dtype=np.float64))
        return pcd

    def save_pcd_with_timestamp(self, prefix="map"):
        """현재 시간을 조합하여 PCD 파일로 저장하는 함수"""
        if self.global_pcd.is_empty():
            self.get_logger().warn("저장 실패: 누적된 3D 점 데이터가 없습니다.")
            return

        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{now_str}.pcd"
        
        o3d.io.write_point_cloud(filename, self.global_pcd)
        self.get_logger().info(f"💾 파일이 성공적으로 저장되었습니다: {filename}")


def main(args=None):
    rclpy.init(args=args)
    mapper = Realtime3DMapper()
    
    # [수정됨] 키보드 이벤트를 받을 수 있는 특수 Visualizer 클래스로 변경!
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Doosan M0609 + RealSense 3D Map Player", width=1024, height=768)
    
    vis.add_geometry(mapper.global_pcd)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
    vis.add_geometry(axis)

    # [수정됨] 키보드 'S' 버튼 감지 콜백 설정 (ord('S') 사용)
    def key_s_callback(visualizer):
        mapper.save_pcd_with_timestamp(prefix="snapshot")
        return False

    # ord('S')는 키보드의 S키(대소문자 무관) 물리적 입력을 의미합니다.
    vis.register_key_callback(ord('S'), key_s_callback)

    try:
        while rclpy.ok():
            rclpy.spin_once(mapper, timeout_sec=0.01)

            if mapper.pcd_updated:
                vis.update_geometry(mapper.global_pcd)
                mapper.pcd_updated = False

            if not vis.poll_events():
                break
                
            vis.update_renderer()
            
    except KeyboardInterrupt:
        mapper.get_logger().info('⚠️ 사용자에 의해 프로그램이 정지되었습니다. 최종 맵 저장을 시작합니다.')
    finally:
        mapper.save_pcd_with_timestamp(prefix="final_map")
        
        vis.destroy_window()
        mapper.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()