import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, JointState
import sensor_msgs_py.point_cloud2 as pc2
from rclpy.qos import qos_profile_sensor_data

import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
import struct
import datetime
import plotly.graph_objects as go  # [추가] Plotly 라이브러리 임포트

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
        
        self.get_logger().info('🚀 3D 매핑 노드가 시작되었습니다.')
        self.get_logger().info('💡 3D 창을 클릭한 상태에서 [S] 키를 누르면 PCD와 HTML 뷰어가 동시 저장됩니다.')

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

    def save_map_data(self, prefix="map"):
        """PCD 원본 파일과 Plotly HTML 뷰어를 동시에 저장하는 통합 함수"""
        if self.global_pcd.is_empty():
            self.get_logger().warn("저장 실패: 누적된 데이터가 없습니다.")
            return

        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        pcd_filename = f"{prefix}_{now_str}.pcd"
        html_filename = f"{prefix}_{now_str}.html"
        
        # 1. 원본 PCD 저장
        o3d.io.write_point_cloud(pcd_filename, self.global_pcd)
        self.get_logger().info(f"💾 원본 PCD 저장 완료: {pcd_filename}")

        # 2. Plotly HTML 뷰어 굽기 시작
        self.get_logger().info(f"⏳ HTML 3D 웹 뷰어 생성 중... 잠시만 대기해 주세요.")
        
        # 웹 브라우저 최적화를 위해 점 간격을 1cm로 약간 낮춰서 복사 (원본 맵은 손상 안 됨)
        web_pcd = self.global_pcd.voxel_down_sample(voxel_size=0.01)
        points = np.asarray(web_pcd.points)
        colors = np.asarray(web_pcd.colors)
        
        if len(points) > 0:
            plotly_colors = [f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})' for r, g, b in colors]

            fig = go.Figure(data=[go.Scatter3d(
                x=points[:, 0], y=points[:, 1], z=points[:, 2],
                mode='markers',
                marker=dict(size=3, color=plotly_colors, opacity=0.8)
            )])

            fig.update_layout(
                title=f"Doosan M0609 + RealSense Map",
                scene=dict(
                    aspectmode='data',
                    xaxis=dict(backgroundcolor="rgb(30, 30, 30)", gridcolor="white"),
                    yaxis=dict(backgroundcolor="rgb(30, 30, 30)", gridcolor="white"),
                    zaxis=dict(backgroundcolor="rgb(30, 30, 30)", gridcolor="white"),
                ),
                paper_bgcolor="black", font=dict(color="white")
            )

            # HTML 파일로 출력
            fig.write_html(html_filename)
            self.get_logger().info(f"🎉 HTML 웹 뷰어 저장 완료: {html_filename}")
        else:
            self.get_logger().warn("경량화 후 남은 데이터가 없어 HTML을 생성하지 않았습니다.")


def main(args=None):
    rclpy.init(args=args)
    mapper = Realtime3DMapper()
    
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Doosan M0609 + RealSense 3D Map Player", width=1024, height=768)
    
    vis.add_geometry(mapper.global_pcd)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
    vis.add_geometry(axis)

    # 단축키 S 감지
    def key_s_callback(visualizer):
        mapper.save_map_data(prefix="snapshot")
        return False

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
        mapper.get_logger().info('⚠️ 종료 신호 수신됨. 최종 맵(PCD + HTML) 저장을 시작합니다.')
    finally:
        # 프로그램 종료 직전 최종본 동시 저장
        mapper.save_map_data(prefix="final_map")
        
        vis.destroy_window()
        mapper.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()