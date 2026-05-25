import rclpy
from rclpy.node import Node
import numpy as np
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float32MultiArray
from dsr_msgs2.srv import DrlStart

class NormalLookAtListenerNode(Node):
    def __init__(self):
        super().__init__('normal_lookat_listener_node')
        
        # 1. 두산 로봇 제어기에 스크립트를 던질 서비스 클라이언트
        self.drl_start_cli = self.create_client(DrlStart, '/dsr01/system/drl_start')
        while not self.drl_start_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('🤖 두산 DRL 서비스 연결 대기 중...')
            
        # 2. 외부에서 "나 계산해서 로봇 움직여줘!" 하고 호출할 수 있는 토픽 채널 개설
        # 데이터 규격: [target_x, target_y, target_z, normal_x, normal_y, normal_z] (총 6개 소수점 리스트)
        self.trigger_sub = self.create_subscription(
            Float32MultiArray,
            '/robot/trigger_normal_lookat',
            self.trigger_callback,
            10
        )
        
        self.get_logger().info('🚀 준비 완료! /robot/trigger_normal_lookat 토픽이 수신될 때마다 움직입니다.')

    def trigger_callback(self, msg):
        """
        외부에서 토픽을 퍼블리시(Call)할 때마다 실시간으로 실행되는 콜백 함수
        """
        if len(msg.data) < 6:
            self.get_logger().error('❌ 데이터 부족! [x, y, z, nx, ny, nz] 총 6개의 값이 들어와야 합니다.')
            return
            
        # 데이터 분리
        target_pt = msg.data[0:3]
        normal_vec = msg.data[3:6]
        
        self.get_logger().info(f'📥 콜백 수신 -> 목표점: {target_pt}, 법선벡터: {normal_vec}')
        
        # 100mm 대기 자세 계산 및 DRL 전송
        self.send_move_command(target_pt, normal_vec)

    def calculate_lookat_pose(self, target_pt, normal_vec, standoff=100.0):
        n = np.array(normal_vec, dtype=float)
        if np.linalg.norm(n) == 0:
            return None
        n = n / np.linalg.norm(n)

        target_pt = np.array(target_pt, dtype=float)
        obs_pt = target_pt + (n * standoff)

        z_vec = -n
        world_z = np.array([0.0, 0.0, 1.0])
        
        if np.abs(np.dot(world_z, z_vec)) > 0.999:
            world_z = np.array([1.0, 0.0, 0.0])

        y_vec = np.cross(world_z, z_vec)
        y_vec = y_vec / np.linalg.norm(y_vec)
        x_vec = np.cross(y_vec, z_vec)
        x_vec = x_vec / np.linalg.norm(x_vec)

        rot_mat = np.column_stack((x_vec, y_vec, z_vec))
        r = R.from_matrix(rot_mat)
        rx, ry, rz = r.as_euler('xyz', degrees=True)

        return [obs_pt[0], obs_pt[1], obs_pt[2], rx, ry, rz]

    def send_move_command(self, target_pt, normal_vec):
        pose = self.calculate_lookat_pose(target_pt, normal_vec, standoff=100.0)
        if pose is None:
            self.get_logger().error("영벡터가 입력되어 계산할 수 없습니다.")
            return
            
        x, y, z, rx, ry, rz = pose
        
        # 동적으로 계산된 좌표값을 품은 DRL 스크립트 껍데기 생성
        drl_code = f"""
target_pos = posx({x:.2f}, {y:.2f}, {z:.2f}, {rx:.2f}, {ry:.2f}, {rz:.2f})
movel(target_pos, v=50, a=50)
"""
        req = DrlStart.Request()
        req.robot_system = 1  # 1: Real 모드 (테스트시 상황에 맞춰 변경)
        req.code = drl_code
        
        self.get_logger().info(f'🤖 계산된 포즈 전송: X:{x:.1f}, Y:{y:.1f}, Z:{z:.1f}')
        self.drl_start_cli.call_async(req)

def main(args=None):
    rclpy.init(args=args)
    node = NormalLookAtListenerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()