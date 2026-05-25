import rclpy
import numpy as np
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float32MultiArray

# DR_init은 가장 먼저 임포트
import DR_init

# 💡 로봇 설정 상수
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"

class NormalLookAtListener:
    """
    Node를 상속받지 않고, main에서 완전히 생성된 node 객체를 전달받아 작동하는 클래스
    """
    def __init__(self, node):
        self.node = node
        
        # 💡 [수정 포인트 1] movej 대신 movejx를 임포트합니다!
        from DSR_ROBOT2 import movel, posx, movejx
        self.movel = movel
        self.posx = posx
        self.movejx = movejx  # X, Y, Z 좌표를 주고 관절 이동(부드러운 곡선)으로 가는 명령어
        
        self.trigger_sub = self.node.create_subscription(
            Float32MultiArray,
            '/robot/trigger_normal_lookat',
            self.trigger_callback,
            10
        )
        
        self.is_moving = False
        self.node.get_logger().info('🚀 DSR Control 준비 완료! /robot/trigger_normal_lookat 대기 중...')

    def trigger_callback(self, msg):
        """외부에서 토픽을 퍼블리시할 때마다 실행되는 콜백 함수"""
        if self.is_moving:
            self.node.get_logger().warning('⚠️ 로봇이 이동 중입니다. 새 명령을 무시합니다.')
            return

        if len(msg.data) < 6:
            self.node.get_logger().error('❌ 데이터 부족! [x, y, z, nx, ny, nz] 총 6개의 값이 들어와야 합니다.')
            return
            
        target_pt = msg.data[0:3]
        normal_vec = msg.data[3:6]
        
        self.node.get_logger().info(f'📥 콜백 수신 -> 목표점: {target_pt}, 법선벡터: {normal_vec}')
        self.send_move_command(target_pt, normal_vec)

    def calculate_lookat_pose(self, target_pt, normal_vec, standoff=100.0):
        """법선 벡터를 기반으로 로봇이 바라볼 100mm 대기 포즈 계산"""
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
        """계산된 포즈로 두산 API (movejx)를 호출하여 부드러운 관절 이동 수행"""
        pose = self.calculate_lookat_pose(target_pt, normal_vec, standoff=100.0)
        if pose is None:
            self.node.get_logger().error("영벡터가 입력되어 계산할 수 없습니다.")
            return
            
        x, y, z, rx, ry, rz = pose
        target_pos = self.posx(x, y, z, rx, ry, rz)
        
        self.node.get_logger().info(f'🤖 DSR API movejx 실행: X:{x:.1f}, Y:{y:.1f}, Z:{z:.1f}')
        
        self.is_moving = True
        try:
            # 💡 [수정 포인트 2] movej(target_pos)를 movejx(target_pos)로 변경!
            self.movejx(target_pos, v=30, a=30)
        except Exception as e:
            self.node.get_logger().error(f"이동 중 에러 발생: {e}")
        finally:
            self.is_moving = False

def main(args=None):
    rclpy.init(args=args)
    
    # 1. rclpy를 이용해 순수 ROS 2 노드부터 먼저 생성
    node = rclpy.create_node('normal_lookat_listener_node', namespace=ROBOT_ID)
    
    # 2. 노드가 생성된 후 전역 설정(DR_init)에 확실하게 매핑
    DR_init.__dsr__id = ROBOT_ID
    DR_init.__dsr__model = ROBOT_MODEL
    DR_init.__dsr__node = node
    
    # 3. 매핑이 완벽히 끝난 상태에서 클래스 인스턴스화 (이때 DSR_ROBOT2가 로드됨)
    listener = NormalLookAtListener(node)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()