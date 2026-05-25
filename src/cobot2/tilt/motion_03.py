import rclpy
import numpy as np
import math
from std_msgs.msg import Float32MultiArray

# DR_init은 가장 먼저 임포트
import DR_init

# =================================================================
# [1] 로봇 설정 상수
# =================================================================
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"

# =================================================================
# [2] 수학 및 기하학 보조 함수
# =================================================================
def normalize(v):
    """벡터 정규화 (길이를 1로 만듦)"""
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else v

def rot_to_zyz(R):
    """회전 행렬 → ZYZ 오일러 각도 변환 (두산 posx 규격)
    
    두산 posx(x, y, z, a, b, c) 에서 자세는 ZYZ:
      Rz(alpha) → Ry(beta) → Rz(gamma)
    반환: [alpha(deg), beta(deg), gamma(deg)]
    """
    beta = math.acos(max(min(R[2, 2], 1.0), -1.0))
    if abs(beta) < 1e-6:
        # 특이점: tool Z가 world Z와 거의 평행할 때
        alpha, gamma = 0.0, math.atan2(R[1, 0], R[0, 0])
    else:
        alpha = math.atan2(R[1, 2], R[0, 2])
        gamma = math.atan2(R[2, 1], -R[2, 0])
    return [math.degrees(alpha), math.degrees(beta), math.degrees(gamma)]

def wrap_angle(angle):
    """각도를 [-180, 180] 범위로 정규화"""
    while angle > 180.0: angle -= 360.0
    while angle < -180.0: angle += 360.0
    return angle


# =================================================================
# [3] 제어 로직 리스너 클래스
# =================================================================
class NormalLookAtListener:
    """
    Node를 상속받지 않고, main에서 생성된 node 객체를 전달받아 작동하는 클래스
    """
    def __init__(self, node):
        self.node = node
        
        # 두산 로봇 제어 API
        from DSR_ROBOT2 import movejx, posx
        self.movejx = movejx
        self.posx = posx
        
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
            
        target_pt = np.array(msg.data[0:3], dtype=float)
        normal_vec = np.array(msg.data[3:6], dtype=float)
        
        self.node.get_logger().info(f'📥 콜백 수신 -> 목표점: {target_pt}, 법선벡터: {normal_vec}')
        self.send_move_command(target_pt, normal_vec)

    def calculate_lookat_pose(self, target_pt, normal_vec, standoff=100.0):
        """
        법선 벡터 방향으로 100mm(10cm) 떨어진 지점 위치 계산 및 중심점을 바라보는 자세 생성
        """
        if np.linalg.norm(normal_vec) < 1e-6:
            return None
            
        n = normalize(normal_vec)
        
        # 1. 이동할 목표 위치: 타겟 중심 좌표에서 법선 벡터(n) 방향으로 100mm 이동
        obs_pt = target_pt + (n * standoff)

        # 2. 중심 좌표를 바라보는 방향 계산 (Tool Z축을 타겟 방향인 '-n'으로 설정)
        z_axis_final = -n
        
        # 3. Up 벡터 기준 설정 (Tool Z가 거의 수직이면 Y축을, 아니면 Z축을 Up으로 사용)
        if abs(z_axis_final[2]) > 0.95:
            up = np.array([0.0, 1.0, 0.0])
        else:
            up = np.array([0.0, 0.0, 1.0])
        
        # 4. 직교 좌표계(X, Y, Z) 구성
        x_axis = normalize(np.cross(up, z_axis_final))
        y_axis = normalize(np.cross(z_axis_final, x_axis))
        
        # 5. 회전 행렬을 두산 posx용 ZYZ 오일러 각도로 변환
        rot_matrix = np.column_stack((x_axis, y_axis, z_axis_final))
        rx, ry, rz = rot_to_zyz(rot_matrix)
        rx, ry, rz = wrap_angle(rx), wrap_angle(ry), wrap_angle(rz)

        self.node.get_logger().info(
            f'📐 계산결과 -> 도착지점: ({obs_pt[0]:.1f}, {obs_pt[1]:.1f}, {obs_pt[2]:.1f}), '
            f'자세: posx({rx:.1f}, {ry:.1f}, {rz:.1f})'
        )
        self.node.get_logger().info(
            f'   마주보는 방향(Tool Z): ({z_axis_final[0]:.3f}, {z_axis_final[1]:.3f}, {z_axis_final[2]:.3f})'
        )

        return [obs_pt[0], obs_pt[1], obs_pt[2], rx, ry, rz]

    def send_move_command(self, target_pt, normal_vec):
        """계산된 좌표와 자세로 로봇 이동 명령 전송"""
        pose = self.calculate_lookat_pose(target_pt, normal_vec, standoff=100.0)
        
        if pose is None:
            self.node.get_logger().error("영벡터가 입력되어 자세를 계산할 수 없습니다.")
            return
            
        x, y, z, rx, ry, rz = pose
        target_pos = self.posx(x, y, z, rx, ry, rz)
        
        self.node.get_logger().info(f'🤖 DSR API movejx 실행: 목표로 부드럽게 이동합니다.')
        
        self.is_moving = True
        try:
            # 안전성을 위해 속도 30으로 설정하여 로봇 이동
            self.movejx(target_pos, v=30, a=30) 
        except Exception as e:
            self.node.get_logger().error(f"이동 중 에러 발생: {e}")
        finally:
            self.is_moving = False

# =================================================================
# [4] 메인 함수
# =================================================================
def main(args=None):
    rclpy.init(args=args)
    
    node = rclpy.create_node('normal_lookat_listener_node', namespace=ROBOT_ID)
    
    DR_init.__dsr__id = ROBOT_ID
    DR_init.__dsr__model = ROBOT_MODEL
    DR_init.__dsr__node = node
    
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