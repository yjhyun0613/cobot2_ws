import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String
from dsr_msgs2.srv import DrlStart
import DR_init
import time
import math
import numpy as np

# =================================================================
# [1] 환경 설정 (Configuration)
# =================================================================
ROBOT_ID    = "dsr01" 
ROBOT_MODEL = "m0609"
ROBOT_TOOL  = "Tool Weight"
ROBOT_TCP   = "GripperDA_v1"

RADIUS       = 49.0            
TOTAL_LAYERS = 5
CENTER       = np.array([425.15, 74.76, 57.0]) # [X, Y, Z]

SAFE_Z_HEIGHT = 200.0  # 대기 위치의 안전 Z 높이

# =================================================================
# [2] 수학 및 기하학 보조 함수 (Math & Kinematics)
# =================================================================
def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else v

def rot_to_zyz(R):
    beta = math.acos(max(min(R[2, 2], 1.0), -1.0))
    if abs(beta) < 1e-6:
        alpha, gamma = 0.0, math.atan2(R[1, 0], R[0, 0])
    else:
        alpha, gamma = math.atan2(R[1, 2], R[0, 2]), math.atan2(R[2, 1], -R[2, 0])
    return [math.degrees(alpha), math.degrees(beta), math.degrees(gamma)]

def wrap_angle(angle):
    while angle > 180.0: angle -= 360.0
    while angle < -180.0: angle += 360.0
    return angle

def make_continuous(base, target):
    diff = (target - base) % 360.0
    if diff > 180.0: diff -= 360.0
    return base + diff

def get_pure_blended_orientation(target_pos, center):
    """Z- 하방 벡터(75%)와 표면 법선 벡터(25%)를 내분하여 최적의 스캔 자세 반환"""
    pure_normal = -normalize(target_pos - center)
    down_z_axis = np.array([0.0, 0.0, -1.0])
    z_axis_final = normalize((down_z_axis * 3.0) + (pure_normal * 1.0))

    up = np.array([0.0, 1.0, 0.0]) if abs(z_axis_final[2]) > 0.95 else np.array([0.0, 0.0, 1.0])
    x_axis = normalize(np.cross(up, z_axis_final))
    y_axis = normalize(np.cross(z_axis_final, x_axis))
    
    rx, ry, rz_raw = rot_to_zyz(np.column_stack((x_axis, y_axis, z_axis_final)))
    return wrap_angle(rx), wrap_angle(ry), rz_raw

# =================================================================
# [3] DRL 경로 생성기 (Path Generator)
# =================================================================
def generate_dome_scan_drl(center, radius, total_layers):
    """모든 궤적을 파이썬에서 계산하여 시작 좌표들과 DRL 스크립트를 반환합니다."""
    cx, cy, cz = center

    # 1. 첫 번째 층(Start) 위치 및 회전각 계산 (하강 시 회전 방지용)
    angle_rad_1 = math.radians(90.0 * (1.0 - 1.0/total_layers))
    curr_z_1 = cz + radius * math.sin(angle_rad_1)
    curr_r_1 = radius * math.cos(angle_rad_1)
    
    start_1_xyz = np.array([cx - curr_r_1, cy, curr_z_1])
    s1_rx, s1_ry, _ = get_pure_blended_orientation(start_1_xyz, center)
    s1_rz = -180.0 # 초기 시작 무조건 -180도

    p_top = [cx, cy, SAFE_Z_HEIGHT, s1_rx, s1_ry, s1_rz]
    p_start_1 = [start_1_xyz[0], start_1_xyz[1], start_1_xyz[2], s1_rx, s1_ry, s1_rz]

    # 2. DRL 코드 뼈대 생성
    drl_code = "set_singularity_handling(1)\n"
    drl_code += "stiff = [3000, 3000, 300, 300, 300, 300]\n"
    drl_code += "task_compliance_ctrl(stiff)\n\n"

    # 3. 층별 지그재그 회전 로직 생성
    for i in range(2):
        angle_rad = math.radians(90.0 * (1.0 - float(i)/total_layers))
        curr_z = cz + radius * math.sin(angle_rad)
        curr_r = radius * math.cos(angle_rad)
        
        start_xyz = np.array([cx - curr_r, cy, curr_z])
        tgt_xyz   = np.array([cx + curr_r, cy, curr_z])
        
        # 층마다 회전 방향 교대 (±180 진자 운동)
        if (i % 2) == 1:
            via_xyz = np.array([cx, cy + curr_r, curr_z])
            direction = "CW"
            target_start_rz = -180.0
        else:
            via_xyz = np.array([cx, cy - curr_r, curr_z])
            direction = "CCW"
            target_start_rz = 180.0

        sr_x, sr_y, _ = get_pure_blended_orientation(start_xyz, center)
        vr_x, vr_y, vr_z_raw = get_pure_blended_orientation(via_xyz, center)
        tr_x, tr_y, tr_z_raw = get_pure_blended_orientation(tgt_xyz, center)

        # 회전 연속성 보장
        sr_z = target_start_rz
        vr_z = make_continuous(sr_z, vr_z_raw)
        tr_z = make_continuous(vr_z, tr_z_raw)

        # DRL 문자열 추가
        drl_code += f"# Layer {i} - {direction}\n"
        drl_code += f"p_start_{i} = posx({start_xyz[0]:.2f}, {start_xyz[1]:.2f}, {start_xyz[2]:.2f}, {sr_x:.2f}, {sr_y:.2f}, {sr_z:.2f})\n"
        drl_code += f"p_via_{i}   = posx({via_xyz[0]:.2f}, {via_xyz[1]:.2f}, {via_xyz[2]:.2f}, {vr_x:.2f}, {vr_y:.2f}, {vr_z:.2f})\n"
        drl_code += f"p_tgt_{i}   = posx({tgt_xyz[0]:.2f}, {tgt_xyz[1]:.2f}, {tgt_xyz[2]:.2f}, {tr_x:.2f}, {tr_y:.2f}, {tr_z:.2f})\n"
        drl_code += f"movel(p_start_{i}, v=40, a=80, r=10.0)\n"
        drl_code += f"movec(p_via_{i}, p_tgt_{i}, v=50, a=100, angle=360.0, ori=2)\n\n"

    drl_code += "release_compliance_ctrl()\n"
    
    return p_top, p_start_1, drl_code

# =================================================================
# [4] ROS2 통신 노드 (Node Communication)
# =================================================================
class ScanRunnerNode(Node):
    def __init__(self, node_name, namespace):
        super().__init__(node_name, namespace=namespace)
        self.state_pub = self.create_publisher(Bool, f'/{namespace}/checking_state', 10)
        self.progress_pub = self.create_publisher(String, f'/{namespace}/progress_status', 10)
        
        self.trigger_scan = False
        self.is_scanning = False 
        
        self.create_subscription(String, f'/{namespace}/progress_status', self.progress_callback, 10)
        
        srv_name = f'/{namespace}/drl/drl_start'
        self.drl_client = self.create_client(DrlStart, srv_name)
        while not self.drl_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f'{srv_name} 서비스 대기 중...')
        self.get_logger().info('✅ DRL 클라이언트 준비 완료')

    def progress_callback(self, msg):
        if msg.data == "02" and not self.is_scanning and not self.trigger_scan:
            self.trigger_scan = True

    def publish_state(self, state: bool):
        self.state_pub.publish(Bool(data=state))
        self.get_logger().info("🟢 기록 시작" if state else "🔴 기록 종료")

    def publish_completion(self):
        self.progress_pub.publish(String(data="03"))
        self.get_logger().info('🏁 작업 완료 신호(03) 송신')

    def send_drl_code(self, drl_script):
        req = DrlStart.Request()
        req.robot_system = 0        
        req.code = drl_script       
        self.drl_client.call_async(req)

# =================================================================
# [5] 메인 제어 흐름 (Main Flow)
# =================================================================
def main(args=None):
    DR_init.__dsr__id = ROBOT_ID
    DR_init.__dsr__model = ROBOT_MODEL
    
    rclpy.init(args=args)
    node = ScanRunnerNode('scan_runner_node', namespace=ROBOT_ID)
    DR_init.__dsr__node = node
     
    try:
        from DSR_ROBOT2 import set_robot_mode, set_tool, set_tcp, movej, movel, get_robot_state
        from DSR_ROBOT2 import ROBOT_MODE_AUTONOMOUS, ROBOT_MODE_MANUAL

        print("\n 로봇 초기화 중...")
        set_robot_mode(ROBOT_MODE_MANUAL)
        set_tool(ROBOT_TOOL)
        set_tcp(ROBOT_TCP)
        set_robot_mode(ROBOT_MODE_AUTONOMOUS)
        time.sleep(1.0) 

        print("\n 대기 모드: 신호(02) 대기 중...")

        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            
            # if node.trigger_scan:
            node.is_scanning = True
            node.trigger_scan = False
            
            # 1. 경로 및 스크립트 계산 (이전의 길었던 코드가 한 줄로 요약됨)
            p_top, p_start, drl_code = generate_dome_scan_drl(CENTER, RADIUS, TOTAL_LAYERS)
            
            # 2. 초기 안전 위치로 이동
            print("\n🚀 초기 위치 이동 중...")
            movej([0.0, 0.0, 90.0, 0.0, 90.0, 0.0], vel=100.0, acc=100.0)
            movel(p_top, vel=50.0, acc=100.0)
            
            # 3. 1층 스캔 지점으로 수직 하강
            print("📍 스캔 시작점으로 수직 하강 중...")
            movel(p_start, vel=40.0, acc=80.0)
            time.sleep(0.5)

            # 4. 상태 신호 켜고 DRL 전송
            node.publish_state(True)
            node.send_drl_code(drl_code)
            print("🛸 돔 스캐닝 진행 중...")
            
            # 5. 로봇 동작 완료까지 대기
            time.sleep(1.0) 
            stop_count = 0
            required_stop_time = 3.0 # 로봇이 3초 연속으로 멈춰있어야 종료로 인정
            check_interval = 0.5
            
            while rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.01)
                state = get_robot_state()
                
                if state == 1: # 로봇이 멈춘 상태(Idle)
                    stop_count += check_interval
                    if stop_count >= required_stop_time:
                        break
                else:
                    stop_count = 0 # 로봇이 다시 움직이면 카운터 리셋!
                
                time.sleep(check_interval)

            # 6. 작업 종료 및 복귀
            node.publish_state(False)
            node.publish_completion()
            print("\n🔄 1 사이클 완료. 다음 신호 대기 중...")
            node.is_scanning = False 
            time.sleep(1.0) 

    except Exception as e:
        print(f"\n🚨 에러 발생: {e}")
        node.publish_state(False) 
        
    finally:
        print("✅ 프로그램을 종료합니다.")
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()