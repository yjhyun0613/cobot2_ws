import rclpy
from rclpy.node import Node
import time
import sys

# 통신을 위한 커스텀 서비스 (가정)
# request: float32[] target_pos
# response: bool is_pass, float32 measured_torque
from custom_interfaces.srv import ScrewTask 

import DR_init
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

try:
    from DSR_ROBOT2 import (
        movej, movel, mwait, task_compliance_ctrl, set_desired_force, 
        release_compliance_ctrl, get_current_tool_force, get_current_posx
    )
except ImportError:
    sys.exit("두산 로봇 API 임포트 실패")

# 로봇 제어 설정값
VELOCITY, ACC = 60, 60
JHOME_POS = [0, 0, 90, 0, 90, 0]
TOOL_STATION_POS = [415.578, -245.516, 370.815, 37.934, -179.505, -51.596]# 예시 드라이버 위치
TORQUE_THRESHOLD = 5.0 # 정상 체결로 판독할 목표 토크값 (Nm)

class ScrewTighteningServer(Node):
    def __init__(self):
        super().__init__("screw_tightening_server")
        
        # 서비스 서버 생성 (메인 노드의 요청을 대기)
        self.srv = self.create_service(ScrewTask, '/execute_screw_tightening', self.tighten_screw_callback)
        self.get_logger().info("나사 체결 및 토크 검수 서버가 준비되었습니다.")

    def pick_and_place_tool(self, action="pick"):
        """드라이버 툴을 집거나 내려놓는 동작"""
        self.get_logger().info(f"드라이버 {action} 진행 중...")
        
        # 툴 스테이션 상단으로 이동
        approach_pos = list(TOOL_STATION_POS)
        approach_pos[2] += 100.0
        movel(approach_pos, vel=VELOCITY, acc=ACC)
        mwait()
        
        # 툴 스테이션으로 하강
        movel(TOOL_STATION_POS, vel=VELOCITY, acc=ACC)
        mwait()
        
        if action == "pick":
            # 그리퍼 닫기 (툴 결합) 로직 추가 
            # gripper.close()
            time.sleep(1.0)
        else:
            # 그리퍼 열기 (툴 반납) 로직 추가
            # gripper.open()
            time.sleep(1.0)
            
        # 다시 상단으로 복귀
        movel(approach_pos, vel=VELOCITY, acc=ACC)
        mwait()

    def tighten_screw_callback(self, request, response):
        """메인 노드로부터 좌표를 받아 체결 공정을 수행하는 콜백"""
        target_pos = request.target_pos
        self.get_logger().info(f"작업 지시 수신: 타겟 좌표 {target_pos[:3]}")

        # 1. 드라이버 Pick
        self.pick_and_place_tool(action="pick")

        # 2. 타겟 나사 위치 상단으로 이동
        target_pos_up = list(target_pos)
        target_pos_up[2] += 50.0
        movel(target_pos_up, vel=VELOCITY, acc=ACC)
        mwait()

        # 3. 타겟 나사 표면으로 접근
        movel(target_pos, vel=VELOCITY, acc=ACC)
        mwait()

        # ==========================================
        # 4. 순응 제어 및 힘 제어 활성화 (협동1 코드 응용)
        # ==========================================
        self.get_logger().info("순응 제어 활성화: 나사 압박 및 회전 시작")
        
        # X, Y축은 단단하게(3000), Z축은 부드럽게(500) 설정하여 툴이 튕기지 않고 나사선을 따라가게 함
        stx = [3000, 3000, 500, 200, 200, 200]
        task_compliance_ctrl(stx)
        
        # Z축(아래 방향)으로 -10N의 힘을 주어 누름 유지
        fd = [0, 0, -10, 0, 0, 0]
        f_dir = [0, 0, 1, 0, 0, 0] # Z축 방향 제어 활성화
        set_desired_force(fd, f_dir)
        time.sleep(1.0) # 힘이 인가될 때까지 잠시 대기

        # 5. 회전 (나사 조이기) 및 토크 검수
        current_pos = get_current_posx()[0]
        target_rot_pos = list(current_pos)
        target_rot_pos[5] += 360.0 # Rz 축으로 360도 회전 시도 (조임)
        
        # 비동기로 회전 명령을 내리고 실시간 토크(Mz) 측정
        # (참고: amove 계열이나 스레드를 쓸 수 있으나, 여기서는 단순화된 로직 적용)
        movel(target_rot_pos, vel=20, acc=20) 
        
        max_torque_measured = 0.0
        start_time = time.time()
        
        # 회전하는 동안(대략 3초) 실시간 조인트 토크 모니터링
        while time.time() - start_time < 3.0:
            forces = get_current_tool_force()
            mz_torque = abs(forces[5]) # Mz (Z축 회전 토크)
            
            if mz_torque > max_torque_measured:
                max_torque_measured = mz_torque
                
            time.sleep(0.1)

        # 회전 완료 후 순응 제어 해제
        release_compliance_ctrl()
        self.get_logger().info("순응 제어 해제")

        # 6. 불량 판독 (Threshold 기준)
        if max_torque_measured >= TORQUE_THRESHOLD:
            self.get_logger().info(f"체결 성공! (측정 토크: {max_torque_measured:.2f} Nm)")
            response.is_pass = True
        else:
            self.get_logger().warn(f"체결 불량 의심! 헛돌음 감지 (측정 토크: {max_torque_measured:.2f} Nm)")
            response.is_pass = False
            
        response.measured_torque = float(max_torque_measured)

        # 7. 안전 높이로 후퇴 및 툴 반납
        movel(target_pos_up, vel=VELOCITY, acc=ACC)
        mwait()
        
        self.pick_and_place_tool(action="place")
        
        # 홈 복귀
        movej(JHOME_POS, vel=VELOCITY, acc=ACC)
        mwait()

        return response

def main(args=None):
    rclpy.init(args=args)
    node = ScrewTighteningServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("서버가 종료됩니다.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()