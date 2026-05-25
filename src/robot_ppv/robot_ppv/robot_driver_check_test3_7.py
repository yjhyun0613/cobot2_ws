import os
import time
import sys
import traceback
from scipy.spatial.transform import Rotation
import numpy as np
import threading
import rclpy
from rclpy.node import Node
import DR_init

from od_msg.srv import SrvDepthPosition
from std_srvs.srv import Trigger
from ament_index_python.packages import get_package_share_directory
from robot_control.onrobot import RG

# 두산 로봇 설정 정보
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
VELOCITY, ACC = 60, 60
JHOME_POS = [0, 0, 90, 0, 90, 0]

# 그리퍼 및 툴체인저 설정
GRIPPER_NAME = "rg2"
TOOLCHARGER_IP = "192.168.1.1"
TOOLCHARGER_PORT = "502"

# [하드웨어 고정 좌표 설정]
TOOL_STATION_POS = [415.578, -245.516, 150.815, 37.934, -179.505, -51.596]   # 드라이버 거치 위치
TARGET_SCREW_POS = [545.1,  -14.8, 64.1, 0.0, 180.0, 0.0]                      # 테스트용 기본 나사 위치
TARGET_SCREW_POS[2] += 100
# 토크 검수 임계값 (Nm)
TORQUE_THRESHOLD = 5.0

# ROS 2 및 두산 API 바인딩 전역 선언
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

ROBOT_TOOL = "Tool Weight"
ROBOT_TCP = "GripperDA_v1"

rclpy.init()
dsr_node = rclpy.create_node("robot_screw_inspection_node", namespace=ROBOT_ID)
DR_init.__dsr__node = dsr_node

def initialize_robot():
    from DSR_ROBOT2 import (
        set_tool,
        set_tcp,
        get_tool,
        get_tcp,
        ROBOT_MODE_MANUAL,
        ROBOT_MODE_AUTONOMOUS,
        get_robot_mode,
        set_robot_mode
    )

    print("#" * 50, flush=True)
    print("[TENDERIZING_ONCE] Initializing robot", flush=True)
    print(f"ROBOT_ID: {ROBOT_ID}", flush=True)
    print(f"ROBOT_MODEL: {ROBOT_MODEL}", flush=True)
    print(f"ROBOT_TCP: {ROBOT_TCP}", flush=True)
    print(f"ROBOT_TOOL: {ROBOT_TOOL}", flush=True)
    print(f"VELOCITY: {VELOCITY}", flush=True)
    print(f"ACC: {ACC}", flush=True)
    print("#" * 50, flush=True)

    set_robot_mode(ROBOT_MODE_MANUAL)
    time.sleep(0.5)

    set_tool(ROBOT_TOOL)
    set_tcp(ROBOT_TCP)

    set_robot_mode(ROBOT_MODE_AUTONOMOUS)
    time.sleep(2.0)

    print("#" * 50, flush=True)
    print("[TENDERIZING_ONCE] Robot initialized", flush=True)
    print(f"ROBOT_TCP: {get_tcp()}", flush=True)
    print(f"ROBOT_TOOL: {get_tool()}", flush=True)
    print(f"ROBOT_MODE: {get_robot_mode()}", flush=True)
    print("#" * 50, flush=True)


try:
    from DSR_ROBOT2 import (
        movej, movel, get_current_posx, mwait, trans, posx,
        task_compliance_ctrl, set_desired_force, release_compliance_ctrl, get_tool_force, get_current_posj, release_force,
        amovej, amovel, check_motion
    )
except ImportError as e:
    sys.exit(f"두산 로봇 API 임포트 실패: {e}")

# 전역 그리퍼 객체 초기화
gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)


class RobotScrewInspector(Node):
    def __init__(self):
        super().__init__("robot_screw_inspector")
        
        # 로봇 초기 위치(홈)로 이동
        self.init_robot()
        
        # 비전 및 음성 캐시용 변수
        self.saved_positions = {"pos1": TARGET_SCREW_POS}  # 테스트용 기본 좌표 등록
        
        self.get_logger().info("=== 나사 체결 검수 시스템이 초기화되었습니다 ===")
        
        # [자동 실행 기능] 노드 실행 1초 후 자동으로 시퀀스 구동
        self.create_timer(1.0, self.auto_start_callback, callback_group=None)

    def init_robot(self):
        self.get_logger().info("로봇을 홈 위치로 이동합니다.")
        movej(JHOME_POS, vel=VELOCITY, acc=ACC)
        mwait()

    def auto_start_callback(self):
        self.get_logger().info("자동 테스트 스크립트를 시작합니다...")
        
        # 성공 여부 확인용 시퀀스 실행
        success = self.execute_screw_inspection_task(target_key="pos1")
        
        if success:
            self.get_logger().info("전체 검수 작업 시퀀스가 정상 종료되었습니다.")
        else:
            self.get_logger().error("검수 시퀀스 수행 중 오류가 발생했거나 불량이 감지되었습니다.")

    def execute_screw_inspection_task(self, target_key="pos1"):
        if target_key not in self.saved_positions:
            self.get_logger().error(f"지정된 번호 '{target_key}'의 나사 좌표가 캐시에 없습니다.")
            return False

        screw_pos = self.saved_positions[target_key]
        
        try:
            # [해결 1] 시작 전 홈 위치에서 그리퍼를 무조건 열어 충돌 방지
            self.get_logger().info("초기 안전 확보: 그리퍼를 개방합니다.")
            gripper.open_gripper()
            time.sleep(1.0)

            # --------------------------------------------------------
            # 1) 드라이버가 있는 위치로 이동
            # --------------------------------------------------------
            self.get_logger().info("Step 1) 드라이버 스테이션 상공으로 이동합니다.")
            tool_pos_up = list(TOOL_STATION_POS)
            tool_pos_up[2] += 100.0  
            
            movel(tool_pos_up, vel=VELOCITY, acc=ACC)
            mwait()
            
            # --------------------------------------------------------
            # 2) 드라이버 Pick
            # --------------------------------------------------------
            self.get_logger().info("Step 2) 하강하여 드라이버를 파지합니다.")
            movel(TOOL_STATION_POS, vel=VELOCITY, acc=ACC)
            mwait()
            
            gripper.close_gripper()  
            time.sleep(1.0)
            
            movel(tool_pos_up, vel=VELOCITY, acc=ACC)
            mwait()

            # --------------------------------------------------------
            # 3) 지정 번호 나사 위로 이동
            # --------------------------------------------------------
            self.get_logger().info(f"Step 3) 나사 위치({target_key}) 상공으로 이동합니다.")
            screw_pos_up = list(screw_pos)
            screw_pos_up[2] += 80.0  
            
            movel(screw_pos_up, vel=VELOCITY, acc=ACC)
            mwait()

            # --------------------------------------------------------
            # 4) 순응 제어 및 회전 진입 (보호 정지 에러 해결 로직)
            # --------------------------------------------------------
            self.get_logger().info("Step 4) 나사 표면을 향해 힘 감지 하강을 시작합니다.")
            
            # [수정] 지정 좌표로 무조건 이동하지 않고, 힘 감지 시 하강 중지
            self.get_logger().info("Step 4) 나사 표면까지 위치 제어로 먼저 하강합니다.")
            # [해결 2] 힘 제어를 켜기 '전에' 나사 접촉 위치까지 미리 이동!
            movel(screw_pos, vel=15, acc=15)
            mwait()
            time.sleep(0.5)
            
            # 비동기 명령 전달 후 실제 움직임이 시작될 때까지 짧게 대기
            time.sleep(0.5)
            
            contact_detected = False
            CONTACT_FORCE_LIMIT = 5.0 # Z축 접촉 임계값(N)
            
            # while check_motion() == 1:
            #     curr_force = get_tool_force()
            #     current_fz = abs(curr_force[2])
                
            #     if current_fz > CONTACT_FORCE_LIMIT:
            #         self.get_logger().info(f"접촉 완료! (Z축 힘: {current_fz:.2f} N). 하강을 중지합니다.")
            #         contact_detected = True
            #         # 로봇 정지: 현재 좌표로 다시 이동 명령을 내려 이전 하강 동작 취소
            #         curr_pos = get_current_posx()[0]
            #         amovel(curr_pos, vel=30, acc=30)
            #         mwait()
            #         break
                    
            #     time.sleep(0.05)
                
            # if not contact_detected:
            #     self.get_logger().warn("끝까지 하강했으나 접촉을 감지하지 못했습니다.")
            
            self.get_logger().info("순응 제어를 켜고 Z축 누르는 힘을 인가합니다.")
            task_compliance_ctrl(stx=[2000, 2000, 500, 200, 200, 200])
            # set_desired_force(fd=[0, 0, 10, 0, 0, 0], dir=[0, 0, 1, 0, 0, 0], mod=0)
            time.sleep(0.5) # 힘이 안정적으로 적용될 여유 시간
            
            # self.get_logger().info("나사 홈 진입을 위한 미세 회전을 시작합니다.")
            # for angle in [10, -20, 10]:
            #     rot_pose = trans(get_current_posx()[0], [0, 0, 0, 0, 0, angle])
            #     movel(rot_pose, vel=10, acc=10)
            #     mwait()
            #     time.sleep(0.2)
            # release_force()
            release_compliance_ctrl()

            # --------------------------------------------------------
            # 5) 토크 검수
            # --------------------------------------------------------
            self.get_logger().info("Step 5) 누르는 힘 유지 상태로 양방향 회전하며 토크를 검수합니다.")
            # 1. 누르는 힘(Force) 및 순응 제어(Compliance) 인가
            # 회전(조이기) 시 비트가 홈에서 이탈하지 않도록 Z축 방향으로 누르는 힘을 유지합니다.
            self.get_logger().info("나사를 조이는 동안 Z축 방향으로 누르는 힘(15N)을 유지합니다.")
            task_compliance_ctrl(stx=[2000, 2000, 500, 100, 100, 100])
            set_desired_force(fd=[0, 0, 15, 0, 0, 0], dir=[0, 0, 1, 0, 0, 0], mod=0)
            time.sleep(0.5) # 힘이 안정적으로 적용될 여유 시간 대기

            # 토크 모니터링 설정
            self.max_torque = 0.0
            is_normal = False
            TORQUE_LIMIT = 5.0 

            # 2. 현재 관절(Joint) 각도 가져오기 및 초기화
            curr_joint = get_current_posj()
            target_joint = list(curr_joint)
            initial_j6 = curr_joint[5]

            # 3. J6 축 회전 각도 및 횟수 설정
            turn_degree = 180.0 
            rotate_count = 3
            
            self.get_logger().info(f"J6 관절을 조이기 방향(+{turn_degree}도) 및 풀기 방향(-{turn_degree}도)으로 왕복하며 총 {rotate_count}회 체결 검수를 진행합니다.")

            for i in range(rotate_count):
                self.get_logger().info(f"[{i+1}/{rotate_count}] 조이기 방향(+{turn_degree}도) 회전 시작")
                
                target_joint[5] = initial_j6 + turn_degree

                # 관절 회전 실행 (amovej로 비동기 이동)
                amovej(target_joint, vel=30, acc=30)
                
                # 비동기 명령 전달 후 실제 움직임이 시작될 때까지 짧게 대기
                time.sleep(0.5)
                
                # 메인 스레드에서 로봇 이동 상태를 확인하며 토크 모니터링
                while check_motion() == 1:
                    curr_force = get_tool_force()
                    current_torque = abs(curr_force[5]) # Mz 토크
                    
                    if current_torque > self.max_torque:
                        self.max_torque = current_torque
                    
                    # 실시간 로깅
                    self.get_logger().info(f"[{i+1}/{rotate_count}] 회전 중 J6 실시간 토크: {current_torque:.2f} N")

                    # 과토크 감지 시 정상 체결로 판단
                    if current_torque > TORQUE_LIMIT:
                        is_normal = True
                        self.get_logger().info(f"✅ 목표 토크 도달! 정상 체결 확인: {current_torque:.2f} N")
                        break
                    
                    time.sleep(0.05) # 50ms 간격 측정
                
                # 이동 명령이 루프 도중 중단되었을 수 있으므로 확실히 대기
                mwait()

                # 정상 체결 확인 시 더 이상 시도하지 않고 루프 탈출
                if is_normal:
                    break
                    
                time.sleep(0.5) # 원위치 복귀 전 짧은 대기
                
                # 원위치(-90도)로 복귀
                self.get_logger().info(f"[{i+1}/{rotate_count}] 풀기 방향(원위치) 복귀 시작")
                target_joint[5] = initial_j6
                amovej(target_joint, vel=30, acc=30)
                time.sleep(0.5)
                while check_motion() == 1:
                    time.sleep(0.05)
                mwait()
                time.sleep(0.5) # 다음 시도 전 대기

            self.is_defective = not is_normal

            self.get_logger().info("나사 검수 동작 완료. 초기 각도로 원상복구를 시작합니다.")
            
            # 6. 원래 각도(초기 각도)로 복귀 (정상 체결로 중간에 멈췄을 수 있으므로)
            target_joint = list(get_current_posj())
            target_joint[5] = initial_j6
            
            amovej(target_joint, vel=30, acc=30)
            time.sleep(0.5)
            
            while check_motion() == 1:
                time.sleep(0.05)
                
            mwait()

            time.sleep(0.5)
            if self.is_defective:
                self.get_logger().error(f"❌ 불량 판정! {rotate_count}회 시도에도 목표 토크에 도달하지 못했습니다. (최대 토크: {self.max_torque:.2f} N)")
            else:
                self.get_logger().info(f"✅ 정상 체결 완료 (최대 토크: {self.max_torque:.2f} N)")
                
            # 누르는 힘과 순응 제어를 해제하여 로봇을 원래 강성 제어 상태로 복구
            release_force()
            release_compliance_ctrl()
            time.sleep(0.5)

            # --------------------------------------------------------
            # 6) 복귀 시퀀스
            # --------------------------------------------------------
            self.get_logger().info("작업 완료. 드라이버를 원위치시킵니다.")
            movel(screw_pos_up, vel=VELOCITY, acc=ACC)
            mwait()
            
            movel(tool_pos_up, vel=VELOCITY, acc=ACC)
            mwait()
            movel(TOOL_STATION_POS, vel=VELOCITY, acc=ACC)
            mwait()
            
            gripper.open_gripper()
            time.sleep(1.0)
            
            movel(tool_pos_up, vel=VELOCITY, acc=ACC)
            mwait()
            self.init_robot()
            
            return is_normal

        except Exception as e:
            # [개선] 에러 발생 시 터미널에 상세 위치 추적(Traceback)을 출력하여 정확한 원인을 파악하게 함
            self.get_logger().error(f"예외 상황 발생으로 시퀀스 중단: {e}")
            self.get_logger().error(traceback.format_exc())
            release_compliance_ctrl() # 안전을 위해 힘 제어 무조건 강제 해제
            return False


def main(args=None):
    node = RobotScrewInspector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("사용자에 의해 시스템이 종료되었습니다.")
    finally:
        try:
            gripper.close_connection()
        except:
            pass
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()