import os
import time
import sys
import traceback
from scipy.spatial.transform import Rotation
import numpy as np
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

ROBOT_TOOL = "Tool Weight"
ROBOT_TCP = "GripperDA_v1"

# [하드웨어 고정 좌표 설정]
TOOL_STATION_POS = [415.578, -245.516, 150.815, 37.934, -179.505, -51.596]   # 드라이버 거치 위치
TARGET_SCREW_POS = [545.1,  -14.8, 64.1, 0.0, 180.0, 0.0]                  # 테스트용 기본 나사 위치

# 토크 검수 임계값 (Nm)
TORQUE_THRESHOLD = 5.0

# ROS 2 및 두산 API 바인딩 전역 선언
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

rclpy.init()
dsr_node = rclpy.create_node("robot_screw_inspection_node", namespace=ROBOT_ID)
DR_init.__dsr__node = dsr_node

stop_requested = False


# ==========================================================
# STOP 처리
# ==========================================================

class StopRequestedException(Exception):
    pass


def signal_handler(signum, frame):
    global stop_requested

    print(f"[TASK_ONCE] Signal received: {signum}", flush=True)
    stop_requested = True


def check_stop():
    if stop_requested:
        print("[TASK_ONCE] STOP detected. Abort seasoning task.", flush=True)
        raise StopRequestedException("STOPPED_BY_USER")



try:
    from DSR_ROBOT2 import (
        movej, movel, get_current_posx, mwait, trans, posx,
        task_compliance_ctrl, set_desired_force, release_compliance_ctrl, get_tool_force,set_tcp,set_tool
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

        set_tool(ROBOT_TOOL)
        set_tcp(ROBOT_TCP)


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

        screw_pos = list(self.saved_positions[target_key])
        screw_pos[1] -= 80
        screw_pos[2] += 120
        
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

            # ✅ [핵심 해결책] 제어기 상태 동기화 및 큐 안정화
            time.sleep(0.5)  # 모션이 완전히 끝나고 제어기가 Ready 상태가 될 때까지 짧은 물리적 대기
            _ = get_current_posx() # 로봇의 현재 위치를 한 번 강제로 읽어와 통신 큐를 갱신하고 상태를 동기화시킴        
            # --------------------------------------------------------
            # 3) 지정 번호 나사 위로 이동
            # --------------------------------------------------------
            self.get_logger().info(f"Step 3) 나사 위치({target_key}) 상공으로 이동합니다.")
            screw_pos_up = list(screw_pos)
            screw_pos_up[1] -= 80.0  
            screw_pos_up[2] += 300.0  
            # movej(JHOME_POS, vel=VELOCITY, acc=ACC)
            # mwait()
            
            movel(screw_pos_up, vel=VELOCITY, acc=ACC)
            mwait()
            check_stop()
              
       # --------------------------------------------------------
            # 4) 순응제어(Compliance)로 조금씩 회전하면서 나사에 박아넣음 (조립 진입)
            # --------------------------------------------------------
            self.get_logger().info("Step 4) 순응제어를 활성화하고 미세 회전하며 나사 홈에 진입(박아넣음)합니다.")
            
         # 나사 표면까지 위치 제어로 하강
            movel(screw_pos, vel=15, acc=15)
            mwait()
            time.sleep(0.5)
            
            # 컴플라이언스 제어 켜기 및 누르는 힘(-15N) 인가
            task_compliance_ctrl(stx=[3000, 3000, 500, 200, 200, 200])
            set_desired_force(fd=[0, 0, -10, 0, 0, 0], dir=[0, 0, 1, 0, 0, 0], mod=0)
            time.sleep(0.5)  # 힘 안정화 대기

            # --- [추가/수정] 홈 진입 탐색(Wiggling) 및 토크 확인 로직 ---
            MESHING_TORQUE = 1.5  # 홈에 들어갔을 때 발생하는 회전 저항 임계값 (Nm) - 환경에 맞게 튜닝 필요
            max_wiggle_attempts = 5
            is_meshed = False
            direction = 1

            for attempt in range(max_wiggle_attempts):
                # 왕복 운동 각도 계산 (처음엔 5도 이동, 그 다음부터는 반대 방향으로 10도씩 이동해야 중앙 기준 ±5도 왕복이 됨)
                move_angle = 5 if attempt == 0 else 10
                angle = move_angle * direction
                
                # J6축(Rx, Ry, Rz 중 Rz) 회전 모션 명령
                rot_pose = trans(get_current_posx()[0], [0, 0, 0, 0, 0, angle])
                movel(rot_pose, vel=10, acc=10)
                mwait()
                time.sleep(0.1) # 회전 직후 관성 안정화를 위한 짧은 대기
                
                # 현재 툴 토크 측정 (tool_force[5]가 Z축 회전 방향의 토크 Mz)
                tool_force = get_tool_force()
                current_mz_torque = abs(tool_force[5])
                
                self.get_logger().info(f"탐색 시도 {attempt+1}/{max_wiggle_attempts} | 방향: {angle}도 | 측정 토크: {current_mz_torque:.2f} Nm")
                
                # 3번 요구사항: 토크 변화로 진입 확인
                if current_mz_torque >= MESHING_TORQUE:
                    self.get_logger().info(f"🎯 나사 홈 진입 감지 완료! (토크 스파이크: {current_mz_torque:.2f} Nm)")
                    is_meshed = True
                    break  # 진입에 성공했으므로 2번 행동(왕복 운동) 즉시 정지
                
                # 다음 번 루프를 위해 회전 방향 반전
                direction *= -1

            if not is_meshed:
                self.get_logger().warn("⚠️ 최대 탐색 횟수를 초과했습니다. 나사 홈을 찾지 못해 헛돌 가능성이 있습니다.")
                # 필요하다면 여기서 return False를 하여 작업을 중단시킬 수도 있습니다.

            # 1번 요구사항: 나사 구멍에 잘 들어가면 순응 제어를 멈추고 그 높이에서 정지
            self.get_logger().info("순응 제어를 중지하고 현재 진입한 깊이(높이) 상태를 단단히 고정합니다.")
            release_compliance_ctrl()
            time.sleep(0.5)

            # --------------------------------------------------------
            # 5) 토크 검수
            # --------------------------------------------------------
            self.get_logger().info("Step 5) 누르는 힘 유지 상태로 양방향 회전하며 토크를 검수합니다.")
            max_detected_torque = 0.0
            
            # +5도
            pos_plus5 = trans(get_current_posx()[0], [0, 0, 0, 0, 0, 5])
            movel(pos_plus5, vel=8, acc=8)
            mwait()
            time.sleep(0.3)
            tool_force = get_tool_force()
            max_detected_torque = max(max_detected_torque, abs(tool_force[5]))
            
            # -5도 (현재 기준 -10도 이동)
            pos_minus5 = trans(get_current_posx()[0], [0, 0, 0, 0, 0, -10])
            movel(pos_minus5, vel=8, acc=8)
            mwait()
            time.sleep(0.3)
            tool_force = get_tool_force()
            max_detected_torque = max(max_detected_torque, abs(tool_force[5]))
            
            # 센터 복귀
            pos_neutral = trans(get_current_posx()[0], [0, 0, 0, 0, 0, 5])
            movel(pos_neutral, vel=10, acc=10)
            mwait()
            
            # [필수] 상승하기 전 힘 제어 해제
            release_compliance_ctrl()
            
            self.get_logger().info(f"검출된 최대 저항 토크: {max_detected_torque:.2f} Nm")
            
            if max_detected_torque >= TORQUE_THRESHOLD:
                self.get_logger().info("★ 판정: [정상] 나사가 꽉 조여져 있습니다.")
                is_normal = True
            else:
                self.get_logger().warn("⚠️ 판정: [불량] 나사가 느슨합니다.")
                is_normal = False

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