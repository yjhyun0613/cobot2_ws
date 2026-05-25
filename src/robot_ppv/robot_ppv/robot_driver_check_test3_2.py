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

# [하드웨어 고정 좌표 설정]
TOOL_STATION_POS = [415.578, -245.516, 150.815, 37.934, -179.505, -51.596]   # 드라이버 거치 위치
TARGET_SCREW_POS = [545.1,  -14.8, 64.1, 0.0, 180.0, 0.0]                      # 테스트용 기본 나사 위치
TARGET_SCREW_POS[2] += 120
# 토크 검수 임계값 (Nm)
TORQUE_THRESHOLD = 5.0

# ROS 2 및 두산 API 바인딩 전역 선언
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

rclpy.init()
dsr_node = rclpy.create_node("robot_screw_inspection_node", namespace=ROBOT_ID)
DR_init.__dsr__node = dsr_node

try:
    from DSR_ROBOT2 import (
        movej, movel, get_current_posx, mwait, trans, posx,
        task_compliance_ctrl, set_desired_force, release_compliance_ctrl, get_tool_force
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
            self.get_logger().info("Step 4) 나사 표면까지 위치 제어로 먼저 하강합니다.")
            # [해결 2] 힘 제어를 켜기 '전에' 나사 접촉 위치까지 미리 이동!
            movel(screw_pos, vel=15, acc=15)
            mwait()
            time.sleep(0.5)
            
            self.get_logger().info("접촉 완료. 순응 제어를 켜고 Z축 누르는 힘을 인가합니다.")
            task_compliance_ctrl(stx=[3000, 3000, 500, 200, 200, 200])
            set_desired_force(fd=[0, 0, -15, 0, 0, 0], dir=[0, 0, 1, 0, 0, 0], mod=0)
            time.sleep(0.5) # 힘이 안정적으로 적용될 여유 시간
            
            self.get_logger().info("나사 홈 진입을 위한 미세 회전을 시작합니다.")
            for angle in [10, -20, 10]:
                rot_pose = trans(get_current_posx()[0], [0, 0, 0, 0, 0, angle])
                movel(rot_pose, vel=10, acc=10)
                mwait()
                time.sleep(0.2)

            # --------------------------------------------------------
            # 5) 토크 검수
            # --------------------------------------------------------
            self.get_logger().info("Step 5) 누르는 힘 유지 상태로 양방향 회전하며 토크를 검수합니다.")
            max_detected_torque = 0.0
            
            # +5도
            pos_plus5 = trans(get_current_posx()[0], [0, 0, 0, 0, 0, 25])
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