import rclpy
from rclpy.node import Node
import time
import sys
import DR_init

try:
    from onrobot import RG
except ImportError:
    sys.exit("onrobot 모듈을 찾을 수 없습니다. 경로를 확인해주세요.")

# ==========================================
# 두산 API 임포트 전, 로봇 노드 선행 생성
# ==========================================
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"

rclpy.init()
dsr_node = rclpy.create_node("screw_task_dsr_node", namespace=ROBOT_ID)

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL
DR_init.__dsr__node = dsr_node

try:
    from DSR_ROBOT2 import (
        movej,
        movel,
        mwait,
        wait,
        posx,
        task_compliance_ctrl,
        set_desired_force,
        release_compliance_ctrl,
        get_tool_force,
        get_current_posx,
    )
except Exception as e:
    sys.exit(f"두산 로봇 API 임포트 실패 (상세 원인): {e}")


# ==========================================
# 로봇 및 그리퍼 제어 설정값
# ==========================================
VELOCITY, ACC = 60, 60

JHOME_POS = [0, 0, 90, 0, 90, 0]

TOOL_STATION_POS = [
    415.578,
    -245.516,
    250.815,
    37.934,
    -179.505,
    -51.596,
]

TARGET_SCREW_POS = [
    540.77,
    0.29,
    466.538,
    0.0,
    180.0,
    0.0,
]

TORQUE_THRESHOLD = 5.0

GRIPPER_NAME = "rg2"
TOOLCHARGER_IP = "192.168.1.1"
TOOLCHARGER_PORT = 502


class ScrewTighteningTask(Node):
    def __init__(self):
        super().__init__("screw_tightening_task_node")
        self.get_logger().info("단일 좌표 나사 체결 및 검수 작업을 시작합니다.")

        try:
            self.gripper = RG(
                GRIPPER_NAME,
                TOOLCHARGER_IP,
                TOOLCHARGER_PORT,
            )
            self.get_logger().info("✅ OnRobot 그리퍼 연결 성공!")
        except Exception as e:
            self.get_logger().error(f"❌ 그리퍼 연결 실패: {e}")
            sys.exit(1)

    def pick_and_place_tool(self, action="pick"):
        movej(JHOME_POS, vel= 50, acc= 50)
        self.get_logger().info(f"드라이버 {action} 시작...")

        approach_pos = list(TOOL_STATION_POS)
        approach_pos[2] += 200.0

        self.get_logger().info("이동: 스테이션 상단")
        movel(posx(approach_pos), vel=VELOCITY, acc=ACC)
        wait(1)

        self.get_logger().info("이동: 스테이션 위치")
        movel(TOOL_STATION_POS, vel=VELOCITY, acc=ACC)
        mwait()

        self.get_logger().info(f"그리퍼 {action} 실행")

        if action == "pick":
            self.gripper.close_gripper(force_val=400)
        else:
            self.gripper.open_gripper()

        # time.sleep(2.0)

        # self.get_logger().info("상승 복귀")
        # movel(approach_pos, vel=VELOCITY, acc=ACC)
        # mwait()

    def execute_task(self):
        self.gripper.open_gripper()

        target_pos = list(TARGET_SCREW_POS)
        self.get_logger().info(f"지정된 타겟 좌표 {target_pos[:3]} 로 작업을 시작합니다.")

        # 1. 드라이버 Pick
        self.pick_and_place_tool(action="pick")

        # 2. 타겟 나사 위치 상단으로 이동
        # target_pos_up = list(target_pos)
        # target_pos_up[2] += 50.0

        # self.get_logger().info(f"이동 명령 전송: {target_pos_up}")
        # movel(target_pos_up, vel=VELOCITY, acc=ACC)
        # mwait()
        # time.sleep(0.5)

        # # 3. 타겟 나사 표면으로 접근
        # self.get_logger().info(f"이동 명령 전송: {target_pos}")
        # movel(target_pos, vel=VELOCITY, acc=ACC)
        # mwait()
        # time.sleep(0.5)

        # # 4. 순응 제어 및 힘 제어 활성화
        # self.get_logger().info("순응 제어 활성화: 나사 압박 시작")

        # stx = [3000, 3000, 500, 200, 200, 200]
        # task_compliance_ctrl(stx)

        # fd = [0, 0, -10, 0, 0, 0]
        # f_dir = [0, 0, 1, 0, 0, 0]
        # set_desired_force(fd, f_dir)

        # time.sleep(1.0)

        # # 5. 회전 및 토크 검수
        # max_torque_measured = 0.0

        # self.get_logger().info("나사 체결 회전 시작: 60도씩 분할 회전")

        # ROTATION_STEP = 60.0
        # ROTATION_COUNT = 6
        # ROTATION_VEL = 10
        # ROTATION_ACC = 10
        # TORQUE_CHECK_DELAY = 0.2

        # for i in range(ROTATION_COUNT):
        #     current_pos = get_current_posx()[0]
        #     next_pos = list(current_pos)

        #     next_pos[5] += ROTATION_STEP

        #     self.get_logger().info(
        #         f"회전 {i + 1}/{ROTATION_COUNT}: "
        #         f"현재 Rz={current_pos[5]:.2f}, 목표 Rz={next_pos[5]:.2f}"
        #     )

        #     movel(next_pos, vel=ROTATION_VEL, acc=ROTATION_ACC)
        #     mwait()

        #     time.sleep(TORQUE_CHECK_DELAY)

        #     forces = get_tool_force()
        #     mz_torque = abs(forces[5])

        #     if mz_torque > max_torque_measured:
        #         max_torque_measured = mz_torque

        #     self.get_logger().info(
        #         f"회전 {i + 1}/{ROTATION_COUNT} 완료, "
        #         f"현재 토크: {mz_torque:.2f} Nm, "
        #         f"최대 토크: {max_torque_measured:.2f} Nm"
        #     )

        #     if max_torque_measured >= TORQUE_THRESHOLD:
        #         self.get_logger().info("목표 토크 도달. 추가 회전 중단.")
        #         break

        # # 6. 순응 제어 해제
        # release_compliance_ctrl()
        # self.get_logger().info("순응 제어 해제")

        # time.sleep(0.5)

        # # 7. 불량 판독
        # if max_torque_measured >= TORQUE_THRESHOLD:
        #     self.get_logger().info(
        #         f"✅ 체결 성공! 측정 토크: {max_torque_measured:.2f} Nm"
        #     )
        #     is_pass = True
        # else:
        #     self.get_logger().warn(
        #         f"❌ 체결 불량 의심! 헛돌음 감지. "
        #         f"측정 토크: {max_torque_measured:.2f} Nm"
        #     )
        #     is_pass = False

        # # 8. 안전 높이로 후퇴
        # self.get_logger().info("안전 높이로 후퇴")
        # movel(target_pos_up, vel=VELOCITY, acc=ACC)
        # mwait()

        # # 9. 툴 반납
        # self.pick_and_place_tool(action="place")

        # # 10. 홈 복귀
        # self.get_logger().info("홈 위치로 복귀")
        # movej(JHOME_POS, vel=VELOCITY, acc=ACC)
        # mwait()

        # return is_pass


def main(args=None):
    node = ScrewTighteningTask()

    try:
        result = node.execute_task()

        if result:
            node.get_logger().info("작업이 성공적으로 종료되었습니다.")
        else:
            node.get_logger().warn("작업이 완료되었으나, 불량 나사가 감지되었습니다.")

    except KeyboardInterrupt:
        node.get_logger().info("사용자에 의해 중단되었습니다.")

    except Exception as e:
        node.get_logger().error(f"실행 중 에러 발생: {e}")

    finally:
        if hasattr(node, "gripper"):
            node.gripper.close_connection()

        node.destroy_node()
        dsr_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()