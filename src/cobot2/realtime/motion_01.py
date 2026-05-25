import rclpy
from rclpy.node import Node
import time
import DR_init

from dsr_msgs2.srv import MoveJoint
from std_srvs.srv import Trigger

ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
ROBOT_TOOL = "Tool Weight"
ROBOT_TCP = "GripperDA_v1"

TARGET_JOINTS = [
    [0.0, -0.0, 90.039, -90.0, 90.0, 0.0],
    [-0.006, -0.019, 89.985, -90.003, -0.003, -0.0],
    [-0.034, -0.019, 89.984, 90.003, 90.005, 0.0],
    [-0.034, -0.018, 89.984, -0.004, 90.005, 0.0],
    [-0.034, -0.022, 89.983, -0.004, -90.004, -0.0],
]
HOME_J = [0.0, 0.0, 90.0, 0.0, 90.0, 0.0]

class RobotMoverClient(Node):
    def __init__(self):
        super().__init__('robot_mover_client_node', namespace=ROBOT_ID)

        # 로봇 제어 서비스
        self.movej_client = self.create_client(MoveJoint, f'/{ROBOT_ID}/motion/move_joint')
        # 🌟 비전 검사 요청 서비스
        self.vision_client = self.create_client(Trigger, '/vision_inspect')

        self.get_logger().info('로봇 제어 및 비전 서버 연결 대기 중...')
        self.movej_client.wait_for_service()
        self.vision_client.wait_for_service()
        self.get_logger().info('✅ 시스템 준비 완료! 자동 탐색 궤적을 시작합니다.')

    def move_to(self, pos, label):
        req = MoveJoint.Request()
        req.pos, req.vel, req.acc = pos, 30.0, 30.0
        req.time, req.radius, req.mode, req.blend_type, req.sync_type = 0.0, 0.0, 0, 0, 0
        
        self.get_logger().info(f"▶️ [이동] {label} (으)로 이동합니다.")
        future = self.movej_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        time.sleep(0.5)  # 카메라 잔진동 대기 (0.5초)

    def request_vision_inspection(self):
        self.get_logger().info("🔍 비전 서버에 검사 및 업로드를 지시합니다... (완료될 때까지 대기)")
        req = Trigger.Request()
        
        # 💡 조건 1: call_async 후 spin_until_future_complete로 비전 처리가 끝날 때까지 여기서 완전히 블로킹됨
        future = self.vision_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        
        result = future.result()
        if result.success and result.message == "SKIPPED":
            self.get_logger().info("⏭️ (결과) 나사 없음 -> 맵핑 생략, 바로 출발합니다.")
        else:
            self.get_logger().info("✅ (결과) 나사 검사 및 DB 업로드 완료!")
        print("-" * 50)

    def execute_sequence(self):
        self.move_to(HOME_J, "홈 위치")

        for i, pos in enumerate(TARGET_JOINTS):
            # 1. 이동
            self.move_to(pos, f"탐색 시점 {i+1}/5")
            # 2. 검사 완료 후 응답이 올 때까지 대기 (없으면 바로 리턴됨)
            self.request_vision_inspection()

        self.move_to(HOME_J, "복귀 홈")
        self.get_logger().info("🎉 모든 구역 탐색이 완벽하게 종료되었습니다!")

def main(args=None):
    DR_init.__dsr__id = ROBOT_ID
    DR_init.__dsr__model = ROBOT_MODEL

    rclpy.init(args=args)
    mover = RobotMoverClient()
    DR_init.__dsr__node = mover

    try:
        from DSR_ROBOT2 import (
            set_robot_mode, ROBOT_MODE_AUTONOMOUS, ROBOT_MODE_MANUAL,
            set_tool, set_tcp
        )

        print("\n로봇 초기화 및 권한 설정 중...")
        set_robot_mode(ROBOT_MODE_MANUAL)
        set_tool(ROBOT_TOOL)
        set_tcp(ROBOT_TCP)
        set_robot_mode(ROBOT_MODE_AUTONOMOUS)
        time.sleep(1.0) 

        mover.execute_sequence()
    except KeyboardInterrupt: pass
    except Exception as e:
        print(f"\n🚨 에러 발생: {e}")
    finally:
        mover.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()