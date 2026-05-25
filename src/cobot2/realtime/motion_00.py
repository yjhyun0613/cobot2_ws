#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
import time

# 두산 로봇 제어를 위한 ROS 2 서비스 임포트
from dsr_msgs2.srv import MoveJoint

# === ⚙️ 설정값 ===
ROBOT_ID = "dsr01"

# 목표 탐색 조인트 위치 5곳
TARGET_JOINTS = [
    [0.0, -0.0, 90.039, -90.0, 90.0, 0.0],
    [-0.006, -0.019, 89.985, -90.003, -0.003, -0.0],
    [-0.034, -0.019, 89.984, 90.003, 90.005, 0.0],
    [-0.034, -0.018, 89.984, -0.004, 90.005, 0.0],
    [-0.034, -0.022, 89.983, -0.004, -90.004, -0.0],
]

# 초기/복귀용 홈 위치
HOME_J = [0.0, 0.0, 90.0, 0.0, 90.0, 0.0]

class RobotArmMover(Node):
    def __init__(self):
        super().__init__('robot_arm_mover_node')

        # 1. 이동 명령을 내리기 위한 Service Client 생성
        self.movej_client = self.create_client(MoveJoint, f'/{ROBOT_ID}/motion/move_joint')

        # 2. 로봇 서비스가 연결될 때까지 대기
        self.get_logger().info('로봇 제어 서비스 연결을 기다리는 중...')
        while not self.movej_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('기다리는 중: move_joint 서비스...')
        self.get_logger().info('✅ 로봇 연결 완료!')

    def movej_sync(self, pos, label="Unknown"):
        """ 로봇을 특정 조인트 각도로 동기식(도착할 때까지 대기)으로 이동시키는 함수 """
        req = MoveJoint.Request()
        req.pos = pos
        req.vel = 30.0   # 이동 속도
        req.acc = 30.0   # 가속도
        req.time = 0.0
        req.radius = 0.0
        req.mode = 0
        req.blend_type = 0
        req.sync_type = 0

        self.get_logger().info(f"▶️ [이동] {label} 로 이동 중 -> {pos}")
        
        # 비동기 호출 후 완료될 때까지 블로킹(스핀)
        future = self.movej_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        # 이동 완료 후 잔진동 안정을 위해 잠시 대기
        time.sleep(1.0)
        self.get_logger().info(f"✅ {label} 도착 완료!")

    def execute_sequence(self):
        """ 정의된 시퀀스대로 로봇을 이동시키는 메인 로직 """
        # 1. 홈 위치로 초기화 이동
        self.movej_sync(HOME_J, "홈(Home) 위치")

        # 2. 5개의 지정된 위치로 순차 이동
        for i, pos in enumerate(TARGET_JOINTS):
            self.movej_sync(pos, f"탐색 시점 {i+1}")

        # 3. 모든 탐색 완료 후 홈으로 복귀
        self.movej_sync(HOME_J, "복귀 홈(Home)")
        self.get_logger().info("🎉 모든 이동 궤적을 성공적으로 마쳤습니다!")

def main(args=None):
    rclpy.init(args=args)
    mover = RobotArmMover()

    try:
        # 궤적 실행 시작
        mover.execute_sequence()
    except KeyboardInterrupt:
        mover.get_logger().info("사용자에 의해 강제 종료되었습니다.")
    finally:
        mover.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()