import rclpy
import DR_init
import time

# 로봇 설정 상수
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
ROBOT_TOOL = "Tool Weight"
ROBOT_TCP = "GripperDA"

# 이동 속도 및 가속도
VELOCITY = 60
ACC = 60

# DR_init 설정
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL


def initialize_robot():
    """로봇의 Tool과 TCP를 설정"""
    from DSR_ROBOT2 import set_tool, set_tcp  # 필요한 기능만 임포트

    # 설정된 상수 출력
    print("#" * 50)
    print("Initializing robot with the following settings:")
    print(f"ROBOT_ID: {ROBOT_ID}")
    print(f"ROBOT_MODEL: {ROBOT_MODEL}")
    print(f"ROBOT_TCP: {ROBOT_TCP}")
    print(f"ROBOT_TOOL: {ROBOT_TOOL}")
    print(f"VELOCITY: {VELOCITY}")
    print(f"ACC: {ACC}")
    print("#" * 50)

    # Tool과 TCP 설정
    set_tool(ROBOT_TOOL)
    set_tcp(ROBOT_TCP)


def perform_task():
    """간단한 조인트 이동(movej)을 수행합니다."""
    # DSR_ROBOT2에서 모션 관련 함수 임포트
    from DSR_ROBOT2 import movej
    
    print("🤖 로봇 이동 작업을 시작합니다...")

    # 1. 초기 안전 자세(Home) 정의 (J1~J6 관절 각도 리스트)
    pos_home = [0.0, 0.0, 90.0, 0.0, 90.0, 0.0]
    
    # 2. 이동할 타겟 자세 정의 (1축과 2축을 조금씩 회전)
    pos_target = [30.0, -20.0, 90.0, 0.0, 90.0, 0.0]

    # Home 위치로 이동
    print(f"  -> Home 위치 {pos_home} (으)로 이동 중...")
    movej(pos_home, v=VELOCITY, a=ACC)
    time.sleep(1.0)

    # 타겟 위치로 이동
    print(f"  -> Target 위치 {pos_target} (으)로 이동 중...")
    movej(pos_target, v=VELOCITY, a=ACC)
    time.sleep(1.0)

    # 다시 Home으로 복귀
    print("  -> 원래 Home 위치로 복귀 중...")
    movej(pos_home, v=VELOCITY, a=ACC)
    
    print("✅ 모든 이동 작업이 성공적으로 완료되었습니다!")


def main(args=None):
    """메인 함수: ROS 2 노드 초기화 및 동작 수행"""
    rclpy.init(args=args)
    node = rclpy.create_node("move_periodic", namespace=ROBOT_ID)

    # DR_init에 노드 설정
    DR_init.__dsr__node = node

    try:
        # 초기화는 한 번만 수행
        initialize_robot()

        # 작업 수행 (한 번만 호출)
        perform_task()

    except KeyboardInterrupt:
        print("\nNode interrupted by user. Shutting down...")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()