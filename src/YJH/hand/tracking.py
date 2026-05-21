import os
import time
import sys
import threading
import math
import numpy as np
import cv2
import mediapipe as mp

import rclpy
from rclpy.node import Node
import DR_init

# 패키지 정보 및 하드웨어 설정 (기존 코드 참고)
from pick_and_place_text.onrobot import RG
from dsr_msgs2.msg import ServolStream  # 실시간 스트리밍 제어 메시지

ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"

# 로봇 초기화 바인딩
rclpy.init()
dsr_node = rclpy.create_node("rokey_shadow_sync", namespace=ROBOT_ID)
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL
DR_init.__dsr__node = dsr_node

try:
    # 섀도우 동기화 중에는 servol을 쓰지만, 초기 위치 이동 등을 위해 movej 포함
    from DSR_ROBOT2 import movej, get_current_posx, mwait, servol
except ImportError as e:
    print(f"Error importing DSR_ROBOT2: {e}")
    sys.exit()

# 그리퍼 설정
GRIPPER_NAME = "rg2"
TOOLCHANGER_IP = "192.168.1.1"
TOOLCHANGER_PORT = "502"
gripper = RG(GRIPPER_NAME, TOOLCHANGER_IP, TOOLCHANGER_PORT)

# --- 섀도우 동기화 설정값 ---
ROBOT_BASE_X = 367.32  # 로봇 초기 X 위치 (앞뒤)
ROBOT_BASE_Y = 3.69    # 로봇 초기 Y 위치 (좌우 고정, XZ평면 제어이므로 고정)
ROBOT_BASE_Z = 422.92  # 로봇 초기 Z 위치 (높이)

SCALE_X = 600.0  # 손의 좌우(X) 이동폭을 로봇 이동(mm)으로 변환할 스케일
SCALE_Z = 400.0  # 손의 상하(Y) 이동폭을 로봇 높이(Z, mm)로 변환할 스케일

# 전역 변수
calibration_triggered = False


# 거리 계산 함수 (손가락 굽힘 판별용)
def distance(p1, p2):
    return math.dist((p1.x, p1.y), (p2.x, p2.y))

# 손가락 상태 판별 함수
def get_finger_state(hand_landmarks):
    points = hand_landmarks.landmark
    open_count = 0
    
    # 엄지 확인
    if distance(points[4], points[9]) > distance(points[3], points[9]): 
        open_count += 1
    # 나머지 손가락 확인
    for i in range(8, 21, 4):
        if distance(points[i], points[0]) > distance(points[i - 1], points[0]):
            open_count += 1
            
    if open_count >= 4: 
        return "OPEN"
    elif open_count <= 1: 
        return "CLOSE"
    return "HOLD"


class ShadowSyncController(Node):
    def __init__(self):
        super().__init__('shadow_sync_controller')
        self.get_logger().info("Shadow Sync Node Initialized. 시스템 준비 중...")
        
        # ROS 퍼블리셔 (ServolStream)
        # 로봇 네임스페이스와 토픽 이름은 환경에 맞게 조정 필요
        topic_name = f'/{ROBOT_ID}/servol_stream'
        self.servol_pub = self.create_publisher(ServolStream, topic_name, 10)

        # 스무딩 필터용 변수
        self.filtered_x = ROBOT_BASE_X
        self.filtered_z = ROBOT_BASE_Z
        self.alpha = 0.15  # 필터 강도 (0~1). 낮을수록 부드러움.
        
        # 그리퍼 제어 스로틀링 (너무 잦은 API 호출 방지)
        self.current_gripper_state = "OPEN"
        self.last_gripper_cmd_time = time.time()
        
        # 로봇 초기 자세 세팅
        self.init_robot()

    def init_robot(self):
        self.get_logger().info("로봇 초기 자세(JReady)로 이동합니다.")
        JReady = [0, 0, 90, 0, 90, 0]
        movej(JReady, vel=60, acc=60)
        gripper.open_gripper()
        mwait()
        self.get_logger().info("로봇 초기화 완료.")

    def update_robot_target(self, target_x, target_z, finger_state):
        # 1. 위치 스무딩 (EMA 필터)
        self.filtered_x = (1 - self.alpha) * self.filtered_x + (self.alpha * target_x)
        self.filtered_z = (1 - self.alpha) * self.filtered_z + (self.alpha * target_z)

        # 2. 안전 영역 (Workspace) 제한 
        safe_x = np.clip(self.filtered_x, ROBOT_BASE_X - 300, ROBOT_BASE_X + 300)
        safe_z = np.clip(self.filtered_z, ROBOT_BASE_Z - 200, ROBOT_BASE_Z + 300)

        # 🌟 3. 현재 로봇 자세 읽어오기 (Singularity Stop 방지 핵심!)
        current_pos = get_current_posx()[0] 
        current_rx = current_pos[3]
        # current_ry = current_pos[4] # B축은 고정할 것이므로 읽기만 함
        current_rz = current_pos[5]

        # 3. ServolStream 메시지 구성 및 퍼블리시
        msg = ServolStream()
        # 로봇의 rx, ry, rz 자세는 초기 자세(그리퍼가 바닥을 보는 형태)를 유지
        msg.pos = [
            float(safe_x), 
            float(ROBOT_BASE_Y), 
            float(safe_z), 
            float(current_rx),   # A, C축 요동 방지 (현재 상태 유지)
            179.9,               # B축은 수직 하방 유지 (특이점 회피를 위해 179.9 권장)
            float(current_rz)    # A, C축 요동 방지 (현재 상태 유지)
        ]
        msg.vel = [250.0, 50.0] # 선속도, 각속도 제한
        msg.acc = [500.0, 100.0] # 선가속도, 각가속도 제한
        msg.time = 0.05 # 50ms마다 갱신 (매우 짧게 두어 실시간 추종)
        self.servol_pub.publish(msg)

        # 4. 그리퍼 상태 업데이트 (딜레이 적용하여 로봇 과부하 방지)
        current_time = time.time()
        if (current_time - self.last_gripper_cmd_time) > 0.5: # 0.5초 쿨타임
            if finger_state == "OPEN" and self.current_gripper_state != "OPEN":
                # self.get_logger().info("그리퍼 명령: OPEN")
                # thread 안에서 그리퍼 제어 시 충돌을 막기 위해 DSR API 대신 통신 활용 고려 
                gripper.open_gripper() 
                self.current_gripper_state = "OPEN"
                self.last_gripper_cmd_time = current_time
            elif finger_state == "CLOSE" and self.current_gripper_state != "CLOSE":
                # self.get_logger().info("그리퍼 명령: CLOSE")
                gripper.close_gripper()
                self.current_gripper_state = "CLOSE"
                self.last_gripper_cmd_time = current_time


def wait_for_enter():
    global calibration_triggered
    input("\n[동기화 준비] 로봇이 대기 중입니다. 가슴 중앙에 손을 얹고 [Enter] 키를 누르세요!\n")
    calibration_triggered = True


def main(args=None):
    controller = ShadowSyncController()

    # 터미널 입력 대기 스레드
    threading.Thread(target=wait_for_enter, daemon=True).start()

    # MediaPipe 초기화
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False, max_num_hands=1,
        min_detection_confidence=0.7, min_tracking_confidence=0.7
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    is_calibrated = False
    base_hand_x, base_hand_y = 0.0, 0.0

    try:
        while rclpy.ok():
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1) # 거울 모드
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            global calibration_triggered

            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # 손가락 상태 (그리퍼) 판별
                finger_state = get_finger_state(hand_landmarks)

                # 손목(0번)이나 중지 밑단(9번)을 기준으로 위치 매핑
                current_hx = hand_landmarks.landmark[9].x
                current_hy = hand_landmarks.landmark[9].y

                if calibration_triggered and not is_calibrated:
                    base_hand_x = current_hx
                    base_hand_y = current_hy
                    is_calibrated = True
                    print(f"영점 설정 완료! 기준 픽셀비: ({base_hand_x:.3f}, {base_hand_y:.3f})")

                if is_calibrated:
                    delta_x = current_hx - base_hand_x
                    delta_y = current_hy - base_hand_y

                    # X축 방향은 그대로, Y축(화면 상하)은 Z축(로봇 높이)으로 역방향 매핑
                    target_x = ROBOT_BASE_X + (delta_x * SCALE_X)
                    target_z = ROBOT_BASE_Z - (delta_y * SCALE_Z) 

                    # 로봇 퍼블리시 및 그리퍼 제어 업데이트
                    controller.update_robot_target(target_x, target_z, finger_state)

                    cv2.putText(frame, f"SYNC ON | Grip: {finger_state}", (20, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Robot Z: {target_z:.1f} X: {target_x:.1f}", (20, 80), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            else:
                if is_calibrated:
                    cv2.putText(frame, "HAND LOST - HOLDING", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if not is_calibrated:
                cv2.putText(frame, "Waiting for [Enter]...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

            cv2.imshow("Real-time Shadow Sync", frame)
            
            # ROS 콜백 
            rclpy.spin_once(controller, timeout_sec=0.01)

            if cv2.waitKey(5) & 0xFF == 27: # ESC 종료
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()