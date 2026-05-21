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

# 패키지 정보 및 하드웨어 설정
from pick_and_place_text.onrobot import RG
from dsr_msgs2.msg import ServolStream

ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"

rclpy.init()
dsr_node = rclpy.create_node("rokey_shadow_sync", namespace=ROBOT_ID)
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL
DR_init.__dsr__node = dsr_node

try:
    from DSR_ROBOT2 import movej, get_current_posx, mwait, servol
except ImportError as e:
    print(f"Error importing DSR_ROBOT2: {e}")
    sys.exit()

GRIPPER_NAME = "rg2"
TOOLCHANGER_IP = "192.168.1.1"
TOOLCHANGER_PORT = "502"
gripper = RG(GRIPPER_NAME, TOOLCHANGER_IP, TOOLCHANGER_PORT)

# --- 섀도우 동기화 설정값 ---
ROBOT_BASE_X = 367.32
ROBOT_BASE_Y = 3.69    
ROBOT_BASE_Z = 422.92  

SCALE_X = 600.0  
# 손 면적(픽셀 제곱) 변화를 로봇 Z축(mm)으로 변환할 스케일
# (실제 환경에 맞게 테스트하며 조절해야 합니다)
SCALE_Z_AREA = 0.05  

calibration_triggered = False

# 신발끈 공식을 이용한 다각형 넓이 계산 함수
def calculate_area(points):
    area = 0.0
    n = len(points)
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    return abs(area) / 2.0

def distance(p1, p2):
    return math.dist((p1.x, p1.y), (p2.x, p2.y))

def get_finger_state(hand_landmarks):
    points = hand_landmarks.landmark
    open_count = 0
    if distance(points[4], points[9]) > distance(points[3], points[9]): 
        open_count += 1
    for i in range(8, 21, 4):
        if distance(points[i], points[0]) > distance(points[i - 1], points[0]):
            open_count += 1
            
    if open_count >= 4: return "OPEN"
    elif open_count <= 1: return "CLOSE"
    return "HOLD"

class ShadowSyncController(Node):
    def __init__(self):
        super().__init__('shadow_sync_controller')
        self.get_logger().info("Shadow Sync Node Initialized.")
        
        topic_name = f'/{ROBOT_ID}/servol_stream'
        self.servol_pub = self.create_publisher(ServolStream, topic_name, 10)

        self.filtered_x = ROBOT_BASE_X
        self.filtered_z = ROBOT_BASE_Z
        self.alpha = 0.15 
        
        self.current_gripper_state = "OPEN"
        self.last_gripper_cmd_time = time.time()
        
        self.init_robot()

    def init_robot(self):
        self.get_logger().info("로봇 초기 자세(JReady)로 이동합니다.")
        JReady = [0, 0, 90, 0, 90, 0]
        movej(JReady, vel=60, acc=60)
        gripper.open_gripper()
        mwait()
        self.get_logger().info("로봇 초기화 완료.")

    def update_robot_target(self, target_x, target_z, finger_state):
        self.filtered_x = (1 - self.alpha) * self.filtered_x + (self.alpha * target_x)
        self.filtered_z = (1 - self.alpha) * self.filtered_z + (self.alpha * target_z)

        safe_x = np.clip(self.filtered_x, ROBOT_BASE_X - 300, ROBOT_BASE_X + 300)
        safe_z = np.clip(self.filtered_z, ROBOT_BASE_Z - 200, ROBOT_BASE_Z + 300)

        current_pos = get_current_posx()[0] 
        current_rx = current_pos[3]
        current_rz = current_pos[5]

        msg = ServolStream()
        msg.pos = [float(safe_x), float(ROBOT_BASE_Y), float(safe_z), 
                   float(current_rx), 179.9, float(current_rz)]
        msg.vel = [250.0, 50.0] 
        msg.acc = [500.0, 100.0] 
        msg.time = 0.05 
        self.servol_pub.publish(msg)

        current_time = time.time()
        if (current_time - self.last_gripper_cmd_time) > 0.5:
            if finger_state == "OPEN" and self.current_gripper_state != "OPEN":
                gripper.open_gripper() 
                self.current_gripper_state = "OPEN"
                self.last_gripper_cmd_time = current_time
            elif finger_state == "CLOSE" and self.current_gripper_state != "CLOSE":
                gripper.close_gripper()
                self.current_gripper_state = "CLOSE"
                self.last_gripper_cmd_time = current_time

def wait_for_enter():
    global calibration_triggered
    input("\n[동기화 준비] 손을 펴고 카메라 아래에 위치시킨 뒤 [Enter]를 누르세요!\n")
    calibration_triggered = True

def main(args=None):
    controller = ShadowSyncController()
    threading.Thread(target=wait_for_enter, daemon=True).start()

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
    base_hand_x = 0.0
    base_area = 0.0

    try:
        while rclpy.ok():
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            global calibration_triggered

            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                finger_state = get_finger_state(hand_landmarks)

                # 면적을 구하기 위한 6개의 주요 랜드마크 인덱스 (손목, 각 손가락 하단)
                indices = [0, 1, 5, 9, 13, 17]
                
                # 랜드마크를 실제 픽셀 좌표로 변환하여 리스트에 저장
                points = []
                for i in indices:
                    lm = hand_landmarks.landmark[i]
                    points.append((int(lm.x * w), int(lm.y * h)))

                # 현재 면적 계산
                current_area = calculate_area(points)
                current_hx = hand_landmarks.landmark[9].x

                if calibration_triggered and not is_calibrated:
                    base_hand_x = current_hx
                    base_area = current_area
                    is_calibrated = True
                    print(f"영점 설정 완료! 기준 면적: {base_area:.1f}")

                if is_calibrated:
                    # X축: 기존과 동일하게 손의 좌우 이동 사용
                    delta_x = current_hx - base_hand_x
                    target_x = ROBOT_BASE_X + (delta_x * SCALE_X)

                    # Z축: 현재 면적과 기준 면적의 차이를 이용 (면적이 커지면 Z 상승)
                    area_diff = current_area - base_area
                    target_z = ROBOT_BASE_Z + (area_diff * SCALE_Z_AREA)

                    controller.update_robot_target(target_x, target_z, finger_state)

                    cv2.putText(frame, f"SYNC ON | Grip: {finger_state}", (20, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Area: {int(current_area)} | Z: {target_z:.1f}", (20, 80), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            else:
                if is_calibrated:
                    cv2.putText(frame, "HAND LOST - HOLDING", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if not is_calibrated:
                cv2.putText(frame, "Waiting for [Enter]...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

            cv2.imshow("Real-time Shadow Sync", frame)
            
            rclpy.spin_once(controller, timeout_sec=0.01)

            if cv2.waitKey(5) & 0xFF == 27: 
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()