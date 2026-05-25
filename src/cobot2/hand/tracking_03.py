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

# 거리(픽셀) 변화를 로봇 Z축(mm)으로 변환할 스케일
# 안전을 위해 값을 0.5로 대폭 낮췄습니다. 움직임이 너무 적으면 1.0, 2.0으로 서서히 올려주세요.
SCALE_Z_DISTANCE = 0.5 

# 데드존 (단위: 픽셀)
# 이 픽셀 수치 이내의 미세한 손떨림이나 비전 노이즈는 무시합니다.
DEADZONE_PIXELS = 3.0  

calibration_triggered = False

# 손가락 굽힘을 통한 그리퍼 제어
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
        self.get_logger().info("Shadow Sync Node Initialized. (Z-Axis Only Mode)")
        
        topic_name = f'/{ROBOT_ID}/servol_stream'
        self.servol_pub = self.create_publisher(ServolStream, topic_name, 10)

        # 현재 필터링된 목표 위치 (X는 고정이므로 Z만 의미 있게 변함)
        self.filtered_x = ROBOT_BASE_X
        self.filtered_z = ROBOT_BASE_Z
        
        # 스무딩 필터 강도 (작을수록 부드러움)
        self.alpha = 0.08 
        # 한 사이클당 최대 이동 허용량 (mm) - 급발진 방지
        self.max_step = 15.0 
        
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
        # 1. 변화량 제한 (Rate Limiting)
        diff_x = target_x - self.filtered_x
        diff_z = target_z - self.filtered_z

        target_x = self.filtered_x + np.clip(diff_x, -self.max_step, self.max_step)
        target_z = self.filtered_z + np.clip(diff_z, -self.max_step, self.max_step)

        # 2. 위치 스무딩 (EMA 필터)
        self.filtered_x = (1 - self.alpha) * self.filtered_x + (self.alpha * target_x)
        self.filtered_z = (1 - self.alpha) * self.filtered_z + (self.alpha * target_z)

        # 3. 작업 영역 제한
        safe_x = np.clip(self.filtered_x, ROBOT_BASE_X - 300, ROBOT_BASE_X + 300)
        safe_z = np.clip(self.filtered_z, ROBOT_BASE_Z - 200, ROBOT_BASE_Z + 300)

        current_pos = get_current_posx()[0] 
        current_rx = current_pos[3]
        current_rz = current_pos[5]

        # 4. 제어 명령 퍼블리시
        msg = ServolStream()
        msg.pos = [float(safe_x), float(ROBOT_BASE_Y), float(safe_z), 
                   float(current_rx), 179.9, float(current_rz)]
                   
        # 물리적 속도 및 가속도 (급발진 방지)
        msg.vel = [150.0, 50.0] 
        msg.acc = [100.0, 50.0] 
        msg.time = 0.05 
        
        self.servol_pub.publish(msg)

        # 5. 그리퍼 제어 
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
    input("\n[동기화 준비] 평소 작업하는 높이에서 손을 펴고 [Enter]를 누르세요!\n")
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
    base_distance = 0.0
    filtered_distance = 0.0  # 데드존 처리를 위한 변수

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

                # 1. 0번(손목)과 9번(중지 밑단) 픽셀 좌표 추출
                lm0 = hand_landmarks.landmark[0]
                lm9 = hand_landmarks.landmark[9]
                
                p0 = (int(lm0.x * w), int(lm0.y * h))
                p9 = (int(lm9.x * w), int(lm9.y * h))
                
                # 2. 두 점 사이의 실제 픽셀 거리 계산
                current_distance = math.dist(p0, p9)

                if calibration_triggered and not is_calibrated:
                    base_distance = current_distance
                    filtered_distance = current_distance # 초기값 세팅
                    is_calibrated = True
                    print(f"영점 설정 완료! 기준 거리: {base_distance:.1f} px")

                if is_calibrated:
                    # X축: 좌우 움직임을 차단하고 무조건 기본 위치로 고정합니다.
                    target_x = ROBOT_BASE_X

                    # --- [데드존(Deadzone) 로직 적용] ---
                    # 현재 거리가 이전에 저장된 거리보다 DEADZONE_PIXELS 이상 변했을 때만 값 갱신
                    if abs(current_distance - filtered_distance) > DEADZONE_PIXELS:
                        filtered_distance = current_distance

                    # Z축: 갱신된 필터링 거리와 기준 거리의 차이를 반영
                    distance_diff = filtered_distance - base_distance
                    target_z = ROBOT_BASE_Z + (distance_diff * SCALE_Z_DISTANCE)

                    controller.update_robot_target(target_x, target_z, finger_state)

                    # 화면에 현재 상태 출력 (X가 고정되었음을 알림)
                    cv2.putText(frame, f"SYNC ON | Grip: {finger_state}", (20, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Dist: {int(filtered_distance)}px | Z: {target_z:.1f} (X Locked)", (20, 80), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
                    # 시각적 피드백: 0번과 9번을 연결하는 선 그리기
                    cv2.line(frame, p0, p9, (255, 0, 0), 3)

            else:
                if is_calibrated:
                    cv2.putText(frame, "HAND LOST - HOLDING", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if not is_calibrated:
                cv2.putText(frame, "Waiting for [Enter]...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

            cv2.imshow("Real-time Shadow Sync (Z-Only)", frame)
            
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