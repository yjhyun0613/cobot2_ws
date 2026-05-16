# .\realsense\venv\Scripts\python.exe c:/CVpythonWork/realsense/mediapipe_hands.py

import cv2
import numpy as np
import pyrealsense2 as rs
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# RealSense 설정
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        cv2.imshow("MediaPipe Hands", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    