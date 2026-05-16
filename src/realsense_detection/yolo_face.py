# c:\CVpythonWork\realsense\venv\Scripts\Activate.ps1
# .\realsense\venv\Scripts\python.exe c:/CVpythonWork/realsense/yolo_face.py

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

# YOLO 얼굴 모델 (자동 다운로드)
model = YOLO("/home/yoon/cobot_ws/src/cobot2_ws/realsense_detection/yolov8n-face.pt")

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

        # YOLO 추론
        results = model(frame)

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()

            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, "Face", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.imshow("YOLO Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows() 