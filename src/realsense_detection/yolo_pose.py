# c:\CVpythonWork\realsense\venv\Scripts\Activate.ps1

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

# YOLO Pose 모델
model = YOLO("yolov8n-pose.pt")

# RealSense 설정
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# COCO Keypoint 연결 구조
skeleton = [
    (5, 7), (7, 9),       # 왼팔
    (6, 8), (8, 10),      # 오른팔
    (5, 6),               # 어깨
    (5, 11), (6, 12),     # 몸통
    (11, 13), (13, 15),   # 왼다리
    (12, 14), (14, 16),   # 오른다리
    (11, 12)              # 골반
]

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        frame = cv2.flip(frame, 1)

        results = model(frame)

        for r in results:
            if r.keypoints is None:
                continue

            keypoints = r.keypoints.xy.cpu().numpy()

            for person in keypoints:

                # 관절 점 찍기
                for (x, y) in person:
                    cv2.circle(frame, (int(x), int(y)), 5, (255, 0, 255), -1)

                # 스켈레톤 연결
                for (i, j) in skeleton:
                    x1, y1 = person[i]
                    x2, y2 = person[j]

                    # 좌표가 0이 아닐 때만 그림
                    if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                        cv2.line(frame,
                                 (int(x1), int(y1)),
                                 (int(x2), int(y2)),
                                 (0, 255, 0), 3)

        cv2.imshow("YOLO Pose Skeleton", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()