# c:\CVpythonWork\realsense\venv\Scripts\Activate.ps1
# .\realsense\venv\Scripts\python.exe c:/CVpythonWork/realsense/yolo_world.py

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

# YOLO-World 모델 로드
model = YOLO("yolov8s-world.pt")  # world 모델

# 원하는 객체 직접 정의
# 사람, 이동체, 동물, 산업, 일상물체, 보안/이상행동 등 다양한 객체카테고리 사용가능
# 단, 인식률을 높이기 위해서 1.프롬프트 튜닝(구체화, 의미 명확화, 정확한 산업용 명칭), 2.클래스 분리(한번에 하지 말고 분리)
# 3. threshold조정 4. 해상도 조정 5.finetuning(재학습)
model.set_classes(["person", "bottle", "cup", "phone", "glasses", "helmet", "glove", "elephant", "dog"])

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

        # YOLO-World 추론
        results = model(frame)

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            class_ids = r.boxes.cls.cpu().numpy()

            for box, score, cls in zip(boxes, scores, class_ids):
                x1, y1, x2, y2 = map(int, box)
                label = model.names[int(cls)]

                # 박스
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

                # 라벨
                cv2.putText(frame,
                            f"{label} {score:.2f}",
                            (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0,255,0),
                            2)

        cv2.imshow("YOLO-World (RealSense)", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()