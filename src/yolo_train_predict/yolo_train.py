from ultralytics import YOLO
from datetime import datetime
import os

project_folder = "cvs"
model_filename = "yolov8n.pt"
data_yaml_filename = "data.yaml"

# 자동으로 현재 폴더 기준 경로 설정
BASE_DIR = os.getcwd()  # 현재 폴더 기준
OUTPUT_DIR = os.path.join(BASE_DIR, project_folder)  # 원하는 상위 폴더 이름

# 날짜/시간 기반 하위 폴더 이름 생성
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
train_name = f"train_{timestamp}"
val_name = f"val_{timestamp}"

model_path = os.path.join(OUTPUT_DIR, model_filename)
data_yaml_path = os.path.join(OUTPUT_DIR, data_yaml_filename)

# Model 로드
model = YOLO(model_path)

# Train 실행
model.train(
    data=data_yaml_path,
    epochs=100,
    imgsz=640,
    batch=8,
    patience=20,
    project=OUTPUT_DIR,
    name=train_name
)

# Validation 실행
model.val(
    project=OUTPUT_DIR,
    name=val_name
)

# 최종 출력 경로 확인
print(f"Train 결과:   {os.path.join(OUTPUT_DIR, train_name)}")
print(f"Val 결과:     {os.path.join(OUTPUT_DIR, val_name)}")
