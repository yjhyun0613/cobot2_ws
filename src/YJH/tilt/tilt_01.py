import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2

class YoloCenterNode(Node):
    def __init__(self):
        super().__init__('yolo_center_node')
        
        # 1. 모델 로딩 (본인의 .pt 파일 경로로 수정하세요)
        model_path = '/home/rokey/cobot_ws/src/yjh/resource/hyupdong2_yolo11x_img960_best.pt'
        self.model = YOLO(model_path)
        self.bridge = CvBridge()
        
        # 2. 카메라 구독 (RealSense color topic)
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10)
        
        self.get_logger().info('★ YOLO 중심 좌표 추출 노드 시작됨 (영상 데이터 대기 중...) ★')

    def image_callback(self, msg):
        # 1. ROS 이미지를 OpenCV로 변환
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # 2. YOLO 추론
        results = self.model(cv_image, verbose=False)
        
        # 3. 객체 탐지 결과 처리
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 바운딩 박스 좌표 추출 (x1, y1, x2, y2)
                b = box.xyxy[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.model.names[class_id]
                
                # 중심 좌표 계산
                cx = (b[0] + b[2]) / 2
                cy = (b[1] + b[3]) / 2
                
                # 결과 출력
                self.get_logger().info(f'탐지: {class_name} | 중심 좌표: (x: {cx:.1f}, y: {cy:.1f})')

def main(args=None):
    rclpy.init(args=args)
    node = YoloCenterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()