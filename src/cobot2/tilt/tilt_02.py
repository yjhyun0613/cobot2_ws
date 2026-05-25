import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
#awds
class YoloSamplingNode(Node):
    def __init__(self):
        super().__init__('yolo_sampling_node')
        
        # 1. 모델 로딩
        model_path = '/home/rokey/cobot_ws/src/yjh/resource/hyupdong2_yolo11x_img960_best.pt'
        self.model = YOLO(model_path)
        self.bridge = CvBridge()
        
        # 2. 최신 바운딩 박스 저장을 위한 변수
        self.latest_box = None
        
        # 3. 카메라 구독
        self.subscription = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.image_callback, 10)
        
        # 4. 1초 쿨타임 타이머 설정 (1.0Hz)
        self.timer = self.create_timer(1.0, self.timer_callback)
        
        self.get_logger().info('★ [1초 샘플링] YOLO 좌표 추출 모드 시작 ★')

    def image_callback(self, msg):
        # 영상 처리 및 최신 박스 업데이트
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.model(cv_image, verbose=False)
        
        if len(results[0].boxes) > 0:
            # 첫 번째 발견된 객체의 바운딩 박스만 저장
            self.latest_box = results[0].boxes[0].xyxy[0].cpu().numpy() # [xmin, ymin, xmax, ymax]
        else:
            self.latest_box = None

    def timer_callback(self):
        # 1초마다 실행되는 함수
        if self.latest_box is None:
            self.get_logger().warn('탐지된 객체가 없습니다.')
            return
            
        xmin, ymin, xmax, ymax = self.latest_box
        
        # 1. 중심 좌표
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        
        # 2. 주변 좌표 (바운딩 박스에서 20픽셀 떨어진 지점 예시)
        offset = 20
        p1 = (xmin - offset, ymin - offset) # 좌측 상단 밖
        p2 = (xmax + offset, ymax + offset) # 우측 하단 밖
        p3 = (cx, ymin - offset)            # 중심점 바로 위
        
        self.get_logger().info(f'--- [1초 쿨타임] 추출 데이터 ---')
        self.get_logger().info(f'Center: ({cx:.1f}, {cy:.1f})')
        self.get_logger().info(f'Point 1: ({p1[0]:.1f}, {p1[1]:.1f})')
        self.get_logger().info(f'Point 2: ({p2[0]:.1f}, {p2[1]:.1f})')
        self.get_logger().info(f'Point 3: ({p3[0]:.1f}, {p3[1]:.1f})')

def main(args=None):
    rclpy.init(args=args)
    node = YoloSamplingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()