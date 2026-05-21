import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
import tf2_ros
import tf2_geometry_msgs
from cv_bridge import CvBridge
from ultralytics import YOLO
import numpy as np

class YoloWorldSamplerNode(Node):
    def __init__(self):
        super().__init__('yolo_world_sampler_node')
        
        model_path = '/home/rokey/cobot_ws/src/yjh/resource/hyupdong2_yolo11x_img960_best.pt'
        self.model = YOLO(model_path)
        self.bridge = CvBridge()
        
        self.latest_box = None
        self.depth_image = None
        self.camera_info = None
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, 10)
        self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.info_callback, 10)
        
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.get_logger().info('★ [디버깅 모드] 좌표 샘플러 시작 ★')

    def info_callback(self, msg): self.camera_info = msg
    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        # Depth 데이터가 들어오는지 확인
        if self.depth_image is not None:
            self.get_logger().info(f'Depth 이미지 수신 중: {self.depth_image.shape}', once=True)

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.model(cv_image, verbose=False)
        if len(results[0].boxes) > 0:
            self.latest_box = results[0].boxes[0].xyxy[0].cpu().numpy()
        else:
            self.latest_box = None

    def timer_callback(self):
        if self.latest_box is None:
            self.get_logger().warn('탐지된 객체 없음')
            return
        if self.depth_image is None:
            self.get_logger().warn('Depth 이미지 데이터 없음!')
            return
            
        xmin, ymin, xmax, ymax = self.latest_box
        cx, cy = int((xmin + xmax) / 2), int((ymin + ymax) / 2)
        
        # 🌟 디버깅: 해당 좌표의 RAW Depth 값 확인
        raw_depth = self.depth_image[cy, cx]
        self.get_logger().info(f'Center ({cx}, {cy})의 Raw Depth 값: {raw_depth}')

        if raw_depth <= 0:
            self.get_logger().error('Depth 값이 0입니다! 카메라 설정/거리 확인 필요')
            return
            
        # 변환 시도
        z = float(raw_depth) * 0.001
        self.get_logger().info(f'계산된 Z거리: {z:.3f}m')

def main(args=None):
    rclpy.init(args=args)
    node = YoloWorldSamplerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()