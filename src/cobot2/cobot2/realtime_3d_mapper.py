import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
import cv2
import numpy as np

class MinimalDepthViewer(Node):
    def __init__(self):
        super().__init__('minimal_depth_viewer')
        
        self.bridge = CvBridge()
        
        # 정렬된 뎁스 이미지 토픽 구독 (QoS 필수 적용!)
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/aligned_depth_to_color/image_raw',
            self.depth_callback,
            qos_profile_sensor_data
        )
        
        self.get_logger().info('초간단 뎁스 뷰어 시작: 깊이 영상을 기다립니다...')

    def depth_callback(self, msg):
        try:
            # 리얼센스 뎁스는 16비트 정수형(16UC1, 단위: mm) 데이터입니다.
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")
        except Exception as e:
            self.get_logger().error(f'뎁스 이미지 변환 에러: {e}')
            return

        # 눈으로 거리 차이를 쉽게 식별할 수 있도록 스케일링 및 컬러맵 적용
        # 1. 0~4m (0~4000mm) 사이의 거리를 0~255 값으로 압축합니다.
        depth_scaling = cv2.convertScaleAbs(depth_image, alpha=255.0/4000.0)
        
        # 2. 압축된 흑백 음영에 제트(JET) 컬러맵을 입혀 화려한 색상으로 바꿉니다.
        # (가까운 곳은 빨간색/보라색, 먼 곳은 파란색 등으로 표현됨)
        depth_colormap = cv2.applyColorMap(depth_scaling, cv2.COLORMAP_JET)

        # 화면에 팝업창을 띄워 실시간 시각화
        cv2.imshow("RealSense Depth Camera", depth_colormap)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = MinimalDepthViewer()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()