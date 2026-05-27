##########################################
# 1. 여러 객체 검출 후 얻은 bb를 좌표로 변환해서 robot_control_test2에게 전달
# 2. 여러 객체 검출 후 번호 할당해서, 좌표+ 번호 robot_control 노드에 전달
##########################################
import numpy as np
import rclpy
from rclpy.node import Node
from typing import Any, Callable, Optional, Tuple

from ament_index_python.packages import get_package_share_directory
from od_msg.srv import SrvDepthPosition
from object_detection.realsense import ImgNode
from object_detection.yolo_all2 import YoloModel

PACKAGE_NAME = 'object_detection'
PACKAGE_PATH = get_package_share_directory(PACKAGE_NAME)

class ObjectDetectionNode(Node):
    def __init__(self, model_name = 'yolo'):
        super().__init__('object_detection_node')
        self.img_node = ImgNode()
        self.model = self._load_model(model_name)
        self.intrinsics = self._wait_for_valid_data(
            self.img_node.get_camera_intrinsic, "camera intrinsics"
        )
        self.create_service(
            SrvDepthPosition,
            'get_3d_position',
            self.handle_get_depth
        )
        self.get_logger().info("ObjectDetectionNode initialized.")

    def _load_model(self, name):
        if name.lower() == 'yolo':
            return YoloModel()
        raise ValueError(f"Unsupported model: {name}")

    def handle_get_depth(self, request, response):
        self.get_logger().info(f"Received request for target: {request.target}")
        coords_list = self._compute_positions(request.target)
        self.get_logger().info(f"Sorted x,y,z list: {coords_list}")
        response.depth_position = [float(x) for x in coords_list]
        return response

    def _compute_positions(self, target):
        rclpy.spin_once(self.img_node)
        
        # yolo_all.py에 target ("all", "good", "ng" 등) 전달
        detections = self.model.get_all_detections(self.img_node, target)
        
        if not detections:
            self.get_logger().warn("No detection found.")
            return []
        
        valid_objects = []
        
        for det in detections:
            box = None
            label_id = None
            
            if isinstance(det, dict) and 'box' in det:
                box = det['box']
                label_id = det.get('label', -1)
            elif isinstance(det, (list, tuple, np.ndarray)) and len(det) >= 4:
                if isinstance(det[0], (list, tuple, np.ndarray)):
                    box = det[0]
                else:
                    box = det
            else:
                continue
            
            if not isinstance(box, (list, tuple, np.ndarray)) or len(box) < 4:
                continue
            
            try:
                cx = int((box[0] + box[2]) / 2)
                cy = int((box[1] + box[3]) / 2)
            except Exception as e:
                self.get_logger().warn(f"Error calculating center for box {box}: {e}")
                continue
                
            cz = self._get_depth(cx, cy)
            
            if cz is None or cz <= 0:
                self.get_logger().warn(f"Invalid depth at ({cx}, {cy}). Skipping object.")
                continue

            x, y, z = self._pixel_to_camera_coords(cx, cy, cz)
            valid_objects.append({'label': label_id, 'cx': cx, 'cy': cy, 'x': x, 'y': y, 'z': z})

        # =================================================================
        # Detection 노드 내 번호 할당 (정렬) 로직
        # =================================================================
        # 이미지의 cy(Y축)가 작을수록 위, cx(X축)가 작을수록 왼쪽을 의미
        tolerance_px = 30.0 
        valid_objects.sort(key=lambda o: (round(o['cy'] / tolerance_px), o['cx']))

        # 감지된 전체 객체 로그 출력
        for i, obj in enumerate(valid_objects):
            class_name = "Class_" + str(obj['label'])
            if obj['label'] == 0: class_name = "good"
            elif obj['label'] == 1: class_name = "ng"
            self.get_logger().info(f"[pos{i+1} - {class_name}] img({obj['cx']}, {obj['cy']}) -> cam({obj['x']:.1f}, {obj['y']:.1f}, {obj['z']:.1f})")

        # 정렬된 객체 중 첫 번째 나사 좌표만 반환
        first = valid_objects[0]
        class_name = "Class_" + str(first['label'])
        if first['label'] == 0: class_name = "good"
        elif first['label'] == 1: class_name = "ng"
        self.get_logger().info(f"[선택됨 - {class_name}] img({first['cx']}, {first['cy']}) -> cam({first['x']:.1f}, {first['y']:.1f}, {first['z']:.1f})")

        return [first['x'], first['y'], first['z']]

    def _get_depth(self, x, y):
        frame = self._wait_for_valid_data(self.img_node.get_depth_frame, "depth frame")
        try:
            return frame[y, x]
        except IndexError:
            self.get_logger().warn(f"Coordinates ({x},{y}) out of range.")
            return None

    def _wait_for_valid_data(self, getter, description):
        data = getter()
        while data is None or (isinstance(data, np.ndarray) and not data.any()):
            rclpy.spin_once(self.img_node)
            self.get_logger().info(f"Retry getting {description}.")
            data = getter()
        return data

    def _pixel_to_camera_coords(self, x, y, z):
        fx = self.intrinsics['fx']
        fy = self.intrinsics['fy']
        ppx = self.intrinsics['ppx']
        ppy = self.intrinsics['ppy']
        return (
            (x - ppx) * z / fx,
            (y - ppy) * z / fy,
            z
        )

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()