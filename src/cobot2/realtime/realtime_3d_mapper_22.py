import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
import sensor_msgs_py.point_cloud2 as pc2
from tf2_ros import Buffer, TransformListener, StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped, Point # Point 추가
import numpy as np
from scipy.spatial.transform import Rotation as R
import struct
import cv2
import datetime
import json
from cv_bridge import CvBridge
from ultralytics import YOLO
import os

# --- 사용자가 지정한 서비스 메시지 임포트 ---
from od_msg.srv import SrvDepthPosition

# --- Firebase 라이브러리 추가 ---
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
from firebase_admin import db

# --- Firebase 설정 변수 (본인 환경에 맞게 확인) ---
SERVICE_ACCOUNT_KEY_PATH = '/home/rokey/yjh/rokey-d-2-4c32a-firebase-adminsdk-fbsvc-7f5d874f48.json'
STORAGE_BUCKET_NAME = 'rokey-d-2-4c32a.firebasestorage.app'
DATABASE_URL = 'https://rokey-d-2-4c32a-default-rtdb.asia-southeast1.firebasedatabase.app'

class Yolo3DServiceNode(Node):
    def __init__(self):
        super().__init__('yolo_3d_mapper_service_node')

        # 1. TF 설정 (순정 상태 유지)
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        self.publish_static_transform()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # 2. YOLO 및 OpenCV 설정
        self.bridge = CvBridge()
        model_path = '/home/rokey/cobot_ws/src/yjh/resource/hyupdong2_yolo11x_img960_best.pt'
        self.get_logger().info(f'YOLO 모델 로딩 중... ({model_path})')
        self.yolo_model = YOLO(model_path)
        self.get_logger().info('✅ YOLO 모델 로딩 완료!')
        
        # 3. Firebase 초기화
        try:
            cred = credentials.Certificate(SERVICE_ACCOUNT_KEY_PATH)
            if not firebase_admin._apps:
                firebase_admin.initialize_app(cred, {
                    'storageBucket': STORAGE_BUCKET_NAME,
                    'databaseURL': DATABASE_URL
                })
            self.bucket = storage.bucket()
            self.get_logger().info('✅ Firebase Storage & Realtime DB 연결 완료!')
        except Exception as e:
            self.get_logger().error(f'❌ Firebase 초기화 실패: {e}')
            self.bucket = None

        self.latest_pc_msg = None
        self.latest_img_msg = None
        self.latest_depth_msg = None
        self.cam_info = None
        
        # 센서 데이터 구독
        self.sub_pc = self.create_subscription(PointCloud2, '/camera/camera/depth/color/points', self.pc_callback, 10)
        self.sub_img = self.create_subscription(Image, '/camera/camera/color/image_raw', self.img_callback, 10)
        self.sub_depth = self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        self.sub_info = self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.info_callback, 10)

        self.capture_count = 0

        # 4. 서비스 서버 생성 (키보드 스레드 대신 들어간 핵심 코드)
        self.srv = self.create_service(
            SrvDepthPosition,
            'get_3d_position',
            self.handle_get_depth
        )
        self.get_logger().info('★ [서비스 대기 중] 클라이언트의 get_3d_position 요청을 기다립니다... ★')

    def publish_static_transform(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'link_6'
        t.child_frame_id = 'camera_link'
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.075
        t.transform.translation.z = 0.03991
        t.transform.rotation.x = -0.5
        t.transform.rotation.y = -0.5
        t.transform.rotation.z = -0.5
        t.transform.rotation.w = 0.5
        self.tf_static_broadcaster.sendTransform(t)

    def pc_callback(self, msg): self.latest_pc_msg = msg
    def img_callback(self, msg): self.latest_img_msg = msg
    def depth_callback(self, msg): self.latest_depth_msg = msg
    def info_callback(self, msg): self.cam_info = msg

    def get_tf_matrix(self, target_frame, source_frame):
        try:
            trans = self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0))
            rot = R.from_quat([trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w]).as_matrix()
            mat = np.eye(4)
            # mm 단위 변환
            mat[:3, 3] = [trans.transform.translation.x * 1000.0, 
                          trans.transform.translation.y * 1000.0, 
                          trans.transform.translation.z * 1000.0]
            return mat
        except: return None
        
    def upload_to_firebase(self, file_path, destination_blob_name, content_type):
        if not self.bucket: return None
        try:
            blob = self.bucket.blob(destination_blob_name)
            blob.upload_from_filename(file_path, content_type=content_type)
            blob.make_public()
            return blob.public_url
        except Exception as e:
            self.get_logger().error(f'❌ 파일 업로드 실패: {e}')
            return None

    # 서비스 요청이 들어왔을 때 실행되는 콜백 함수
    def handle_get_depth(self, request, response):
        self.get_logger().info('📸 [서비스 요청 수신] 단차 검사 및 데이터 업로드를 시작합니다...')

        if not all([self.latest_pc_msg, self.latest_img_msg, self.latest_depth_msg, self.cam_info]):
            self.get_logger().warn('카메라 데이터를 기다리는 중입니다... 다시 요청해주세요.')
            return response

        pc_msg, img_msg, depth_msg, cam_info = self.latest_pc_msg, self.latest_img_msg, self.latest_depth_msg, self.cam_info
        self.capture_count += 1
        current_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        tf_pc = self.get_tf_matrix('base_link', pc_msg.header.frame_id)
        tf_color = self.get_tf_matrix('base_link', img_msg.header.frame_id)
        
        fx, fy, cx, cy = cam_info.k[0], cam_info.k[4], cam_info.k[2], cam_info.k[5]
        cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        
        yolo_results = self.yolo_model(cv_image, verbose=False)[0]
        
        markers_data = {}
        marker_idx = 1
        
        # 반환할 XYZ 좌표를 담을 리스트 (Point[] 형태)
        response_positions = []

        for box in yolo_results.boxes:
            b = box.xyxy[0].cpu().numpy()
            xmin, ymin, xmax, ymax = b[0], b[1], b[2], b[3]
            cx_img, cy_img = int((xmin + xmax) / 2), int((ymin + ymax) / 2)
            bw_img, bh_img = int(xmax - xmin), int(ymax - ymin)

            screw_pts, surf_pts = [], []
            h, w = cv_depth.shape
            search_rx, search_ry = int(bw_img * 0.75), int(bh_img * 0.75)

            for dy in range(-search_ry, search_ry + 1):
                for dx in range(-search_rx, search_rx + 1):
                    nx, ny = cx_img + dx, cy_img + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        z_mm = float(cv_depth[ny, nx])
                        if z_mm > 0:
                            if abs(dx) < bw_img * 0.1 and abs(dy) < bh_img * 0.1: screw_pts.append(z_mm)
                            elif abs(dx) > bw_img * 0.5 or abs(dy) > bh_img * 0.5: surf_pts.append(z_mm)

            if not screw_pts or not surf_pts: continue
            
            z_screw = min(screw_pts) 
            cam_z = z_screw 
            if cam_z > 1000.0: continue # 1m 이상 컷
            
            z_surf = np.median(surf_pts) 
            delta_z_mm = z_surf - z_screw
            is_normal = not (delta_z_mm > 10.0)
            
            pt_base = tf_color @ np.array([(cx_img - cx) * cam_z / fx, (cy_img - cy) * cam_z / fy, cam_z, 1.0])
            
            # DB용 JSON 포맷
            markers_data[str(marker_idx)] = {
                "position": {"x": float(pt_base[0]), "y": float(pt_base[1]), "z": float(pt_base[2])},
                "status": is_normal,
                "time": current_time_str,
                "torque": 0.0 
            }
            status_text = "정상" if is_normal else "불량(돌출)"
            self.get_logger().info(f'🎯 [나사 {marker_idx}] 상태: {status_text} | Z: {pt_base[2]:.1f}mm')
            
            # # 🚀 서비스 응답을 위한 Point 객체 생성 및 리스트에 추가
            # p = Point()
            # p.x = float(pt_base[0])
            # p.y = float(pt_base[1])
            # p.z = float(pt_base[2])
            # response_positions.append(p)

            response_positions.extend([float(pt_base[0]), float(pt_base[1]), float(pt_base[2])])
            
            marker_idx += 1

        # 배경 포인트 클라우드 처리 (m -> mm)
        pc_data = list(pc2.read_points(pc_msg, field_names=("x", "y", "z", "rgb"), skip_nans=True))
        pts = np.array([[p[0], p[1], p[2]] for p in pc_data]) * 1000.0
        colors = np.array([[(struct.unpack('I', struct.pack('f', p[3]))[0] >> 16 & 0xFF)/255.0, 
                            (struct.unpack('I', struct.pack('f', p[3]))[0] >> 8 & 0xFF)/255.0, 
                            (struct.unpack('I', struct.pack('f', p[3]))[0] & 0xFF)/255.0] for p in pc_data])
        
        distances = np.linalg.norm(pts, axis=1)
        mask = distances <= 1000.0 
        pts_filtered, colors_filtered = pts[mask], colors[mask]
        pts_transformed = (tf_pc @ np.hstack([pts_filtered, np.ones((len(pts_filtered), 1))]).T).T[:, :3]
        
        # JS 파일 생성 및 Storage 업로드
        step = 3
        bg_data = {
            "x": pts_transformed[:, 0].tolist()[::step],
            "y": pts_transformed[:, 1].tolist()[::step],
            "z": pts_transformed[:, 2].tolist()[::step],
            "colors": [f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' for c in colors_filtered][::step]
        }
        js_content = "window.latestBackground = " + json.dumps(bg_data) + ";"
        bg_filename = f'background_{self.capture_count}.js'
        with open(bg_filename, 'w') as f:
            f.write(js_content)
            
        bg_url = self.upload_to_firebase(bg_filename, f'backgrounds/{bg_filename}', 'application/javascript')

        # Realtime DB 업로드 (요청이 들어올 때마다 매번 업로드)
        db_data = {"markers": markers_data, "background_url": bg_url if bg_url else ""}
        if self.bucket:
            ref = db.reference('linestatus') 
            ref.child(str(self.capture_count)).set(db_data) 
            self.get_logger().info(f'💾 DB (회차: {self.capture_count}) 업데이트 완료!')

        # 🚀 클라이언트에게 최종 응답 보내기
        # 주의: .srv 파일 내에 배열 이름이 무엇으로 정의되어 있는지에 따라 아래 'response.positions' 부분을 수정해 주세요.
        try:
            # 예: srv 파일에 `geometry_msgs/Point[] positions` 라고 정의되어 있는 경우
            response.depth_position = response_positions  
        except AttributeError:
            self.get_logger().warn('SrvDepthPosition 내의 배열 변수명이 positions 가 아닙니다. 코드를 수정해주세요.')
        
        self.get_logger().info('✅ 서비스 처리 완료! 좌표 리턴 성공.')
        return response

def main(args=None):
    rclpy.init(args=args)
    node = Yolo3DServiceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()