import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
import sensor_msgs_py.point_cloud2 as pc2
from tf2_ros import Buffer, TransformListener, StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
import numpy as np
from scipy.spatial.transform import Rotation as R
import struct
import cv2
from cv_bridge import CvBridge
from ultralytics import YOLO
import os
import json
from datetime import datetime

# 🌟 ROS 2 표준 서비스 (통신용)
from std_srvs.srv import Trigger

# --- Firebase ---
import firebase_admin
from firebase_admin import credentials, storage, db

SERVICE_ACCOUNT_KEY_PATH = '/home/yoon/YJH/resource/rokey-d-2-4c32a-firebase-adminsdk-fbsvc-3c31d8ba34.json'
STORAGE_BUCKET_NAME = 'rokey-d-2-4c32a.firebasestorage.app'
DATABASE_URL = 'https://rokey-d-2-4c32a-default-rtdb.asia-southeast1.firebasedatabase.app'

class VisionServerNode(Node):
    def __init__(self):
        super().__init__('vision_server_node')

        # TF 설정
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        self.publish_static_transform()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # YOLO 로딩
        self.bridge = CvBridge()
        model_path = '/home/yoon/YJH/resource/hyupdong2_yolo11x_img960_best.pt'
        self.get_logger().info('YOLO 모델 로딩 중...')
        self.yolo_model = YOLO(model_path)
        self.get_logger().info('✅ YOLO 모델 로딩 완료!')
        
        # Firebase 초기화
        try:
            cred = credentials.Certificate(SERVICE_ACCOUNT_KEY_PATH)
            if not firebase_admin._apps:
                firebase_admin.initialize_app(cred, {
                    'storageBucket': STORAGE_BUCKET_NAME,
                    'databaseURL': DATABASE_URL
                })
            self.bucket = storage.bucket()
            self.db_ref = db.reference('linestatus')
            self.init_capture_count()
            self.get_logger().info('✅ Firebase 연결 완료!')
        except Exception as e:
            self.get_logger().error(f'Firebase 오류: {e}')
            self.bucket, self.db_ref, self.capture_count = None, None, 0

        # 데이터 구독
        self.latest_pc_msg = self.latest_img_msg = self.latest_depth_msg = self.cam_info = None
        self.create_subscription(PointCloud2, '/camera/camera/depth/color/points', self.pc_callback, 1)
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.img_callback, 1)
        self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, 1)
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.info_callback, 1)

        # 🌟 로봇의 검사 요청을 기다리는 서비스 서버 생성
        self.srv = self.create_service(Trigger, '/vision_inspect', self.inspect_callback)
        self.get_logger().info('🤖 비전 서버 대기 중... 로봇의 이동 완료 신호를 기다립니다.')

    def init_capture_count(self):
        snapshot = self.db_ref.get() if self.db_ref else None
        self.capture_count = len(snapshot.keys()) if snapshot else 0
        self.get_logger().info(f'🔄 기존 데이터 확인: 다음 저장될 이름은 "작업대 {self.capture_count + 1}" 입니다.')

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

    def get_tf_matrix(self, target, source):
        try:
            trans = self.tf_buffer.lookup_transform(target, source, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0))
            rot = R.from_quat([trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w]).as_matrix()
            mat = np.eye(4); mat[:3, :3] = rot; mat[:3, 3] = [trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z]
            return mat
        except: return None

    # 🌟 핵심: 로봇이 호출할 때 실행되는 함수
    def inspect_callback(self, request, response):
        if not all([self.latest_pc_msg, self.latest_img_msg, self.latest_depth_msg, self.cam_info]):
            response.success = False; response.message = "데이터 수신 오류"
            return response

        cv_image = self.bridge.imgmsg_to_cv2(self.latest_img_msg, desired_encoding='bgr8')
        cv_depth = self.bridge.imgmsg_to_cv2(self.latest_depth_msg, desired_encoding='passthrough')
        
        yolo_results = self.yolo_model(cv_image, verbose=False)[0]

        # 💡 조건 2: 나사가 하나도 없으면 스킵! (서버 연산 종료 후 즉각 리턴)
        if len(yolo_results.boxes) == 0:
            self.get_logger().info('⏩ 나사 미발견: 검사를 생략하고 다음으로 넘어갑니다.')
            response.success = True 
            response.message = "SKIPPED"
            return response

        # 💡 조건 3: 나사가 있을 때만 순서(카운트)를 올림
        self.capture_count += 1
        section_name = f"Workspace {self.capture_count}"
        self.get_logger().info(f'📸 나사 발견! [{section_name}] 정밀 분석 및 업로드 시작...')

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        time_display_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        tf_pc = self.get_tf_matrix('base_link', self.latest_pc_msg.header.frame_id)
        tf_color = self.get_tf_matrix('base_link', self.latest_img_msg.header.frame_id)
        fx, fy, cx, cy = self.cam_info.k[0], self.cam_info.k[4], self.cam_info.k[2], self.cam_info.k[5]
        
        markers_data = {}
        screw_idx = 0
        
        for box in yolo_results.boxes:
            b = box.xyxy[0].cpu().numpy()
            cx_img, cy_img = int((b[0]+b[2])/2), int((b[1]+b[3])/2)
            bw, bh = int(b[2]-b[0]), int(b[3]-b[1])
            s_pts, surf_pts = [], []
            h, w = cv_depth.shape
            
            for dy in range(-int(bh*0.75), int(bh*0.75)+1):
                for dx in range(-int(bw*0.75), int(bw*0.75)+1):
                    nx, ny = cx_img+dx, cy_img+dy
                    if 0<=nx<w and 0<=ny<h:
                        z_mm = float(cv_depth[ny, nx])
                        if z_mm > 0:
                            if abs(dx)<bw*0.1 and abs(dy)<bh*0.1: s_pts.append(z_mm)
                            elif abs(dx)>bw*0.5 or abs(dy)>bh*0.5: surf_pts.append(z_mm)
            if not s_pts or not surf_pts: continue
            
            cam_z = min(s_pts) / 1000.0
            if not (0.25 < cam_z < 1.0): continue
            
            delta_z = np.median(surf_pts) - (cam_z*1000)
            status_bool = bool(delta_z <= 10.0)
            pt_base = tf_color @ np.array([(cx_img-cx)*cam_z/fx, (cy_img-cy)*cam_z/fy, cam_z, 1.0])
            
            markers_data[str(screw_idx)] = {
                "status": status_bool,
                "position": {"x": float(pt_base[0]*1000), "y": float(pt_base[1]*1000), "z": float(pt_base[2]*1000)},
                "time": time_display_str
            }
            screw_idx += 1

        self.get_logger().info('☁️ 3D 맵핑 처리 중...')
        pc_data = list(pc2.read_points(self.latest_pc_msg, field_names=("x","y","z","rgb"), skip_nans=True))
        pts = np.array([[p[0], p[1], p[2]] for p in pc_data])
        colors = np.array([[(struct.unpack('I', struct.pack('f', p[3]))[0] >> 16 & 0xFF)/255.0, 
                            (struct.unpack('I', struct.pack('f', p[3]))[0] >> 8 & 0xFF)/255.0, 
                            (struct.unpack('I', struct.pack('f', p[3]))[0] & 0xFF)/255.0] for p in pc_data])
        mask = (np.linalg.norm(pts, axis=1) >= 0.25) & (np.linalg.norm(pts, axis=1) <= 1.0)
        pts_filtered, colors_filtered = pts[mask], colors[mask]
        pts_transformed = (tf_pc @ np.hstack([pts_filtered, np.ones((len(pts_filtered), 1))]).T).T[:, :3] * 1000.0
        
        js_filename = f'bg_{timestamp_str}.js'
        with open(js_filename, 'w') as f:
            f.write(f"window.latestBackground = {json.dumps({'x': np.round(pts_transformed[:,0],4).tolist(), 'y': np.round(pts_transformed[:,1],4).tolist(), 'z': np.round(pts_transformed[:,2],4).tolist(), 'colors': [f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' for c in colors_filtered]})};")

        blob = self.bucket.blob(f'backgrounds/{js_filename}')
        blob.upload_from_filename(js_filename, content_type='application/javascript')
        blob.make_public()

        if self.db_ref:
            self.db_ref.child(section_name).set({
                "capture_index": self.capture_count, "timestamp": timestamp_str, 
                "background_url": blob.public_url, "markers": markers_data
            })
            self.get_logger().info(f'🎉 [{section_name}] 저장 완료! (나사 {screw_idx}개)')
        
        if os.path.exists(js_filename): os.remove(js_filename)

        # 💡 조건 1: 모든 업로드가 끝나면 로봇에게 "이제 움직여도 돼!"라고 응답
        response.success = True
        response.message = "UPLOADED"
        return response

def main(args=None):
    rclpy.init(args=args)
    node = VisionServerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()