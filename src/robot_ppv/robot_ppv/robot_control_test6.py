##########################
"""
1. 단일 객체 인식 됬을 때 그 좌표로 이동
2. 복수 의 객체 인식 됬을 때 순서대로 좌표 이동 후 노드 종료
2.1 음성 추가
3. 전체 조사 기능 추가(
    - 전체 조사 기능>> 높은 좌표에서 각 나사 확인
    - 각 나사 번호 메기기 >> 좌표 통해 규칙 정해서 순서
4. 번호 받고 이동 기능
    - 번호를 yolo에서 할당으로 수정 >> 
    - 사용자가 번호 부르면 그 나사로 이동 조사 후 귀환
5. yolo에서 받는것이 아니라 DB에서 모든 나사 좌표와 번호를 가져옴 (Firebase 연동)
6. 명령(vision_check, torque_check)과 위치(pos)를 분리하여 다중/개별 동작 수행
"""
##########################

import os
import time
import sys
from scipy.spatial.transform import Rotation
import numpy as np
import rclpy
from rclpy.node import Node
import DR_init

# Firebase Admin SDK import
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

from od_msg.srv import SrvDepthPosition
from std_srvs.srv import Trigger
from ament_index_python.packages import get_package_share_directory
from robot_control.onrobot import RG

package_path = get_package_share_directory("robot_ppv")

ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
VELOCITY, ACC = 60, 60
JHOME_POS = [0, 0, 90, 0, 90, 0]
GRIPPER_NAME = "rg2"
TOOLCHARGER_IP = "192.168.1.1"
TOOLCHARGER_PORT = "502"
DEPTH_OFFSET = -39.0
MIN_DEPTH = 2.0

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

rclpy.init()
dsr_node = rclpy.create_node("robot_control_node", namespace=ROBOT_ID)
DR_init.__dsr__node = dsr_node

try:
    from DSR_ROBOT2 import movej, movel, get_current_posx, mwait, trans, posx
except ImportError as e:
    sys.exit()

gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)


class RobotController(Node):
    def __init__(self):
        super().__init__("robot_integrated_controller")
        
        # 1. Firebase 초기화 설정
        self.init_firebase()
        
        self.init_robot()

        self.get_position_client = self.create_client(SrvDepthPosition, "/get_3d_position")
  
        while not self.get_position_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().info("Waiting for get_depth_position service...")

        self.get_keyword_client = self.create_client(Trigger, "/get_keyword")
        while not self.get_keyword_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().info("Waiting for get_keyword service...")
        self.get_keyword_request = Trigger.Request()
        
        # 번호별 위치 좌표를 기억하기 위한 딕셔너리 (DB에서 가져와 저장)
        self.saved_positions = {}

    def init_firebase(self):
        """Firebase Admin SDK 초기화"""
        # 제공된 JSON 인증 키 파일 경로 설정
        cred_path = os.path.join(package_path, "resource", "rokey-d-2-4c32a-firebase-adminsdk-fbsvc-7f5d874f48.json")
        
        # Firebase 앱이 중복 초기화되지 않도록 확인
        if not firebase_admin._apps:
            try:
                cred = credentials.Certificate(cred_path)
                # 데이터베이스 URL 설정 [cite: 1, 3]
                firebase_admin.initialize_app(cred, {
                    'databaseURL': 'https://rokey-d-2-4c32a-default-rtdb.asia-southeast1.firebasedatabase.app'
                })
                self.get_logger().info("Firebase 초기화 완료")
            except Exception as e:
                self.get_logger().error(f"Firebase 초기화 실패: {e}")

    def fetch_positions_from_db(self, line_id="3"):
        """Firebase DB에서 특정 라인의 마커 좌표와 할당 번호를 가져옵니다."""
        try:
            # DB 구조 기반: /linestatus/3/markers 경로 참조
            ref = db.reference(f'/linestatus/{line_id}/markers')
            markers = ref.get()
            
            if not markers:
                self.get_logger().warn("DB에서 데이터를 찾을 수 없습니다.")
                return False
                
            self.saved_positions.clear() # 기존 저장소 초기화
            robot_posx = get_current_posx()[0] # 현재 로봇 자세 (Rx, Ry, Rz 기본값용)

            # markers가 리스트인지 딕셔너리인지에 따라 파싱
            items = markers.items() if isinstance(markers, dict) else enumerate(markers)
            
            for marker_id, data in items:
                if data and 'position' in data:
                    pos_data = data['position']
                    
                    # DB에 저장된 position 형태에 맞춰 파싱 (리스트 또는 딕셔너리 가정)
                    if isinstance(pos_data, dict):
                        x, y, z = pos_data.get('x', 0.0), pos_data.get('y', 0.0), pos_data.get('z', 0.0)
                        target_coords = [x, y, z, robot_posx[3], robot_posx[4], robot_posx[5]]
                    elif isinstance(pos_data, (list, tuple)):
                        if len(pos_data) == 3:
                            target_coords = list(pos_data) + list(robot_posx[3:])
                        elif len(pos_data) == 6:
                            target_coords = list(pos_data)
                        else:
                            continue
                    else:
                        continue
                        
                    # 딕셔너리에 저장 (예: "1", "2" ...)
                    str_marker_id = str(marker_id)
                    self.saved_positions[str_marker_id] = target_coords
                    self.get_logger().info(f"[DB 저장 완료] 나사 번호 {str_marker_id}: {target_coords[:3]}")
                    
            return True
            
        except Exception as e:
            self.get_logger().error(f"DB 데이터 불러오기 오류: {e}")
            return False

    def get_robot_pose_matrix(self, x, y, z, rx, ry, rz):
        R = Rotation.from_euler("ZYZ", [rx, ry, rz], degrees=True).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T

    def transform_to_base(self, camera_coords, gripper2cam_path, robot_pos):
        gripper2cam = np.load(gripper2cam_path)
        coord = np.append(np.array(camera_coords), 1)
        x, y, z, rx, ry, rz = robot_pos
        base2gripper = self.get_robot_pose_matrix(x, y, z, rx, ry, rz)
        base2cam = base2gripper @ gripper2cam
        td_coord = np.dot(base2cam, coord)
        return td_coord[:3]

    def get_target_list(self, target):
        """기존 Detection 서비스 호출 노드 로직"""
        new_request = SrvDepthPosition.Request()
        new_request.target = target
       
        future = self.get_position_client.call_async(new_request)
        rclpy.spin_until_future_complete(self, future)

        if future.result():
            result = future.result().depth_position.tolist()
            if not result or sum(result) == 0: return []

            path = os.path.join(package_path, "resource", "T_gripper2camera.npy")
            robot_posx = get_current_posx()[0]
            pos_list = []
            
            for i in range(0, len(result), 3):
                coord = result[i:i+3]
                if sum(coord) == 0: continue
                td = self.transform_to_base(coord, path, robot_posx)
                if td[2]:
                    td[2] = max(td[2] + DEPTH_OFFSET, MIN_DEPTH)
                    pos_list.append(list(td[:3]) + robot_posx[3:])
            return pos_list
        return []

    def init_robot(self):
        movej(JHOME_POS, vel=VELOCITY, acc=ACC)
        mwait()

    def vision_action(self, target_pos):
        """특정 위치에서 자세히 보는 동작"""
        # 나사 위치보다 약간 높은 곳(Z + 100mm)으로 이동하여 카메라로 관찰
        view_pos = list(target_pos)
        view_pos[2] += 100.0  
        
        movel(view_pos, vel=VELOCITY, acc=ACC)
        mwait()
        time.sleep(2.0) # 카메라로 자세히 보는 시간 대기
        self.get_logger().info("시각 검사 완료.")

    def torque_action(self, target_pos):
        """드라이버로 조이는 동작"""
        safe_pos = list(target_pos)
        safe_pos[2] += 150.0

        # 1. 안전 높이 이동
        movel(safe_pos, vel=VELOCITY, acc=ACC)
        mwait()
        
        # 2. 나사 위치로 접근 (터치/조임)
        movel(target_pos, vel=VELOCITY, acc=ACC)
        mwait()
        time.sleep(1.5) # 여기서 실제 엔드이펙터(드라이버) 회전 동작을 추가할 수 있습니다.
        
        # 3. 안전 높이로 복귀
        movel(safe_pos, vel=VELOCITY, acc=ACC)
        mwait()
        self.get_logger().info("토크 검사(조임) 완료.")

    def robot_control(self):
        self.get_logger().info("음성 명령을 기다립니다... (대상을 말해주세요)")
        
        get_keyword_future = self.get_keyword_client.call_async(self.get_keyword_request)
        rclpy.spin_until_future_complete(self, get_keyword_future)
        
        result = get_keyword_future.result()

        if not result or not result.success or not result.message:
            self.get_logger().warn("음성 인식 실패 또는 빈 메시지. 다시 시도합니다.")
            return False

        # 예: "vision_check torque_check pos1 pos3"
        received_msg = result.message.strip().lower()
        self.get_logger().info(f"LLM 해석 결과: '{received_msg}'")
        
        # --- 1. 전체 조사(all_check) 처리 ---
        if "all_check" in received_msg:
            self.get_logger().info("전체 조사(all_check) 모드 진입. 지정된 높은 좌표로 이동합니다.")
            all_check_pos = [-7.997, 24.168, 48.78, -0.063, 107.244, -7.838]
            movej(all_check_pos, vel=VELOCITY, acc=ACC)
            mwait()
            time.sleep(1.0) # 카메라 안정화를 위한 대기

            # Detection 노드에 동작을 요청하여 DB를 업데이트 하도록 유도
            self.get_logger().info("Detection 노드에 검사 요청 중...")
            _ = self.get_target_list("all") 
            
            # DB가 업데이트 될 수 있도록 약간의 시간 지연 대기
            time.sleep(2.0)
            
            self.get_logger().info("작업 완료 확인. Firebase DB에서 나사 좌표 및 할당 번호를 조회합니다.")
            success = self.fetch_positions_from_db(line_id="3")
            
            if success:
                self.get_logger().info("DB 동기화 완료. 다음 명령을 대기하기 위해 홈으로 복귀합니다.")
            else:
                self.get_logger().warn("DB 동기화에 실패했습니다.")
                
            movej(JHOME_POS, vel=VELOCITY, acc=ACC)
            mwait()
            return True

        # --- 2. 명령(Command)과 위치(Target) 분리 파싱 ---
        words = received_msg.split()
        commands = [w for w in words if "check" in w] # ['vision_check', 'torque_check'] 등
        targets = [w for w in words if "pos" in w or w.isdigit()] # ['pos1', 'pos3'] 등
        
        # 'pos1' 등에서 숫자만 추출 -> ['1', '3']
        target_keys = ["".join(filter(str.isdigit, t)) for t in targets]

        if not target_keys:
            self.get_logger().warn("위치(번호) 정보가 없습니다. 명령을 다시 확인해주세요.")
            return False

        # --- 3. 추출된 번호별로 지정된 명령 수행 ---
        for i, key in enumerate(target_keys):
            if key in self.saved_positions:
                target_pos = self.saved_positions[key]
                
                # 명령이 위치 개수보다 적으면 마지막 명령이나 기본값(vision_check)을 사용
                if i < len(commands):
                    cmd = commands[i]
                else:
                    cmd = commands[-1] if commands else "vision_check"

                # 분기 처리
                if cmd == "vision_check":
                    self.get_logger().info(f"[{key}번 나사] 시각 검사(vision_check)를 수행합니다.")
                    # self.vision_action(target_pos)
                elif cmd == "torque_check":
                    self.get_logger().info(f"[{key}번 나사] 토크 검사(torque_check)를 수행합니다.")
                    # self.torque_action(target_pos)
                else:
                    self.get_logger().warn(f"알 수 없는 명령입니다: {cmd}")
            else:
                self.get_logger().warn(f"DB에 할당된 '{key}'번 나사의 좌표가 없습니다. 'all_check'를 먼저 실행했는지 확인하세요.")

        self.get_logger().info("모든 개별 작업 완료. 홈(JHOME_POS) 위치로 복귀합니다.")
        movej(JHOME_POS, vel=VELOCITY, acc=ACC)
        mwait()
        return True
        

def main(args=None):
    node = RobotController()
    try:
        while rclpy.ok():
            node.robot_control()
            time.sleep(1.0) 
    except KeyboardInterrupt:
        node.get_logger().info("사용자에 의해 노드가 종료됩니다.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()