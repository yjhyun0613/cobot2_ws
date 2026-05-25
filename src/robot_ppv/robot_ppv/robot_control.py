import os
import time
import sys
from scipy.spatial.transform import Rotation
import numpy as np
import rclpy
from rclpy.node import Node
import DR_init

from od_msg.srv import SrvDepthPosition
from std_srvs.srv import Trigger
from ament_index_python.packages import get_package_share_directory
from robot_control.onrobot import RG

package_path = get_package_share_directory("robot_ppv")

# for single robot
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
VELOCITY, ACC = 60, 60
BUCKET_POS = [445.5, -242.6, 174.4, 156.4, 180.0, -112.5]
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
    from DSR_ROBOT2 import movej, movel, get_current_posx, mwait, trans
except ImportError as e:
    print(f"Error importing DSR_ROBOT2: {e}")
    sys.exit()

########### Gripper Setup. Do not modify this area ############

gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)


########### Robot Controller ############


class RobotController(Node):
    def __init__(self):
        super().__init__("pick_and_place")
        self.init_robot()

        self.get_position_client = self.create_client(
            SrvDepthPosition, "/get_3d_position"
        )
        while not self.get_position_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().info("Waiting for get_depth_position service...")

        # self.get_position_request = SrvDepthPosition.Request()

        self.get_keyword_client = self.create_client(Trigger, "/get_keyword")
        while not self.get_keyword_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().info("Waiting for get_keyword service...")
        self.get_keyword_request = Trigger.Request()

    def get_robot_pose_matrix(self, x, y, z, rx, ry, rz):
        R = Rotation.from_euler("ZYZ", [rx, ry, rz], degrees=True).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T

    def transform_to_base(self, camera_coords, gripper2cam_path, robot_pos):
        """
        Converts 3D coordinates from the camera coordinate system
        to the robot's base coordinate system.
        """
        gripper2cam = np.load(gripper2cam_path)
        coord = np.append(np.array(camera_coords), 1)  # Homogeneous coordinate

        x, y, z, rx, ry, rz = robot_pos
        base2gripper = self.get_robot_pose_matrix(x, y, z, rx, ry, rz)

        # 좌표 변환 (그리퍼 → 베이스)
        base2cam = base2gripper @ gripper2cam
        td_coord = np.dot(base2cam, coord)

        return td_coord[:3]

    def robot_control(self):
        self.get_logger().info("call get_keyword service")
        self.get_logger().info("say 'Hello Rokey' and speak what you want to pick up")
        
        # 음성 인식 서비스 요청
        get_keyword_future = self.get_keyword_client.call_async(self.get_keyword_request)
        rclpy.spin_until_future_complete(self, get_keyword_future)
        
        get_keyword_result = get_keyword_future.result()
        
        # ===== [여기다가 테스트용 로그 심기] =====
        self.get_logger().info(f"★[스팟 1] 음성 노드 수신 원본 데이터: '{get_keyword_result.message}'")
        # =========================================
        
        # 확실하게 성공했고, 메시지가 비어있지 않을 때만 진입!
        if get_keyword_result and get_keyword_result.success:
            raw_message = get_keyword_result.message.strip()
            
            # 메시지가 진짜 빈칸이면 다음 루프로 패스
            if not raw_message:
                self.get_logger().warn("받은 음성 메시지가 빈 문자열입니다. 다시 대기합니다.")
                return

            target_list = raw_message.split()
            self.get_logger().info(f"정상 인식된 타겟 리스트: {target_list}")

            for target in target_list:
                # 기호 필터링
                clean_target = target.replace("'", "").replace('"', "").replace("[", "").replace("]", "").strip()

                if not clean_target or clean_target.lower() == "none":
                    continue

                # 정상 단어일 때만 도커 YOLO 호출
                target_pos = self.get_target_pos(clean_target)
                if target_pos is None:
                    self.get_logger().warn(f"[{clean_target}]의 좌표를 찾지 못했습니다.")
                else:
                    self.get_logger().info(f"[{clean_target}] 좌표 획득 성공: {target_pos}")
                    self.pick_and_place_target(target_pos)
                    self.init_robot()
        else:
            self.get_logger().warn("음성 인식 노드로부터 응답을 받지 못했거나 실패했습니다.")  

    def get_target_pos(self, target):
        # ======= [★핵심 변경: 함수 안에서 완전히 독립된 새 Request 객체 생성] =======
        new_request = SrvDepthPosition.Request()
        new_request.target = target # 확실하게 문자열로 고정하여 대입
        # =========================================================================

        # ===== [YOLO 서비스 호출 직전에 로그 심기] =====
        self.get_logger().info("call depth position service with object_detection node")
        self.get_logger().info(f"★[스팟 2] 도커(YOLO)로 전송할 최종 변수값: '{new_request.target}'")
        # ===============================================
       
        # 멤버 변수(self.) 대신, 스레드 간섭을 받지 않는 new_request를 인자로 쏏니다!
        get_position_future = self.get_position_client.call_async(new_request)
        rclpy.spin_until_future_complete(self, get_position_future)

        if get_position_future.result():
            result = get_position_future.result().depth_position.tolist()
            self.get_logger().info(f"Received depth position: {result}")
            if sum(result) == 0:
                print("No target position")
                return None

            gripper2cam_path = os.path.join(
                package_path, "resource", "T_gripper2camera.npy"
            )
            robot_posx = get_current_posx()[0]
            td_coord = self.transform_to_base(result, gripper2cam_path, robot_posx)

            if td_coord[2] and sum(td_coord) != 0:
                td_coord[2] += DEPTH_OFFSET  # DEPTH_OFFSET
                td_coord[2] = max(td_coord[2], MIN_DEPTH)  # MIN_DEPTH: float = 2.0

            target_pos = list(td_coord[:3]) + robot_posx[3:]
        return target_pos

    def init_robot(self):
        JReady = [0, 0, 90, 0, 90, 0]
        movej(JReady, vel=VELOCITY, acc=ACC)
        gripper.open_gripper()
        mwait()

    def pick_and_place_target(self, target_pos):
        movel(target_pos, vel=VELOCITY, acc=ACC)
        mwait()
        gripper.close_gripper()
        
        time.sleep(1.5)

        target_pos_up = trans(target_pos, [0, 0, 200, 0, 0, 0]).tolist()

        target_pos_up = trans(target_pos, [0, 0, 200, 0, 0, 0]).tolist()
        movel(target_pos_up, vel=VELOCITY, acc=ACC)
        mwait() 
        
        # 2. 안전한 홈 자세(관절 좌표)로 이동 (movej 사용)
        movej(JHOME_POS, vel=VELOCITY, acc=ACC) # movej로 변경
        mwait() 
        
        # 놓을 위치로 이동
        movel(BUCKET_POS, vel=VELOCITY, acc=ACC)
        mwait()
        gripper.open_gripper()

        while gripper.get_status()[0]:
            time.sleep(0.5)
        mwait()

        ### 재권추가시작
        # 안전하게 들어올리는 동작
        # pick_up_pos = target_pos[0:2] + [target_pos[2]+50] + target_pos[3:]
        # movel(pick_up_pos, vel=VELOCITY, acc=ACC)
        # mwait()
        
        # 2. 안전한 홈 자세(관절 좌표)로 이동 (movej 사용)
        # # movej(JHOME_POS, vel=VELOCITY, acc=ACC) # movej로 변경
        # # mwait() 
        
        # # 놓을 위치로 이동
        # movej(BUCKET_POS, vel=VELOCITY, acc=ACC)
        # mwait()
        
        ### 재권 추가 끝

        # gripper.open_gripper()
        # while gripper.get_status()[0]:
        #     time.sleep(0.5)


def main(args=None):
    node = RobotController()
    try:
        while rclpy.ok():
            node.robot_control()
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()
        node.destroy_node()


if __name__ == "__main__":
    main()
