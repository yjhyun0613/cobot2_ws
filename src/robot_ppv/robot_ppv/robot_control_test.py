import os
import time
import sys
from scipy.spatial.transform import Rotation
import numpy as np
import rclpy
from rclpy.node import Node
import DR_init

from od_msg.srv import SrvDepthPosition
from ament_index_python.packages import get_package_share_directory
from robot_control.onrobot import RG

package_path = get_package_share_directory("robot_ppv")

# for single robot
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
VELOCITY, ACC = 60, 60
JHOME_POS = [0, 0, 90, 0, 90, 0]
GRIPPER_NAME = "rg2"
TOOLCHARGER_IP = "192.168.1.1"
TOOLCHARGER_PORT = "502"
DEPTH_OFFSET = -10.0
MIN_DEPTH = 2.0

# [수정] 테스트할 대상 객체 이름 (YOLO 클래스명과 동일하게 설정)
TARGET_OBJECT = "hammer" 

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

########### Gripper Setup ############
# 그리퍼 제어는 하지 않지만, 물리적 연결/초기화 에러 방지를 위해 객체 생성은 유지합니다.
gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)

########### Robot Controller ############
class RobotController(Node):
    def __init__(self):
        super().__init__("robot_test_controller")
        self.init_robot()

        # YOLO(오브젝트 디텍션) 서비스 클라이언트
        self.get_position_client = self.create_client(
            SrvDepthPosition, "/get_3d_position"
        )
        while not self.get_position_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().info("Waiting for get_depth_position service...")

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
        self.get_logger().info(f"YOLO에 '{TARGET_OBJECT}' 위치 탐색을 요청합니다...")
        
        target_pos = self.get_target_pos(TARGET_OBJECT)
        
        if target_pos is None:
            self.get_logger().warn(f"[{TARGET_OBJECT}]의 좌표를 찾지 못했습니다. 재시도합니다.")
        else:
            self.get_logger().info(f"[{TARGET_OBJECT}] 좌표 획득 성공: {target_pos}")
            # 목표 위치로 이동하여 한 번 찍고 오는 동작 실행
            self.touch_target(target_pos)

    def get_target_pos(self, target):
        new_request = SrvDepthPosition.Request()
        new_request.target = target

        self.get_logger().info(f"★[테스트 스팟] 도커(YOLO)로 전송할 타겟: '{new_request.target}'")
       
        get_position_future = self.get_position_client.call_async(new_request)
        rclpy.spin_until_future_complete(self, get_position_future)

        if get_position_future.result():
            result = get_position_future.result().depth_position.tolist()
            self.get_logger().info(f"Received depth position: {result}")
            if sum(result) == 0:
                self.get_logger().warn("No target position detected by YOLO.")
                return None

            gripper2cam_path = os.path.join(
                package_path, "resource", "T_gripper2camera.npy"
            )
            robot_posx = get_current_posx()[0]
            td_coord = self.transform_to_base(result, gripper2cam_path, robot_posx)

            if td_coord[2] and sum(td_coord) != 0:
                td_coord[2] += DEPTH_OFFSET
                td_coord[2] = max(td_coord[2], MIN_DEPTH)

            target_pos = list(td_coord[:3]) + robot_posx[3:]
            return target_pos
        
        return None

    def init_robot(self):
        # 초기 홈 위치로 이동
        movej(JHOME_POS, vel=VELOCITY, acc=ACC)
        mwait()

    def touch_target(self, target_pos):
        """
        목표 위치(나사 정중앙) 위로 이동 -> 하강하여 콕 찍기 -> 안전 높이로 상승 -> 홈 복귀
        """
        # 목표 위치에서 Z축으로 150mm 위인 안전한 접근 위치 계산
        target_pos_up = trans(target_pos, [0, 0, 150, 0, 0, 0]).tolist()

        # 1. 목표 위치의 상단으로 접근
        self.get_logger().info("1. 목표 위치 상단으로 이동 중...")
        movel(target_pos_up, vel=VELOCITY, acc=ACC)
        mwait()

        # 2. 나사 정중앙 위치로 하강 (콕 찍기)
        self.get_logger().info("2. 나사 정중앙으로 하강 (터치 테스트)...")
        movel(target_pos, vel=VELOCITY, acc=ACC)
        mwait()
        
        # 정확도를 육안으로 확인할 수 있도록 1.5초 대기
        gripper.close_gripper()
        time.sleep(1.5)
        gripper.open_gripper()

        # 3. 다시 안전 위치로 수직 상승
        self.get_logger().info("3. 안전 높이로 수직 상승...")
        movel(target_pos_up, vel=VELOCITY, acc=ACC)
        mwait() 
        
        # 4. 홈 자세로 복귀
        self.get_logger().info("4. 홈(JHOME_POS) 위치로 복귀 완료.")
        movej(JHOME_POS, vel=VELOCITY, acc=ACC)
        mwait() 

        while gripper.get_status()[0]:
            time.sleep(0.5)
        mwait()


def main(args=None):
    node = RobotController()
    try:
        while rclpy.ok():
            node.robot_control()
            # 연속적인 테스트를 방지하고 싶다면 아래 대기 시간을 늘려주세요. (현재 3초 대기 후 재탐색)
            time.sleep(3.0) 
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()
        node.destroy_node()


if __name__ == "__main__":
    main()