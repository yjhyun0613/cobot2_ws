##########################
#2. 복수 의 객체 인식 됬을 때 순서대로 좌표 이동 후 노드 종료
##########################

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
DEPTH_OFFSET = -39.0
MIN_DEPTH = 2.0

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
gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)

########### Robot Controller ############
class RobotController(Node):
    def __init__(self):
        super().__init__("robot_test_controller")
        self.init_robot()

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
        gripper2cam = np.load(gripper2cam_path)
        coord = np.append(np.array(camera_coords), 1)

        x, y, z, rx, ry, rz = robot_pos
        base2gripper = self.get_robot_pose_matrix(x, y, z, rx, ry, rz)

        base2cam = base2gripper @ gripper2cam
        td_coord = np.dot(base2cam, coord)

        return td_coord[:3]

    def robot_control(self):
        self.get_logger().info(f"YOLO에 '{TARGET_OBJECT}' 위치 탐색을 요청합니다...")
        
        target_pos_list = self.get_target_list(TARGET_OBJECT)
        
        if not target_pos_list:
            self.get_logger().warn(f"[{TARGET_OBJECT}]의 좌표를 찾지 못했습니다. 재시도합니다.")
            return False  # <--- 실패 시 False 반환 (루프 계속 돌게 됨)
        else:
            self.get_logger().info(f"[{TARGET_OBJECT}] 총 {len(target_pos_list)}개의 객체 좌표 획득 성공!")
            
            # 다중 객체 좌표 순회 (콕 찍기)
            for i, target_pos in enumerate(target_pos_list):
                self.get_logger().info(f"--- {i+1}번째 객체 작업 시작 ---")
                self.touch_target(target_pos)
            
            # 순회가 끝나면 홈 위치로 복귀
            self.get_logger().info("모든 작업 완료. 홈(JHOME_POS) 위치로 복귀합니다.")
            movej(JHOME_POS, vel=VELOCITY, acc=ACC)
            mwait()
            return True  # <--- 성공적으로 끝났음을 True로 반환

    def get_target_list(self, target):
        new_request = SrvDepthPosition.Request()
        new_request.target = target
       
        get_position_future = self.get_position_client.call_async(new_request)
        rclpy.spin_until_future_complete(self, get_position_future)

        if get_position_future.result():
            # 1차원 리스트 결과 획득
            result = get_position_future.result().depth_position.tolist()
            
            if not result or sum(result) == 0:
                self.get_logger().warn("No target position detected by YOLO.")
                return []

            gripper2cam_path = os.path.join(
                package_path, "resource", "T_gripper2camera.npy"
            )
            robot_posx = get_current_posx()[0]
            target_pos_list = []
            
            # 1차원 리스트를 3개(x, y, z)씩 묶어서 해석 및 좌표 변환
            for i in range(0, len(result), 3):
                coord_3d = result[i:i+3]
                
                if sum(coord_3d) == 0:
                    continue
                    
                td_coord = self.transform_to_base(coord_3d, gripper2cam_path, robot_posx)

                if td_coord[2] and sum(td_coord) != 0:
                    td_coord[2] += DEPTH_OFFSET
                    td_coord[2] = max(td_coord[2], MIN_DEPTH)

                target_pos = list(td_coord[:3]) + robot_posx[3:]
                target_pos_list.append(target_pos)
                
            return target_pos_list
        
        return []

    def init_robot(self):
        movej(JHOME_POS, vel=VELOCITY, acc=ACC)
        mwait()

    def touch_target(self, target_pos):
        """
        목표물 상단 150mm 위치 접근 -> 정중앙 터치 -> 상단 안전 복귀
        """
        # 1. trans() 대신 파이썬 리스트 조작으로 Z축(높이)에만 150mm를 정확하게 추가합니다.
        # 이렇게 하면 Base 좌표계 기준으로 무조건 하늘 위로 올라갑니다.
        target_pos_up = list(target_pos)
        target_pos_up[2] += 150.0

        self.get_logger().info(f"1. 상단 안전 높이({target_pos_up[2]:.1f}mm)로 이동 중...")
        movel(target_pos_up, vel=VELOCITY, acc=ACC)
        mwait()
        time.sleep(0.5)  # [추가] 동작이 둥글게 깎이는 것(블렌딩)을 방지하는 확실한 정지 대기

        self.get_logger().info(f"2. 타겟 정중앙({target_pos[2]:.1f}mm)으로 하강 (터치)...")
        movel(target_pos, vel=VELOCITY, acc=ACC)
        mwait()
        
        time.sleep(1.5)  # 터치 유지

        self.get_logger().info("3. 안전 높이로 수직 상승 완료.")
        movel(target_pos_up, vel=VELOCITY, acc=ACC)
        mwait()
        time.sleep(0.5)  # [추가] 수직 상승을 완전히 마친 후 다음 객체로 이동하기 위한 대기
        

def main(args=None):
    node = RobotController()
    try:
        while rclpy.ok():
            is_completed = node.robot_control()
            
            # 반환값이 True이면 (작업을 완수했으면) 무한 루프 탈출
            if is_completed:
                node.get_logger().info("✅ 계획된 모든 타겟 작업을 마쳤습니다. 노드를 종료합니다.")
                break 
                
            # 타겟을 못 찾아서 False가 반환되었을 때만 3초 대기 후 다시 시도
            time.sleep(3.0) 
            
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()
        node.destroy_node()


if __name__ == "__main__":
    main()