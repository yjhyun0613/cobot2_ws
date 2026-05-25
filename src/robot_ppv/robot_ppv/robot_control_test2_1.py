##########################
"""
1. 단일 객체 인식 됬을 때 그 좌표로 이동
2 복수 의 객체 인식 됬을 때 순서대로 좌표 이동 후 노드 종료
2.1 음성 추가
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

from od_msg.srv import SrvDepthPosition
from std_srvs.srv import Trigger
from ament_index_python.packages import get_package_share_directory
from robot_control.onrobot import RG

package_path = get_package_share_directory("robot_ppv")

# 기존 DSR 설정 유지
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
    from DSR_ROBOT2 import movej, movel, get_current_posx, mwait, trans
except ImportError as e:
    sys.exit()

gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)

class RobotController(Node):
    def __init__(self):
        super().__init__("robot_integrated_controller")
        self.init_robot()

        self.get_position_client = self.create_client(SrvDepthPosition, "/get_3d_position")
  
        while not self.get_position_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().info("Waiting for get_depth_position service...")

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
        gripper2cam = np.load(gripper2cam_path)
        coord = np.append(np.array(camera_coords), 1)
        x, y, z, rx, ry, rz = robot_pos
        base2gripper = self.get_robot_pose_matrix(x, y, z, rx, ry, rz)
        base2cam = base2gripper @ gripper2cam
        td_coord = np.dot(base2cam, coord)
        return td_coord[:3]

    def robot_control(self):
        self.get_logger().info("음성 명령을 기다립니다... (대상을 말해주세요)")
        
        get_keyword_future = self.get_keyword_client.call_async(self.get_keyword_request)
        rclpy.spin_until_future_complete(self, get_keyword_future)
        
        check_all_pos = [-7.997, 24.168, 48.78, -0.063, 107.244, -7.838]
        movel(check_all_pos, vel=VELOCITY, acc=ACC)
        mwait()
        
        result = get_keyword_future.result()

        if not result or not result.success or not result.message:
            self.get_logger().warn("음성 인식 실패 또는 빈 메시지. 다시 시도합니다.")
            return False

        target_obj = result.message.strip().replace("'", "").replace('"', "")
        self.get_logger().info(f"검출 대상: '{target_obj}'")
        
        target_pos_list = self.get_target_list(target_obj)
        
        if not target_pos_list:
            self.get_logger().warn(f"[{target_obj}]의 좌표를 찾지 못했습니다.")
            return False
        else:
            self.get_logger().info(f"총 {len(target_pos_list)}개의 객체 발견. 작업을 시작합니다.")
            for i, pos in enumerate(target_pos_list):
                self.get_logger().info(f"--- {i+1}번째 객체 이동 중 ---")
                self.touch_target(pos)
            
        self.get_logger().info("모든 작업 완료. 프로그램을 종료합니다.")
        # 순회가 끝나면 홈 위치로 복귀
        self.get_logger().info("모든 작업 완료. 홈(JHOME_POS) 위치로 복귀합니다.")
        movej(JHOME_POS, vel=VELOCITY, acc=ACC)
        mwait()
        return True # 작업 완료 신호

    def get_target_list(self, target):
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
        # 이제 break 없이 계속 루프를 돌며 음성 명령을 기다립니다.
        while rclpy.ok():
            node.robot_control()
            time.sleep(1.0) # 연속 동작 방지 및 CPU 리소스 최적화
    except KeyboardInterrupt:
        node.get_logger().info("사용자에 의해 노드가 종료됩니다.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()