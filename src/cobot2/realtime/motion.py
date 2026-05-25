#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
explore_environment_joint_sequence.py

음성 명령으로 워크스페이스 탐색을 시작하면,
사용자가 지정한 5개 joint 시점을 순차적으로 이동하면서 YOLO 탐지를 수행하고 결과를 JSON으로 저장한다.

탐색 시점:
1번: [0.0, -0.0, 90.039, -90.0, 90.0, 0.0]
2번: [-0.006, -0.019, 89.985, -90.003, -0.003, -0.0]
3번: [-0.034, -0.019, 89.984, 90.003, 90.005, 0.0]
4번: [-0.034, -0.018, 89.984, -0.004, 90.005, 0.0]
5번: [-0.034, -0.022, 89.983, -0.004, -90.004, -0.0]

결과 저장:
~/cobot_ws/explore_logs/environment_explore_YYYYMMDD_HHMMSS.json
~/cobot_ws/explore_logs/latest_environment_result.json
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime

import rclpy
from dotenv import load_dotenv
import pyaudio
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import DR_init
from ultralytics import YOLO

from voice_processing.MicController import MicController, MicConfig
from voice_processing.wakeup_word import WakeupWord
from voice_processing.stt import STT
from ament_index_python.packages import get_package_share_directory


# ============================================================
# Doosan Robot Settings
# ============================================================

ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"

ROBOT_TOOL = "Tool Weight"
ROBOT_TCP = "GripperDA_v1"

HOME_J = [0, 0, 90, 0, 90, 0]

MOVE_VEL = 20
MOVE_ACC = 10
CAMERA_STABILIZE_SEC = 1.0

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

rclpy.init()
node = rclpy.create_node("explore_environment_node", namespace=ROBOT_ID)
DR_init.__dsr__node = node

try:
    from DSR_ROBOT2 import movej, mwait

    try:
        from DSR_ROBOT2 import set_tool, set_tcp, get_tool, get_tcp
        HAS_TOOL_TCP = True
    except Exception:
        HAS_TOOL_TCP = False

    try:
        from DSR_ROBOT2 import release_force, release_compliance_ctrl
        HAS_RELEASE = True
    except Exception:
        HAS_RELEASE = False

except ImportError as e:
    print(f"Error importing DSR_ROBOT2: {e}")
    sys.exit(1)


# ============================================================
# 탐색 시점 Joint 좌표
# ============================================================

VIEWPOINTS = [
    {
        "id": 1,
        "name": "viewpoint_1",
        "joints": [0.0, -0.0, 90.039, -90.0, 90.0, 0.0],
    },
    {
        "id": 2,
        "name": "viewpoint_2",
        "joints": [-0.006, -0.019, 89.985, -90.003, -0.003, -0.0],
    },
    {
        "id": 3,
        "name": "viewpoint_3",
        "joints": [-0.034, -0.019, 89.984, 90.003, 90.005, 0.0],
    },
    {
        "id": 4,
        "name": "viewpoint_4",
        "joints": [-0.034, -0.018, 89.984, -0.004, 90.005, 0.0],
    },
    {
        "id": 5,
        "name": "viewpoint_5",
        "joints": [-0.034, -0.022, 89.983, -0.004, -90.004, -0.0],
    },
]


# ============================================================
# Vision / Save Settings
# ============================================================

COLOR_TOPIC = "/camera/camera/color/image_raw"
YOLO_IMGSZ = 960
YOLO_CONF = 0.25

RESULT_SAVE_DIR = Path.home() / "cobot_ws" / "explore_logs"

WAKEUP_PRINT_NAME = "Hello Rokey / Hey Jarvis"


# ============================================================
# Utility
# ============================================================

def initialize_robot_setting():
    """Tool/TCP를 매번 명시적으로 세팅."""
    if not HAS_TOOL_TCP:
        print("[INIT] set_tool/set_tcp unavailable. Skip Tool/TCP setup.")
        return

    if ROBOT_TOOL:
        set_tool(ROBOT_TOOL)

    if ROBOT_TCP:
        set_tcp(ROBOT_TCP)

    print("[INIT] current tool:", get_tool())
    print("[INIT] current tcp :", get_tcp())


def release_modes():
    if not HAS_RELEASE:
        return

    try:
        release_force()
    except Exception:
        pass

    try:
        release_compliance_ctrl()
    except Exception:
        pass


def find_model_path() -> str:
    candidates = []

    try:
        package_path = get_package_share_directory("robot_control")
        candidates.append(Path(package_path) / "resource" / "hyupdong2_yolo11x_realtest_corrected_best.pt")
    except Exception:
        pass

    candidates.extend([
        Path.home() / "cobot_ws/src/cobot2_ws/robot_control/resource/hyupdong2_yolo11x_realtest_corrected_best.pt",
        Path.home() / "cobot_ws/src/robot_control/resource/hyupdong2_yolo11x_realtest_corrected_best.pt",
        Path.cwd() / "hyupdong2_yolo11x_realtest_corrected_best.pt",
    ])

    for p in candidates:
        if p.exists():
            return str(p)

    raise FileNotFoundError(
        "YOLO model not found. "
        "hyupdong2_yolo11x_realtest_corrected_best.pt를 robot_control/resource에 넣으세요."
    )


def is_explore_command(command_text: str) -> bool:
    command_text = command_text or ""
    keywords = ["탐색", "환경", "워크스페이스", "작업환경", "찾아", "스캔"]
    return any(k in command_text for k in keywords)


# ============================================================
# Main Class
# ============================================================

class EnvironmentExplorer:
    def __init__(self):
        self.node = node

        # ========================================================
        # Voice setup
        # ========================================================
        package_path = get_package_share_directory("voice_processing")
        env_path = os.path.join(package_path, "resource", ".env")
        load_dotenv(dotenv_path=env_path)

        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.stt = STT(openai_api_key=self.openai_api_key)

        mic_device_index = int(os.getenv("MIC_DEVICE_INDEX", "10"))

        mic_config = MicConfig(
            chunk=12000,
            rate=48000,
            channels=1,
            record_seconds=5,
            fmt=pyaudio.paInt16,
            device_index=mic_device_index,
            buffer_size=24000,
        )

        self.mic_controller = MicController(config=mic_config)
        self.wakeup_word = WakeupWord(mic_config.buffer_size)

        # ========================================================
        # Vision setup
        # ========================================================
        self.bridge = CvBridge()
        self.latest_image = None

        model_path = find_model_path()
        print(f"[YOLO] Loading model: {model_path}")
        self.model = YOLO(model_path)
        print(f"[YOLO] model names: {self.model.names}")

        self.sub = self.node.create_subscription(
            Image,
            COLOR_TOPIC,
            self.image_callback,
            10,
        )

    def image_callback(self, msg):
        self.latest_image = msg

    def movej_sync(self, joints, label):
        print(f"[MOVEJ] {label} | joints={joints}")
        ret = movej(joints, vel=MOVE_VEL, acc=MOVE_ACC)
        print("[MOVEJ] return =", ret)
        mwait()

    def wait_for_image(self, timeout_sec=2.0):
        self.latest_image = None
        timeout = time.time() + timeout_sec

        while self.latest_image is None and time.time() < timeout:
            rclpy.spin_once(self.node, timeout_sec=0.1)

        return self.latest_image is not None

    def check_for_screw(self):
        print("[VISION] Capturing frame for YOLO...")

        if not self.wait_for_image(timeout_sec=2.0):
            print("[VISION] Failed to get image from camera.")
            return {
                "detected": False,
                "count": 0,
                "detections": [],
                "error": "no_image",
            }

        try:
            frame = self.bridge.imgmsg_to_cv2(self.latest_image, desired_encoding="bgr8")

            results = self.model.predict(
                source=frame,
                imgsz=YOLO_IMGSZ,
                conf=YOLO_CONF,
                verbose=False,
            )

            detections = []

            if results and results[0].boxes is not None:
                names = results[0].names

                for box in results[0].boxes:
                    cls_id = int(box.cls.item())
                    raw_label = names.get(cls_id, str(cls_id))
                    conf = float(box.conf.item())
                    xyxy = box.xyxy[0].detach().cpu().numpy().tolist()

                    status = self.normalize_status(raw_label)

                    detections.append({
                        "label": raw_label,
                        "merged_label": "screw",
                        "status": status,
                        "conf": round(conf, 4),
                        "xyxy": [round(float(v), 2) for v in xyxy],
                    })

            if detections:
                print(f"[YOLO] found {len(detections)} object(s)")
                for d in detections:
                    print(
                        f"  - screw raw={d['label']} status={d['status']} "
                        f"conf={d['conf']} box={d['xyxy']}"
                    )
            else:
                print("[YOLO] no object detected")

            return {
                "detected": len(detections) > 0,
                "count": len(detections),
                "detections": detections,
                "error": None,
            }

        except Exception as e:
            print(f"[YOLO] predict failed: {e}")
            return {
                "detected": False,
                "count": 0,
                "detections": [],
                "error": str(e),
            }

    @staticmethod
    def normalize_status(raw_label):
        label = str(raw_label).lower()

        if "good" in label or "ok" in label or "pass" in label or "normal" in label:
            return "GOOD"

        if "ng" in label or "bad" in label or "fail" in label or "defect" in label:
            return "NG"

        return "UNKNOWN"

    def save_result(self, command_text, view_results):
        RESULT_SAVE_DIR.mkdir(parents=True, exist_ok=True)

        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")

        detected_viewpoints = [
            r["viewpoint_id"] for r in view_results if r["detected"]
        ]

        payload = {
            "timestamp": timestamp,
            "created_at": now.isoformat(),
            "command_text": command_text,
            "mode": "joint_sequence_exploration",
            "detected_viewpoints": detected_viewpoints,
            "view_results": view_results,
        }

        result_path = RESULT_SAVE_DIR / f"environment_explore_{timestamp}.json"
        latest_path = RESULT_SAVE_DIR / "latest_environment_result.json"

        result_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        latest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"[SAVE] result saved: {result_path}")
        print(f"[SAVE] latest saved: {latest_path}")

        return result_path

    def run_exploration_once(self, command_text):
        print("[EXPLORE] Starting joint-sequence exploration for screw...")
        print("[EXPLORE] Sequential viewpoints: 1 -> 2 -> 3 -> 4 -> 5")

        view_results = []

        for view in VIEWPOINTS:
            print(f"\n[EXPLORE] Moving to viewpoint {view['id']} ({view['name']})")

            self.movej_sync(view["joints"], f"Viewpoint {view['id']} {view['name']}")
            time.sleep(CAMERA_STABILIZE_SEC)

            detection_result = self.check_for_screw()

            view_result = {
                "viewpoint_id": view["id"],
                "viewpoint_name": view["name"],
                "joints": view["joints"],
                "detected": detection_result["detected"],
                "count": detection_result["count"],
                "detections": detection_result["detections"],
                "error": detection_result.get("error"),
            }

            view_results.append(view_result)

            if detection_result["detected"]:
                print(f">>> Screw detected at viewpoint {view['id']} ({view['name']}) <<<")
            else:
                print(f"No screw at viewpoint {view['id']} ({view['name']})")

        print("\n=== Exploration Complete ===")
        print("Detected viewpoints:", [r["viewpoint_id"] for r in view_results if r["detected"]])

        self.save_result(command_text, view_results)

        print("[EXPLORE] Returning Home...")
        self.movej_sync(HOME_J, "Home")

    def explore(self):
        initialize_robot_setting()
        release_modes()

        print("[INIT] Moving to Home [0, 0, 90, 0, 90, 0]")
        self.movej_sync(HOME_J, "Home")

        print("[MIC] Opening Mic Stream...")

        try:
            self.mic_controller.open_stream()
            self.wakeup_word.set_stream(self.mic_controller.stream)

        except OSError:
            print("[MIC] Failed to open microphone. Check MIC_DEVICE_INDEX or device_index.")
            return

        while rclpy.ok():
            print(f"\n[VOICE] Waiting for Wakeup Word ({WAKEUP_PRINT_NAME})...")

            while rclpy.ok() and not self.wakeup_word.is_wakeup():
                time.sleep(0.1)

            if not rclpy.ok():
                break

            print("[VOICE] Wakeword detected! Command standby mode...")

            command_text = self.stt.speech2text()
            command_text = command_text or ""

            print(f"[VOICE] Command received: {command_text}")

            if is_explore_command(command_text):
                self.run_exploration_once(command_text)
            else:
                print("[VOICE] Unknown command. Try saying '작업환경 탐색해'")


def main():
    explorer = EnvironmentExplorer()

    try:
        explorer.explore()

    except KeyboardInterrupt:
        print("User interrupted.")

    finally:
        try:
            explorer.mic_controller.close_stream()
        except Exception:
            pass

        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()