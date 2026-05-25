import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT_DIR / ".env"


def load_env_file(env_path: Path = ENV_PATH):
    if not env_path.exists():
        return

    with env_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if key and key not in os.environ:
                os.environ[key] = value


load_env_file()


def get_env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


COMPANY_ID = get_env("COMPANY_ID", "company_joingo_001")
COMPANY_NAME = get_env("COMPANY_NAME", "JoinGo")

SITE_ID = get_env("SITE_ID", "site_joingo_lab_001")
SITE_NAME = get_env("SITE_NAME", "JoinGo Lab")

ROBOT_ID = get_env("ROBOT_ID", "dsr01")
ROBOT_NAME = get_env("ROBOT_NAME", "Doosan M0609")

CAMERA_ID = get_env("CAMERA_ID", "realsense_d435i")
CAMERA_NAME = get_env("CAMERA_NAME", "Intel RealSense D435i")

WORKSTATION_ID = get_env("WORKSTATION_ID", "workstation_01")
WORKSTATION_NAME = get_env("WORKSTATION_NAME", "작업대 1")

LINE_ID = get_env("LINE_ID", "line_01")
LINE_NAME = get_env("LINE_NAME", "검사 라인 1")

BASE_FRAME = get_env("BASE_FRAME", "base_link")
CAMERA_FRAME = get_env("CAMERA_FRAME", "camera_link")

FIREBASE_DATABASE_URL = get_env("FIREBASE_DATABASE_URL")
FIREBASE_STORAGE_BUCKET = get_env("FIREBASE_STORAGE_BUCKET")
FIREBASE_SERVICE_ACCOUNT = get_env("FIREBASE_SERVICE_ACCOUNT")
EXTERNAL_API_KEY = get_env("EXTERNAL_API_KEY")


def validate_firebase_settings():
    missing = []

    if not FIREBASE_DATABASE_URL:
        missing.append("FIREBASE_DATABASE_URL")

    if not FIREBASE_STORAGE_BUCKET:
        missing.append("FIREBASE_STORAGE_BUCKET")

    if not FIREBASE_SERVICE_ACCOUNT:
        missing.append("FIREBASE_SERVICE_ACCOUNT")

    if FIREBASE_SERVICE_ACCOUNT and not Path(FIREBASE_SERVICE_ACCOUNT).exists():
        missing.append(f"FIREBASE_SERVICE_ACCOUNT file not found: {FIREBASE_SERVICE_ACCOUNT}")

    if missing:
        raise RuntimeError("Firebase setting error: " + ", ".join(missing))
