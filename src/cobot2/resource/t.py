import numpy as np
from scipy.spatial.transform import Rotation as R

# 파일 경로 설정 (윤님의 경로에 맞게 설정)
calib_path = '/home/yoon/YJH/resource/T_gripper2camera.npy'

# 데이터 로드
matrix = np.load(calib_path)

# 과학적 표기법(e-05 등)을 방지하고 보기 편하게 소수점 4자리까지만 출력
np.set_printoptions(suppress=True, precision=4)

print("=== 🛠️ T_gripper2camera 4x4 행렬 원본 ===")
print(matrix)
print("\n" + "="*40 + "\n")

# 직관적으로 볼 수 있게 위치와 회전값 분리
trans = matrix[:3, 3]
rot_mat = matrix[:3, :3]
euler_angles = R.from_matrix(rot_mat).as_euler('xyz', degrees=True)

print("=== 📊 추출된 세부 데이터 ===")
print(f"📍 이동 (Translation X, Y, Z) : {trans}")
print(f"🔄 회전 (Roll, Pitch, Yaw 단위:도) : {euler_angles}")