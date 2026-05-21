import open3d as o3d
import numpy as np
import plotly.graph_objects as go
import os

def merge_pcd_with_icp(source_path, target_path, voxel_size=0.005):
    print("---------------------------------------")
    print(f"[1/4] 파일 불러오는 중...")
    print(f"      - Source (움직여서 맞출 데이터): {source_path}")
    print(f"      - Target (고정된 기준 데이터): {target_path}")
    
    # 1. PCD 파일 로드
    source = o3d.io.read_point_cloud(source_path)
    target = o3d.io.read_point_cloud(target_path)

    if not source.has_points() or not target.has_points():
        print("❌ 에러: 파일을 불러오지 못했거나 점 데이터가 없습니다. 경로를 확인해주세요.")
        return

    # 2. ICP 정합을 위한 초기 설정
    # 두 파일이 이미 base_link 기준으로 대략 맞물려 있으므로 초기 변환 행렬은 단위 행렬(Identity Matrix)을 사용합니다.
    current_transformation = np.eye(4)
    
    # ICP가 일치하는 점을 찾을 최대 거리 임계값 (예: 3cm 이내의 점들끼리 매칭 시도)
    threshold = 0.20

    print(f"[2/4] ICP 알고리즘 기반 정밀 정합(Fine Registration) 시작...")
    
    # Point-to-Plane 방식을 위해 법선 벡터(Normal) 계산
    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # ICP 실행 (반복적으로 가까운 점들을 매칭하며 오차 행렬을 계산)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, current_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    print("✅ 정합 완료!")
    print("--- 계산된 미세 오차 보정 행렬 (Transformation Matrix) ---")
    print(reg_p2p.transformation)
    print("---------------------------------------------------------")

    # 3. Source 데이터를 계산된 행렬로 변환(움직임)한 뒤 Target 데이터와 합치기
    print(f"[3/4] 두 데이터 포인트 병합 및 중복 점 제거 중...")
    source.transform(reg_p2p.transformation)
    
    # 두 포인트 클라우드 합치기
    merged_pcd = target + source
    
    # 너무 겹쳐서 빽빽해진 점들을 5mm 간격으로 다운샘플링하여 정리
    merged_pcd = merged_pcd.voxel_down_sample(voxel_size=voxel_size)

    # 4. 결과 저장
    output_pcd_filename = "merged_final.pcd"
    o3d.io.write_point_cloud(output_pcd_filename, merged_pcd)
    print(f"✅ 최종 원본 데이터 저장 완료: {output_pcd_filename}")

    # 5. 시각화용 HTML 파일 생성
    print(f"[4/4] 웹 브라우저 확인용 HTML 파일 생성 중...")
    pts = np.asarray(merged_pcd.points)
    colors = np.asarray(merged_pcd.colors) * 255

    if len(pts) > 0:
        html_colors = [f'rgb({int(r)}, {int(g)}, {int(b)})' for r, g, b in colors]
        fig = go.Figure(data=[go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode='markers',
            marker=dict(size=1.5, color=html_colors) # 점 크기를 살짝 줄여 정밀하게 표현
        )])
        fig.update_layout(
            scene=dict(
                aspectmode='data',
                xaxis_title='X (base_link 기준)',
                yaxis_title='Y (base_link 기준)',
                zaxis_title='Z (base_link 기준)'
            ),
            title="ICP 후처리 병합 결과 (Merged 3D Map)"
        )
        
        output_html_filename = "merged_final.html"
        fig.write_html(output_html_filename)
        print(f"✅ 최종 웹 뷰어 파일 저장 완료: {output_html_filename}")
        print("---------------------------------------")

if __name__ == "__main__":
    # 병합할 두 파일 이름 지정 (같은 폴더에 있다고 가정)
    source_file = "/home/yoon/cobot2_ws/src/cobot2/testfile/capture_2.pcd"  # 움직여서 맞출 대상
    target_file = "/home/yoon/cobot2_ws/src/cobot2/testfile/capture_1.pcd"  # 고정된 기준점
    
    if os.path.exists(source_file) and os.path.exists(target_file):
        merge_pcd_with_icp(source_file, target_file)
    else:
        print(f"❌ 에러: 폴더에 {source_file} 또는 {target_file} 파일이 없습니다.")
        print("먼저 이전 코드를 실행하여 두 개 이상의 캡처 파일을 만들어주세요.")