import socket

# 두산 로봇 컨트롤러 IP 설정 (실제 사용 중인 로봇의 IP로 변경하세요)
ROBOT_IP = "192.168.1.100" 
PORT = 7171

def send_scan_code():
    # 전송할 DRL(Doosan Robot Language) 스크립트 작성
    drl_script = """
    # 1. 속도 및 가속도 설정 (사용자 설정값: 30)
    set_velj(30)
    set_accj(30)

    # 2. 초기 준비 자세 설정 (Joint Space)
    # 중심(X=0, Y=0)에서 특이점을 피하기 위해 조인트 각도로 직접 진입합니다.
    # 아래는 M0609가 높이 500mm 부근에서 팔을 오므리고 정면을 보는 예시 각도입니다.
    # *주의: 실제 로봇을 티칭 펜던트로 원하는 시작 자세로 움직인 후, 
    # 그때의 J1~J6 각도를 확인하여 아래 값을 반드시 업데이트해 주세요.
    init_q = posj(0.0, -15.0, -100.0, 0.0, -65.0, 0.0) 
    movej(init_q)

    # 3. 스캔 파라미터
    step_angle = 15.0     # 5번 조인트(고개) 상승 각도
    max_tilt = 90.0       # 최대 상승 각도 (완전 위쪽)
    current_tilt = 0.0    # 현재 상승 진행도
    direction = 1         # 1: 정방향(+360), -1: 역방향(-360)

    # 4. 반구 스캔 루프 시작
    while current_tilt <= max_tilt:
        # 현재 관절 각도 읽어오기
        curr_q = get_current_posj()
        
        # [수평 스캔] 1번 조인트 360도 회전
        target_q = curr_q
        if direction == 1:
            target_q[0] = target_q[0] + 360.0
        else:
            target_q[0] = target_q[0] - 360.0
            
        movej(target_q)
        
        # 천장(90도)까지 다 봤다면 고개 들기를 생략하고 루프 종료
        if current_tilt >= max_tilt:
            break
            
        # [고개 들기] 5번 조인트 15도 상승
        # M0609의 기구학 설정에 따라 J5의 (-) 방향이 고개를 드는 방향인지 
        # (+) 방향인지 확인이 필요합니다. (아래는 -로 꺾는 것을 상승으로 가정)
        curr_q = get_current_posj()
        next_tilt_q = curr_q
        next_tilt_q[4] = next_tilt_q[4] - step_angle 
        movej(next_tilt_q)
        
        # 다음 층을 위해 회전 방향 반전 (지그재그) 및 각도 누적
        direction = direction * -1
        current_tilt = current_tilt + step_angle

    # 5. 스캔 완료 후 꼬임 방지를 위해 초기 위치로 복귀
    movej(init_q)
    """

    # TCP/IP 소켓을 통해 로봇 컨트롤러로 스크립트 전송
    try:
        print(f"로봇({ROBOT_IP})에 연결 중...")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((ROBOT_IP, PORT))
            # DRL 문자열을 인코딩하여 전송
            s.sendall(drl_script.encode('utf-8'))
            print("DRL 스크립트 전송이 완료되었습니다. 로봇이 스캔 동작을 시작합니다.")
    except ConnectionRefusedError:
        print("연결 실패: 로봇의 IP 주소가 맞는지, 로봇이 켜져 있는지 확인해 주세요.")
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    send_scan_code()