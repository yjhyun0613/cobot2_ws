import rclpy
from rclpy.node import Node
import threading
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

# 두산 로봇 메시지
from dsr_msgs2.srv import GetExternalTorque

class SimpleTorqueMonitor(Node):
    def __init__(self):
        super().__init__('simple_torque_monitor')
        
        # 외부 토크 서비스 클라이언트 생성
        self.tq_cli = self.create_client(GetExternalTorque, '/dsr01/aux_control/get_external_torque')
        
        # 그래프에 표시할 데이터 버퍼 (최대 100개 저장 = 약 5초 분량)
        self.max_len = 100
        self.torque_data = [deque(maxlen=self.max_len) for _ in range(6)]
        self.time_data = deque(maxlen=self.max_len)
        self.start_time = self.get_clock().now().nanoseconds / 1e9
        
        # 0.05초(20Hz) 주기로 토크 데이터 요청
        self.timer = self.create_timer(0.05, self.timer_callback)

    def timer_callback(self):
        if self.tq_cli.wait_for_service(timeout_sec=0.1):
            req = GetExternalTorque.Request()
            self.tq_cli.call_async(req).add_done_callback(self.tq_cb)

    def tq_cb(self, future):
        try:
            res = future.result()
            # 경과 시간 계산
            current_time = self.get_clock().now().nanoseconds / 1e9 - self.start_time
            
            # 시간 및 6개 관절의 토크 값 저장
            self.time_data.append(current_time)
            for i in range(6):
                self.torque_data[i].append(res.ext_torque[i])
        except Exception as e:
            self.get_logger().error(f"토크 데이터를 가져오는데 실패했습니다: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = SimpleTorqueMonitor()
    
    # 💡 ROS 2 데이터 수신은 백그라운드 스레드에서 실행 (그래프 창이 멈추지 않도록)
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # =========================================================
    # 📊 Matplotlib 실시간 그래프 설정
    # =========================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    lines = []
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    
    # 6개 축(Joint 1~6)에 대한 라인 생성
    for i in range(6):
        line, = ax.plot([], [], color=colors[i], label=f'Joint {i+1}')
        lines.append(line)
    
    # 그래프 UI 세팅
    ax.set_ylim(-30, 30)  # Y축 범위 (토크 값이 이 범위를 넘어가면 수치를 늘려주세요)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('External Torque (Nm)')
    ax.set_title('Real-time External Torque Monitor')
    ax.legend(loc='upper right')
    ax.grid(True)

    # 그래프를 매 프레임 업데이트하는 함수
    def update(frame):
        if not node.time_data:
            return lines
        
        # X축을 최신 시간에 맞춰 스크롤링 (최근 5초 구간만 보여줌)
        current_time = node.time_data[-1]
        ax.set_xlim(max(0, current_time - 5), max(5, current_time))
        
        # 각 관절의 라인 데이터 업데이트
        for i in range(6):
            lines[i].set_data(node.time_data, node.torque_data[i])
        return lines

    # 50ms 마다 그래프 갱신
    ani = animation.FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    
    # 그래프 창 띄우기 (이 창을 닫으면 프로그램이 종료됩니다)
    plt.show()

    # 종료 처리
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()