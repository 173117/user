# main.py

def check_license_compliance():
    """
    运行时许可证合规性检查提示。
    此提示仅为告知，不影响程序功能。
    """
    print("=" * 60)
    print("⚠️  注意：本软件受非商业许可证保护")
    print("   严禁用于任何商业目的。")
    print("   详情请参阅项目根目录下的 LICENSE 文件。")
    print("   商业授权请联系: [你的联系邮箱]")
    print("=" * 60)

# 在程序启动时调用
if __name__ == "__main__":
    check_license_compliance()
    # ... 你的主程序逻辑
# 这是一个运行在笔记本上的 Python 脚本 (main.py)
import cv2
import mediapipe as mp
import requests # 用来给小程序发信号

# 初始化姿态识别模型
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# 开启笔记本外接摄像头 (0代表默认摄像头)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    # 1. 把画面转成 AI 能看懂的格式
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    
    if results.pose_landmarks:
        # 2. 获取关键点坐标 (例如：鼻子是 landmark[0], 左膝是 landmark[25])
        # 注意：y坐标是 0-1 之间，0是顶端，1是底端
        nose_y = results.pose_landmarks.landmark[0].y
        knee_y = results.pose_landmarks.landmark[25].y
        
        # 3. 判断逻辑：如果鼻子比膝盖还低，说明人倒了
        if nose_y > knee_y: 
            print("⚠️ 检测到摔倒！")
            # 4. 发送信号给小程序后端
            send_alert_to_server() # pyright: ignore[reportUndefinedVariable]
            
    cv2.imshow('AI Monitor', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'): break

def send_alert_to_server():
    # 调用微信小程序的云函数接口，或者你的服务器接口
    requests.post("你的服务器地址/api/alert", json={"status": "fall_detected"})