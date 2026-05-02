# ===================== 系统配置 =====================

# 摄像头设置
CAMERA_INDEX = 0          # 摄像头编号，0=默认摄像头
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS_TARGET = 30

# MediaPipe Pose 设置
CONFIDENCE_THRESHOLD = 0.5        # 检测/追踪置信度阈值（0~1）

# YOLOv8-pose 模型路径（仅 YOLO 模式使用，首次运行自动下载 ~6MB）
MODEL_PATH = "yolov8n-pose.pt"

# ===================== 跌倒检测参数 =====================
# 【条件1】身体倾斜角度阈值（度）：肩髋连线与垂直轴夹角超过此值
TILT_ANGLE_THRESHOLD = 60

# 【条件2】躯干垂直压缩比：(髋部Y - 肩部Y) / 帧高 < 此值时视为人体水平
# 站立时约 0.25~0.40；跌倒时趋近 0；建议范围 0.10~0.15
TORSO_HEIGHT_RATIO_THRESHOLD = 0.12

# 【条件3】头髋垂直距离比：(髋部Y - 鼻子Y) / 帧高 < 此值时视为头部接近地面
# 正常站立时约 0.35~0.55（头在髋上方）；跌倒时趋近 0 或负值
HEAD_HIP_DIFF_THRESHOLD = 0.08

# 【条件4】髋部Y坐标下降速度阈值（像素/3帧）：快速下降表示正在跌倒
HIP_DROP_SPEED_THRESHOLD = 15

# 满足以上几个条件才判定为跌倒（推荐 2，可调高减少误报）
FALL_SCORE_THRESHOLD = 2

# 确认跌倒所需的连续触发帧数
FALL_CONFIRM_FRAMES = 8

# 启动时跳过检测的预热帧数（帧）：让 MediaPipe 先稳定再开始判断
WARMUP_FRAMES = 20

# 跌倒后多少秒内不重复报警
ALERT_COOLDOWN_SECONDS = 60

# ===================== 报警设置 =====================
# 报警声音文件路径（留空则使用系统蜂鸣声）
ALARM_SOUND_PATH = "60"

# 报警重复次数
ALARM_REPEAT = 3

# ===================== 联系人设置 =====================
# 家属/社区联系电话（用于弹窗提示和自动拨号）
EMERGENCY_CONTACTS = [
    {"name": "家属-张三", "phone": "13800138000"},
    {"name": "社区医院", "phone": "12345"},
]

# ===================== 自动拨打电话配置 =====================
# 选择拨号服务商："aliyun" | "twilio" | "none"
CALL_PROVIDER = "none"

# --- 阿里云语音通知（国内推荐）---
# 开通地址：https://dysms.console.aliyun.com/
# 需要开通「语音服务」，创建语音通知模板后填写以下信息
ALIYUN_ACCESS_KEY_ID = "your_access_key_id"
ALIYUN_ACCESS_KEY_SECRET = "your_access_key_secret"
ALIYUN_CALLED_SHOW_NUMBER = "your_bought_number"  # 购买的阿里云号码（主叫）
# 语音通知TTS模板CODE（在阿里云控制台申请，模板内容如："检测到老人跌倒，请立即确认安全"）
ALIYUN_TTS_CODE = "TTS_xxxxxxxxx"

# --- Twilio（国际备选）---
# 注册地址：https://www.twilio.com/
TWILIO_ACCOUNT_SID = "ACxxxxxxxxxxxxxxxx"
TWILIO_AUTH_TOKEN = "your_auth_token"
TWILIO_FROM_NUMBER = "+1xxxxxxxxxx"  # Twilio购买的号码
# 电话接通后播报的内容（TwiML语音）
TWILIO_VOICE_MESSAGE = "警告！检测到老人跌倒，请立即确认安全。"

# 是否启用自动拨号
AUTO_CALL_ENABLED = False

# ===================== 日志与录像 =====================
# 是否保存跌倒事件截图
SAVE_FALL_SCREENSHOT = True
SCREENSHOT_DIR = "fall_records"

# 是否显示骨架可视化
SHOW_SKELETON = True

# 是否显示调试信息（角度、速度等）
SHOW_DEBUG_INFO = True
