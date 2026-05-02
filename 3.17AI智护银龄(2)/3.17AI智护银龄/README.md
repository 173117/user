# AI智护银龄 —— 居家老人跌倒监测与应急处理系统

## 功能简介

通过摄像头实时监测老人动作，利用 AI 骨骼关键点分析自动识别跌倒行为，并立即触发声音报警和弹窗提示，通知家属或社区。

## 技术方案

| 组件 | 技术 | 说明 |
|------|------|------|
| 动作识别 | **YOLOv8-pose** | 检测17个人体骨骼关键点，无需自训练 |
| 跌倒判断 | 规则引擎 | 躯干倾角 + 宽高比 + 髋部下降速度，满足2项即判定 |
| 报警 | winsound / pygame | 系统蜂鸣或自定义音频 |
| 通知 | Windows弹窗 + 日志 | 显示紧急联系人电话，可选自动拨号 |

## 目录结构

```
3.17AI智护银龄/
├── fall_detector.py    # 主程序（运行此文件）
├── alert.py            # 报警模块
├── config.py           # 配置文件（修改联系人、阈值等）
├── requirements.txt    # 依赖包
├── fall_records/       # 自动创建，保存跌倒截图和日志
└── yolov8n-pose.pt     # 首次运行自动下载
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 修改联系人（重要）

编辑 `config.py`，填写家属/社区电话：

```python
EMERGENCY_CONTACTS = [
    {"name": "家属-张三", "phone": "13800138000"},
    {"name": "社区医院",  "phone": "12345"},
]
```

### 3. 运行

```bash
python fall_detector.py
```

首次运行会自动下载 YOLOv8-pose 模型（约6MB），需要网络连接。

### 4. 退出

在监测窗口按 **Q** 键退出。

## 跌倒检测原理

系统通过以下3个指标综合判断，**满足任意2项**确认为跌倒：

1. **躯干倾斜角** > 30°（肩部中点→髋部中点连线偏离垂直方向）
2. **检测框宽高比** > 1.2（人体变水平，躺倒姿态）
3. **髋部快速下降** > 15像素/帧（人体突然向下运动）

连续 8 帧确认后触发报警，60秒内不重复报警（可在 `config.py` 调整）。

## 配置说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `CAMERA_INDEX` | 0 | 摄像头编号 |
| `TILT_ANGLE_THRESHOLD` | 50 | 倾斜角阈值（度）|
| `ASPECT_RATIO_THRESHOLD` | 1.2 | 宽高比阈值 |
| `HIP_DROP_SPEED_THRESHOLD` | 15 | 下降速度阈值（像素/帧）|
| `FALL_CONFIRM_FRAMES` | 8 | 确认帧数（越大越不易误报）|
| `ALERT_COOLDOWN_SECONDS` | 60 | 报警冷却时间（秒）|
| `AUTO_CALL_ENABLED` | False | 是否启用自动拨号 |
| `SAVE_FALL_SCREENSHOT` | True | 是否保存跌倒截图 |

## 硬件要求

- 普通 USB 摄像头或笔记本内置摄像头
- CPU 即可运行（Intel i5 以上推荐），有 NVIDIA GPU 更流畅
- 内存 ≥ 4GB

## 报警记录

跌倒事件自动记录到 `fall_records/fall_log.txt`，截图保存为 `fall_records/fall_YYYYMMDD_HHMMSS.jpg`。
