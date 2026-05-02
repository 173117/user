import os
import time
import threading
import subprocess
import datetime
import config


def play_alarm():
    """播放报警声音"""
    def _play():
        for _ in range(config.ALARM_REPEAT):
            if config.ALARM_SOUND_PATH and os.path.exists(config.ALARM_SOUND_PATH):
                try:
                    import pygame
                    pygame.mixer.init()
                    pygame.mixer.music.load(config.ALARM_SOUND_PATH)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                except Exception:
                    _system_beep()
            else:
                _system_beep()
            time.sleep(0.5)

    t = threading.Thread(target=_play, daemon=True)
    t.start()


def _system_beep():
    """系统蜂鸣声（跨平台）"""
    try:
        import winsound
        for _ in range(5):
            winsound.Beep(1000, 400)
            time.sleep(0.1)
    except ImportError:
        print("\a\a\a")


def show_alert_window(timestamp: str):
    """弹出系统通知窗口"""
    contacts_text = "\n".join(
        [f"  {c['name']}: {c['phone']}" for c in config.EMERGENCY_CONTACTS]
    )
    message = (
        f"⚠️  检测到老人跌倒！\n\n"
        f"时间：{timestamp}\n\n"
        f"紧急联系人：\n{contacts_text}\n\n"
        f"请立即拨打以上电话确认安全！"
    )
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(
            0,
            message,
            "【AI智护银龄】跌倒预警",
            0x00001030,  # MB_ICONWARNING | MB_SYSTEMMODAL
        )
    except Exception:
        print(f"\n{'='*50}")
        print("【紧急警报】检测到老人跌倒！")
        print(f"时间：{timestamp}")
        print(f"请联系：{contacts_text}")
        print('='*50)


def auto_dial(contact: dict):
    """
    自动拨打电话。
    根据 config.CALL_PROVIDER 选择服务商：
      "aliyun" -> 阿里云语音通知API（国内推荐）
      "twilio" -> Twilio API（国际备选）
      "none"   -> 仅打印提示，不实际拨号
    """
    phone = contact["phone"]
    name = contact["name"]
    provider = config.CALL_PROVIDER.lower()
    print(f"[报警] 正在拨打 {name}: {phone}（服务商: {provider}）")

    if provider == "aliyun":
        _dial_aliyun(phone, name)
    elif provider == "twilio":
        _dial_twilio(phone, name)
    else:
        print(f"[报警] CALL_PROVIDER='none'，未拨号。")
        print(f"       请在 config.py 中配置 CALL_PROVIDER='aliyun' 或 'twilio'")


def _dial_aliyun(phone: str, name: str):
    """
    阿里云语音通知 API。
    依赖：pip install alibabacloud-dyvmsapi20170525
    开通：https://dysms.console.aliyun.com/ -> 语音服务
    """
    try:
        from alibabacloud_dyvmsapi20170525.client import Client
        from alibabacloud_dyvmsapi20170525 import models as vms_models
        from alibabacloud_tea_openapi import models as open_api_models

        cfg = open_api_models.Config(
            access_key_id=config.ALIYUN_ACCESS_KEY_ID,
            access_key_secret=config.ALIYUN_ACCESS_KEY_SECRET,
        )
        cfg.endpoint = "dyvmsapi.aliyuncs.com"
        client = Client(cfg)

        request = vms_models.SingleCallByTtsRequest(
            called_show_number=config.ALIYUN_CALLED_SHOW_NUMBER,
            called_number=phone,
            tts_code=config.ALIYUN_TTS_CODE,
        )
        response = client.single_call_by_tts(request)
        if response.body.code == "OK":
            print(f"[报警] 阿里云语音拨出成功 -> {name} ({phone})")
        else:
            print(f"[报警] 阿里云拨号失败: {response.body.code} - {response.body.message}")

    except ImportError:
        print("[报警] 缺少阿里云SDK，请执行：pip install alibabacloud-dyvmsapi20170525")
    except Exception as e:
        print(f"[报警] 阿里云拨号异常: {e}")


def _dial_twilio(phone: str, name: str):
    """
    Twilio 语音通话 API。
    依赖：pip install twilio
    注册：https://www.twilio.com/
    """
    try:
        from twilio.rest import Client
        from twilio.twiml.voice_response import VoiceResponse

        client = Client(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)

        # 构造 TwiML：电话接通后播报文字
        response = VoiceResponse()
        response.say(config.TWILIO_VOICE_MESSAGE, language="zh-CN", voice="alice")
        response.pause(length=1)
        response.say(config.TWILIO_VOICE_MESSAGE, language="zh-CN", voice="alice")

        call = client.calls.create(
            twiml=str(response),
            to=phone,
            from_=config.TWILIO_FROM_NUMBER,
        )
        print(f"[报警] Twilio 拨出成功 -> {name} ({phone})，Call SID: {call.sid}")

    except ImportError:
        print("[报警] 缺少Twilio SDK，请执行：pip install twilio")
    except Exception as e:
        print(f"[报警] Twilio拨号异常: {e}")


def log_fall_event(timestamp: str, screenshot_path: str = None):
    """记录跌倒事件到日志文件"""
    log_path = os.path.join(config.SCREENSHOT_DIR, "fall_log.txt")
    os.makedirs(config.SCREENSHOT_DIR, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        line = f"[{timestamp}] 检测到跌倒事件"
        if screenshot_path:
            line += f"，截图：{screenshot_path}"
        f.write(line + "\n")
    print(f"[日志] 事件已记录：{log_path}")


def trigger_alert(screenshot_path: str = None):
    """触发完整报警流程"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[⚠️  报警] 跌倒事件触发 - {timestamp}")

    # 1. 记录日志
    log_fall_event(timestamp, screenshot_path)

    # 2. 播放报警声
    play_alarm()

    # 3. 弹出警告窗口（另开线程，不阻塞检测）
    t_window = threading.Thread(
        target=show_alert_window, args=(timestamp,), daemon=True
    )
    t_window.start()

    # 4. 自动拨号（可选）
    if config.AUTO_CALL_ENABLED:
        for contact in config.EMERGENCY_CONTACTS:
            auto_dial(contact)
            time.sleep(2)
