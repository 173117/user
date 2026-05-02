import os
import time
import datetime
import numpy as np
import cv2
from typing import Optional, Tuple, Dict, List
from collections import deque

import config
import alert

# COCO 17 关键点索引
KP_NOSE           = 0
KP_LEFT_SHOULDER  = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_HIP       = 11
KP_RIGHT_HIP      = 12


class FallDetectorYOLO:
    def __init__(self):
        from ultralytics import YOLO
        print("[YOLO] 加载 YOLOv8-pose 模型...")
        self.model = YOLO(config.MODEL_PATH)
        print("[YOLO] 模型加载完成")

        self.hip_history: Dict[int, deque] = {}
        self.fall_frame_count: Dict[int, int] = {}
        self.last_alert_time = 0
        self._warmup_remaining = config.WARMUP_FRAMES
        self._stop_flag = False

        os.makedirs(config.SCREENSHOT_DIR, exist_ok=True)

    def reset(self):
        """切换视频/摄像头时重置所有追踪状态和预热计数器。"""
        self.hip_history.clear()
        self.fall_frame_count.clear()
        self._warmup_remaining = config.WARMUP_FRAMES

    # ── 关键点工具 ─────────────────────────────────────────────
    def _get_kp(self, kps: np.ndarray, idx: int):
        """从 YOLO kps 数组 (N,3) 取关键点，置信度低时返回 None。"""
        if kps is None or len(kps) <= idx:
            return None
        x, y, conf = kps[idx]
        if conf < 0.3:
            return None
        return float(x), float(y)

    def _calc_body_angle(self, kps: np.ndarray, h: int) -> Optional[float]:
        ls = self._get_kp(kps, KP_LEFT_SHOULDER)
        rs = self._get_kp(kps, KP_RIGHT_SHOULDER)
        lh = self._get_kp(kps, KP_LEFT_HIP)
        rh = self._get_kp(kps, KP_RIGHT_HIP)
        if not all([ls, rs, lh, rh]):
            return None
        shoulder_mid = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
        hip_mid      = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
        dx = hip_mid[0] - shoulder_mid[0]
        dy = hip_mid[1] - shoulder_mid[1]
        if dy == 0:
            return 90.0
        return float(abs(np.degrees(np.arctan2(dx, dy))))

    def _get_hip_y(self, kps: np.ndarray) -> Optional[float]:
        lh = self._get_kp(kps, KP_LEFT_HIP)
        rh = self._get_kp(kps, KP_RIGHT_HIP)
        if lh and rh:
            return (lh[1] + rh[1]) / 2
        return (lh or rh or (None,))[1] if (lh or rh) else None

    def _calc_hip_drop_speed(self, track_id: int, hip_y: float) -> float:
        if track_id not in self.hip_history:
            self.hip_history[track_id] = deque(maxlen=15)
        history = self.hip_history[track_id]
        speed = 0.0
        if len(history) >= 3:
            speed = hip_y - history[-3]
        history.append(hip_y)
        return speed

    # ── 跌倒判断（4 条件，同 MediaPipe 版）──────────────────────
    def _is_fallen(
        self, track_id: int, kps: np.ndarray,
        bbox: Optional[List[float]], w: int, h: int,
    ) -> Tuple[bool, dict]:
        debug = {}
        score = 0

        # 条件1：躯干倾斜角
        angle = self._calc_body_angle(kps, h)
        debug["angle"] = round(angle, 1) if angle is not None else None
        if angle is not None and angle > config.TILT_ANGLE_THRESHOLD:
            score += 1

        # 条件2：躯干垂直压缩比
        ls = self._get_kp(kps, KP_LEFT_SHOULDER)
        rs = self._get_kp(kps, KP_RIGHT_SHOULDER)
        lh = self._get_kp(kps, KP_LEFT_HIP)
        rh = self._get_kp(kps, KP_RIGHT_HIP)
        if all([ls, rs, lh, rh]):
            shoulder_y  = (ls[1] + rs[1]) / 2
            hip_y_mid   = (lh[1] + rh[1]) / 2
            torso_ratio = (hip_y_mid - shoulder_y) / h
            debug["torso"] = round(torso_ratio, 3)
            if torso_ratio < config.TORSO_HEIGHT_RATIO_THRESHOLD:
                score += 1
        else:
            debug["torso"] = None

        # 条件3：头部接近髋部
        hip_y = self._get_hip_y(kps)
        nose  = self._get_kp(kps, KP_NOSE)
        if nose and hip_y is not None:
            head_hip = (hip_y - nose[1]) / h
            debug["head_hip"] = round(head_hip, 3)
            if head_hip < config.HEAD_HIP_DIFF_THRESHOLD:
                score += 1
        else:
            debug["head_hip"] = None

        # 条件4：髋部快速下降
        drop_speed = 0.0
        if hip_y is not None:
            drop_speed = self._calc_hi;p_drop_speed(track_id, hip_y) # pyright: ignore[reportUndefinedVariable] # pyright: ignore[reportUndefinedVariable]
        debug["drop"]  = round(drop_speed, 1)
        debug["score"] = score
        if drop_speed > config.HIP_DROP_SPEED_THRESHOLD:
            score += 1

        return score >= config.FALL_SCORE_THRESHOLD, debug

    def _confirm_fall(self, track_id: int, is_fallen_frame: bool) -> bool:
        if track_id not in self.fall_frame_count:
            self.fall_frame_count[track_id] = 0
        cnt = self.fall_frame_count[track_id]
        already_confirmed = cnt >= config.FALL_CONFIRM_FRAMES
        if is_fallen_frame:
            self.fall_frame_count[track_id] = min(cnt + 1, config.FALL_CONFIRM_FRAMES * 3)
        else:
            self.fall_frame_count[track_id] = max(0, cnt - (1 if already_confirmed else 2))
        return self.fall_frame_count[track_id] >= config.FALL_CONFIRM_FRAMES

    def _trigger_alert(self, frame: np.ndarray):
        now = time.time()
        if now - self.last_alert_time < config.ALERT_COOLDOWN_SECONDS:
            return
        self.last_alert_time = now
        screenshot_path = None
        if config.SAVE_FALL_SCREENSHOT:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = os.path.join(config.SCREENSHOT_DIR, f"fall_yolo_{ts}.jpg")
            cv2.imwrite(screenshot_path, frame)
        alert.trigger_alert(screenshot_path)

    def _draw_info(self, frame, bbox, is_fallen, confirmed, debug, track_id):
        if bbox is None:
            return
        x1, y1, x2, y2 = [int(v) for v in bbox]
        if confirmed:
            color, label = (0, 0, 255),   f"[FALL-Y] ID:{track_id}"
        elif is_fallen:
            color, label = (0, 140, 255), f"[WARN-Y] ID:{track_id}"
        else:
            color, label = (0, 200, 50),  f"[OK-Y]   ID:{track_id}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if config.SHOW_DEBUG_INFO:
            ang   = debug.get("angle")
            torso = debug.get("torso")
            hh    = debug.get("head_hip")
            line1 = (f"Ang:{ang:.1f}" if ang is not None else "Ang:N/A") + "  " + \
                    (f"Torso:{torso:.2f}" if torso is not None else "Torso:N/A")
            line2 = (f"HH:{hh:.2f}" if hh is not None else "HH:N/A") + \
                    f"  Drop:{debug.get('drop', 0):.1f}  Sc:{debug.get('score', 0)}"
            cv2.putText(frame, line1, (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            cv2.putText(frame, line2, (x1, y2 + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # ── 主处理入口 ──────────────────────────────────────────────
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, bool]:
        h, w = frame.shape[:2]
        output_frame = frame.copy()

        # 预热阶段
        if self._warmup_remaining > 0:
            self._warmup_remaining -= 1
            cv2.rectangle(output_frame, (0, 0), (w, 50), (0, 0, 0), -1)
            cv2.putText(
                output_frame,
                f"[YOLO] Initializing... ({self._warmup_remaining} frames)",
                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 100), 2,
            )
            return output_frame, False

        results = self.model.track(
            frame, persist=True,
            conf=config.CONFIDENCE_THRESHOLD,
            classes=[0], verbose=False,
        )
        any_confirmed_fall = False

        if results and results[0].keypoints is not None:
            result   = results[0]
            boxes    = result.boxes
            kps_data = result.keypoints.data

            if config.SHOW_SKELETON:
                output_frame = result.plot(boxes=False, labels=False, conf=False)

            if boxes is not None:
                for i, (box, kps) in enumerate(zip(boxes, kps_data)):
                    bbox     = box.xyxy[0].cpu().numpy().tolist()
                    track_id = int(box.id[0]) if box.id is not None else i
                    kps_np   = kps.cpu().numpy()

                    is_fallen_frame, debug = self._is_fallen(track_id, kps_np, bbox, w, h)
                    confirmed = self._confirm_fall(track_id, is_fallen_frame)
                    self._draw_info(output_frame, bbox, is_fallen_frame, confirmed, debug, track_id)

                    if confirmed:
                        any_confirmed_fall = True

        if any_confirmed_fall:
            overlay = output_frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.15, output_frame, 0.85, 0, output_frame)
            cv2.putText(output_frame, "!! FALL DETECTED !!",
                        (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            self._trigger_alert(output_frame)

        return output_frame, any_confirmed_fall

    def stop(self):
        self._stop_flag = True