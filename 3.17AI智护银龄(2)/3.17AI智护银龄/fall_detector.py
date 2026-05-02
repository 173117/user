import cv2
import numpy as np
import time
import os
import datetime
from collections import deque
from typing import Optional, Dict, Tuple, List
import mediapipe as mp


import config
import alert

# MediaPipe Pose 33关键点索引
MP_NOSE           = 0
MP_LEFT_SHOULDER  = 11
MP_RIGHT_SHOULDER = 12
MP_LEFT_HIP       = 23
MP_RIGHT_HIP      = 24
MP_LEFT_KNEE      = 25
MP_RIGHT_KNEE     = 26
MP_LEFT_ANKLE     = 27
MP_RIGHT_ANKLE    = 28


class FallDetector:
    def __init__(self):
        print("[初始化] 加载 MediaPipe Pose 模型...")
        self._mp_pose    = mp.solutions.pose
        self._mp_draw    = mp.solutions.drawing_utils
        self._mp_styles  = mp.solutions.drawing_styles
        self.pose = self._mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=config.CONFIDENCE_THRESHOLD,
            min_tracking_confidence=0.5,
        )
        print("[初始化] 模型加载完成")

        # 每个追踪ID的髋部Y坐标历史（用于计算下落速度）
        self.hip_history: Dict[int, deque] = {}

        # 每个追踪ID的连续跌倒帧计数
        self.fall_frame_count: Dict[int, int] = {}

        # 上次报警时间（全局冷却）
        self.last_alert_time = 0

        self._stop_flag = False
        self._warmup_remaining = config.WARMUP_FRAMES

        os.makedirs(config.SCREENSHOT_DIR, exist_ok=True)

    def reset(self):
        """切换视频/摄像头时重置所有追踪状态和预热计数器。"""
        self.hip_history.clear()
        self.fall_frame_count.clear()
        self._warmup_remaining = config.WARMUP_FRAMES

    def _get_lm(self, landmarks, idx: int, w: int, h: int):
        """获取 MediaPipe 关键点像素坐标，可见度低时返回 None。"""
        lm = landmarks[idx]
        if lm.visibility < 0.3:
            return None
        return float(lm.x * w), float(lm.y * h)

    def _calc_body_angle(self, landmarks, w: int, h: int) -> Optional[float]:
        """计算躯干与垂直方向的夹角（度）：肩部中点→髋部中点连线。"""
        ls = self._get_lm(landmarks, MP_LEFT_SHOULDER,  w, h)
        rs = self._get_lm(landmarks, MP_RIGHT_SHOULDER, w, h)
        lh = self._get_lm(landmarks, MP_LEFT_HIP,       w, h)
        rh = self._get_lm(landmarks, MP_RIGHT_HIP,      w, h)
        if not all([ls, rs, lh, rh]):
            return None
        shoulder_mid = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
        hip_mid      = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
        dx = hip_mid[0] - shoulder_mid[0]
        dy = hip_mid[1] - shoulder_mid[1]
        if dy == 0:
            return 90.0
        return float(abs(np.degrees(np.arctan2(dx, dy))))

    def _get_hip_y(self, landmarks, h: int) -> Optional[float]:
        """获取髋部中心 Y 坐标（像素）。"""
        lh = landmarks[MP_LEFT_HIP]
        rh = landmarks[MP_RIGHT_HIP]
        if lh.visibility >= 0.3 and rh.visibility >= 0.3:
            return (lh.y + rh.y) / 2 * h
        if lh.visibility >= 0.3:
            return lh.y * h
        if rh.visibility >= 0.3:
            return rh.y * h
        return None
 

    def _get_bbox(self, landmarks, w: int, h: int) -> Optional[List[float]]:
        """从所有可见关键点计算包围盒 [x1,y1,x2,y2]。"""
        xs = [lm.x * w for lm in landmarks if lm.visibility > 0.2]
        ys = [lm.y * h for lm in landmarks if lm.visibility > 0.2]
        if not xs or not ys:
            return None
        pad = 10
        return [max(0, min(xs) - pad), max(0, min(ys) - pad),
                min(w, max(xs) + pad), min(h, max(ys) + pad)]

    def _calc_hip_drop_speed(-3self, track_id: int, hip_y: float) -> float:
        """计算髋部Y坐标下降速度（像素/帧）"""
        if track_id not in self.hip_history:
            self.hip_history[track_id] = deque(maxlen=10)

        history = self.hip_history[track_id]
        speed = 0.0
        if len(history) >= 3:
            speed = hip_y - history[-3]kd;, 
        history.append(hip_y)
        return speed      

    def _is_fallen(
        self,
        track_id: int,
        landmarks,
        bbox: Optional[List[float]],
        w: int,
        h: int,
    ) -> Tuple[bool, dict]:
        """
        综合判断是否跌倒，返回 (is_fallen, debug_info)。
        4个条件，满足 FALL_SCORE_THRESHOLD 条即确认：
          1. 躯干倾斜角 > TILT_ANGLE_THRESHOLD
          2. 躯干垂直压缩比 < TORSO_HEIGHT_RATIO_THRESHOLD
          3. 头髋垂直距离 < HEAD_HIP_DIFF_THRESHOLD
          4. 髋部快速下降 > HIP_DROP_SPEED_THRESHOLD
        """
        debug = {}
        score = 0

        # --- 条件1：躯干倾斜角 ---
        angle = self._calc_body_angle(landmarks, w, h)
        debug["angle"] = round(angle, 1) if angle is not None else None
        if angle is not None and angle > config.TILT_ANGLE_THRESHOLD:
            score += 1

        # --- 条件2：躯干垂直压缩比
        # 站立时 (hip_y - shoulder_y) / h 约 0.25~0.40
        # 跌倒时躯干平啦，该比趋近 0 ---
        ls = self._get_lm(landmarks, MP_LEFT_SHOULDER,  w, h)
        rs = self._get_lm(landmarks, MP_RIGHT_SHOULDER, w, h)
        lh = self._get_lm(landmarks, MP_LEFT_HIP,       w, h)
        rh = self._get_lm(landmarks, MP_RIGHT_HIP,      w, h)
        if all([ls, rs, lh, rh]):
            shoulder_y = (ls[1] + rs[1]) / 2
            hip_y_mid  = (lh[1] + rh[1]) / 2
            torso_ratio = (hip_y_mid - shoulder_y) / h
            debug["torso"] = round(torso_ratio, 3)
            if torso_ratio < config.TORSO_HEIGHT_RATIO_THRESHOLD:
                score += 1
        else:
            debug["torso"] = None
        # --- 条件3：头髋垂直距离
        # 正常 (hip_y - nose_y)/h > 0.35；跌倒时趋近 0 或负值 ---
        hip_y = self._get_hip_y(landmarks, h)
        nose_lm = landmarks[MP_NOSE]
        if hip_y is not None and nose_lm.visibility >= 0.3:
            nose_y = nose_lm.y * h
            head_hip = (hip_y - nose_y) / h
            debug["head_hip"] = round(head_hip, 3)
            if head_hip < config.HEAD_HIP_DIFF_THRESHOLD:
                score += 1
        else:
            debug["head_hip"] = None

        # --- 条件4：髋部快速下降 ---
        drop_speed = 0.0
        if hip_y is not None:
            drop_speed = self._calc_hip_drop_speed(track_id, hip_y)
        debug["drop"] = round(drop_speed, 1)
        if drop_speed > config.HIP_DROP_SPEED_THRESHOLD:
            score += 1

        debug["score"] = score
        return score >= config.FALL_SCORE_THRESHOLD, debug

    def _confirm_fall(self, track_id: int, is_fallen_frame: bool) -> bool:
        """
        连续 N 帧确认跌倒。
        确认前：非跌倒帧每帧 -2（快速退激避免短暂误报）。
        确认后：非跌倒帧每帧 -1（慢慢恢复，防止跌倒持续时短暂误判）。
        """
        if track_id not in self.fall_frame_count:
            self.fall_frame_count[track_id] = 0

        cnt = self.fall_frame_count[track_id]
        already_confirmed = cnt >= config.FALL_CONFIRM_FRAMES

        if is_fallen_frame:
            # 最高超过 3x 上限，避免过長恢复期
            self.fall_frame_count[track_id] = min(cnt + 1, config.FALL_CONFIRM_FRAMES * 3)
        else:
            step = 1 if already_confirmed else 2
            self.fall_frame_count[track_id] = max(0, cnt - step)

        return self.fall_frame_count[track_id] >= config.FALL_CONFIRM_FRAMES

    def _trigger_alert(self, frame: np.ndarray):
        """触发报警，保存截图"""
        now = time.time()
        if now - self.last_alert_time < config.ALERT_COOLDOWN_SECONDS:
            return
        self.last_alert_time = now

        screenshot_path = None
        if config.SAVE_FALL_SCREENSHOT:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = os.path.join(config.SCREENSHOT_DIR, f"fall_{ts}.jpg")
            cv2.imwrite(screenshot_path, frame)
            print(f"[截图] 已保存：{screenshot_path}")

        alert.trigger_alert(screenshot_path)

    def _draw_info(
        self,
        frame: np.ndarray,
        bbox: Optional[List[float]],
        is_fallen: bool,
        confirmed: bool,
        debug: dict,
        track_id: int,
    ):
        """在画面上绘制检测框和状态信息"""
        if bbox is None:
            return
        x1, y1, x2, y2 = [int(v) for v in bbox]

        if confirmed:
            color = (0, 0, 255)
            label = f"[FALL] ID:{track_id}"
        elif is_fallen:
            color = (0, 140, 255)
            label = f"[WARN] ID:{track_id}"
        else:
            color = (0, 255, 0)
            label = f"[OK] ID:{track_id}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame, label, (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
        )

        if config.SHOW_DEBUG_INFO:
            ang   = debug.get("angle")
            torso = debug.get("torso")
            hh    = debug.get("head_hip")
            line1 = (
                f"Ang:{ang:.1f}" if ang is not None else "Ang:N/A"
            ) + "  " + (
                f"Torso:{torso:.2f}" if torso is not None else "Torso:N/A"
            )
            line2 = (
                f"HH:{hh:.2f}" if hh is not None else "HH:N/A"
            ) + f"  Drop:{debug.get('drop', 0):.1f}  Sc:{debug.get('score', 0)}"
            cv2.putText(frame, line1, (x1, y2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            cv2.putText(frame, line2, (x1, y2 + 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, bool]:
        """处理单帧，返回 (标注后的帧, 是否确认跌倒)。供 Web 和独立模式共用。"""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        output_frame = frame.copy()
        track_id = 0  # MediaPipe 单人检测，固定 ID=0

        # --- 预热阶段：先让 MediaPipe 稳定关键点，不做跌倒判断 ---
        if self._warmup_remaining > 0:
            self._warmup_remaining -= 1
            if results.pose_landmarks and config.SHOW_SKELETON:
                self._mp_draw.draw_landmarks(
                    output_frame,
                    results.pose_landmarks,
                    self._mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self._mp_styles.get_default_pose_landmarks_style(),
                )
            remaining = self._warmup_remaining
            cv2.rectangle(output_frame, (0, 0), (w, 50), (0, 0, 0), -1)
            cv2.putText(
                output_frame,
                f"Initializing... ({remaining} frames)",
                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 200), 2,
            )
            return output_frame, False

        # --- 正式检测阶段 ---
        any_confirmed_fall = False

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            if config.SHOW_SKELETON:
                self._mp_draw.draw_landmarks(
                    output_frame,
                    results.pose_landmarks,
                    self._mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self._mp_styles.get_default_pose_landmarks_style(),
                )

            bbox = self._get_bbox(landmarks, w, h)
            is_fallen_frame, debug = self._is_fallen(track_id, landmarks, bbox, w, h)
            confirmed = self._confirm_fall(track_id, is_fallen_frame)
            self._draw_info(output_frame, bbox, is_fallen_frame, confirmed, debug, track_id)

            if confirmed:
                any_confirmed_fall = True
        else:
            self.fall_frame_count[track_id] = 0

        if any_confirmed_fall:
            overlay = output_frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.15, output_frame, 0.85, 0, output_frame)
            cv2.putText(
                output_frame, "!! FALL DETECTED !!",
                (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3
            )
            self._trigger_alert(output_frame)

        return output_frame, any_confirmed_fall

    def stop(self):
        """停止 run() 主循环"""
        self._stop_flag = True

    def run(self):
        """独立运行模式（直接命令行启动，弹出窗口）"""
        self._stop_flag = False
        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, config.FPS_TARGET)

        if not cap.isOpened():
            print(f"[错误] 无法打开摄像头（索引 {config.CAMERA_INDEX}）")
            return

        print("[运行] 摄像头已启动，按 Q 退出")
        fps_counter = 0
        fps_start = time.time()
        fps_display = 0

        while not self._stop_flag:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            output_frame, _ = self.process_frame(frame)

            fps_counter += 1
            if time.time() - fps_start >= 1.0:
                fps_display = fps_counter
                fps_counter = 0
                fps_start = time.time()

            cv2.putText(
                output_frame, f"FPS:{fps_display} AI Fall Monitor",
                (10, output_frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
            )
            cv2.imshow("AI Fall Monitor", output_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


    def __del__(self):
        if hasattr(self, "pose"):
            self.pose.close()


if __name__ == "__main__":
    detector = FallDetector()
    detector.run()
