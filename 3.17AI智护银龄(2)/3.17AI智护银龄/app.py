import os
import time
import threading
import datetime
import webbrowser

import cv2
from flask import Flask, render_template, Response, request, jsonify
from werkzeug.utils import secure_filename

import config
import alert as alert_module
from fall_detector import FallDetector

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ─── Global State ──────────────────────────────────────────────
_detector: FallDetector = None
_frame_lock = threading.Lock()
_current_frame_bytes = None

_status = {
    "source": "idle",
    "is_running": False,
    "fall_count": 0,
    "last_fall_time": None,
    "video_progress": 0,
    "video_frame": 0,
    "video_total": 0,
    "fps": 0,
}
_event_log = []
_worker_thread = None
_stop_event = threading.Event()


def _get_detector() -> FallDetector:
    global _detector
    if _detector is None:
        _detector = FallDetector()
    return _detector


def _update_frame(frame):
    global _current_frame_bytes
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    with _frame_lock:
        _current_frame_bytes = buf.tobytes()


def _log(msg: str):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    _event_log.insert(0, {"time": ts, "msg": msg})
    if len(_event_log) > 100:
        _event_log.pop()


# ─── Camera Worker ──────────────────────────────────────────────
def _camera_worker():
    global _status
    det = _get_detector()
    det.reset()

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        _log("错误：无法打开摄像头，请检查设备连接")
        _status["source"] = "idle"
        _status["is_running"] = False
        return

    _log("摄像头已启动，开始监测")
    fps_counter = 0
    fps_start = time.time()
    prev_fallen = False

    while not _stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        output_frame, fall_confirmed = det.process_frame(frame)

        if fall_confirmed and not prev_fallen:
            _status["fall_count"] += 1
            _status["last_fall_time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            _log(f"检测到跌倒！（第 {_status['fall_count']} 次）")
        prev_fallen = fall_confirmed

        fps_counter += 1
        if time.time() - fps_start >= 1.0:
            _status["fps"] = fps_counter
            fps_counter = 0
            fps_start = time.time()

        cv2.putText(
            output_frame, f"FPS:{_status['fps']} | Camera",
            (10, output_frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )
        _update_frame(output_frame)

    cap.release()
    _log("摄像头已停止")
    _status["source"] = "idle"
    _status["is_running"] = False


# ─── Video Worker ──────────────────────────────────────────────
def _video_worker(video_path: str):
    global _status
    print(f"[worker] 线程已进入, path={video_path}")
    try:
        _video_worker_inner(video_path)
    except Exception as e:
        import traceback
        err = traceback.format_exc()
        print(f"[worker] 未捕获异常:\n{err}")
        _log(f"处理异常: {e}")
        _status["source"] = "idle"
        _status["is_running"] = False


def _video_worker_inner(video_path: str):
    global _status
    det = _get_detector()
    det.reset()
    print(f"[worker] Detector 初始化完成（预热帧数: {det._warmup_remaining}）")

    cap = cv2.VideoCapture(video_path)
    print(f"[worker] VideoCapture.isOpened()={cap.isOpened()}")

    if not cap.isOpened():
        msg = f"错误：无法打开视频: {video_path}"
        print(f"[worker] {msg}")
        _log(msg)
        _status["source"] = "idle"
        _status["is_running"] = False
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_src = cap.get(cv2.CAP_PROP_FPS)
    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[worker] 视频信息: {width}x{height}, fps={fps_src}, total_frames={total_frames}")

    frame_idx = 0
    _status["video_total"] = total_frames
    _status["video_frame"] = 0

    if total_frames > 0:
        _log(f"开始处理视频 {width}x{height} @{fps_src:.0f}fps 共{total_frames}帧")
    else:
        _log(f"开始处理视频 {width}x{height} @{fps_src:.0f}fps （总帧数未知）")

    while not _stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print(f"[worker] 读取帧失败，已处理 {frame_idx} 帧，读取结束")
            break

        if frame_idx == 0:
            print(f"[worker] 成功读取第一帧, shape={frame.shape}")

        output_frame, fall_confirmed = det.process_frame(frame)
        frame_idx += 1

        if frame_idx == 1:
            print(f"[worker] 第一帧处理完成, output_shape={output_frame.shape}")

        if fall_confirmed and not prev_fallen:
            _status["fall_count"] += 1
            _status["last_fall_time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            _log(f"第 {frame_idx} 帧检测到跌倒（共 {_status['fall_count']} 次）")
        prev_fallen = fall_confirmed

        _status["video_frame"] = frame_idx
        if total_frames > 0:
            _status["video_progress"] = int(frame_idx / total_frames * 100)
        else:
            _status["video_progress"] = -1

        cv2.putText(
            output_frame,
            f"Frame {frame_idx}" + (f"/{total_frames}" if total_frames > 0 else ""),
            (10, output_frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )
        _update_frame(output_frame)
        time.sleep(0.01)

    cap.release()
    try:
        os.remove(video_path)
    except Exception:
        pass

    msg = f"视频处理完成，共 {frame_idx} 帧，检测到 {_status['fall_count']} 次跌倒"
    print(f"[worker] {msg}")
    _log(msg)
    _status["source"] = "idle"
    _status["is_running"] = False
    _status["video_progress"] = 100


# ─── Routes ───────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            with _frame_lock:
                frame_bytes = _current_frame_bytes
            if frame_bytes:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
            time.sleep(0.04)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/start_camera", methods=["POST"])
def start_camera():
    global _worker_thread, _status
    if _status["is_running"]:
        return jsonify({"ok": False, "msg": "已有任务在运行，请先停止"})
    _stop_event.clear()
    _status.update({"source": "camera", "is_running": True,
                    "fall_count": 0, "fps": 0, "video_progress": 0})
    _worker_thread = threading.Thread(target=_camera_worker, daemon=True)
    _worker_thread.start()
    return jsonify({"ok": True, "msg": "摄像头已启动"})


@app.route("/upload_video", methods=["POST"])
def upload_video():
    global _worker_thread, _status
    print(f"[upload] 收到请求, files={list(request.files.keys())}, is_running={_status['is_running']}")
    if _status["is_running"]:
        return jsonify({"ok": False, "msg": "已有任务在运行，请先停止"})
    if "video" not in request.files:
        print("[upload] 错误：未找到 'video' 字段")
        return jsonify({"ok": False, "msg": "未收到视频文件"})
    f = request.files["video"]
    print(f"[upload] 文件名: '{f.filename}', 内容类型: {f.content_type}")
    if f.filename == "":
        return jsonify({"ok": False, "msg": "文件名为空"})

    filename = secure_filename(f.filename)
    save_path = os.path.abspath(os.path.join(UPLOAD_FOLDER, filename))
    f.save(save_path)
    file_size = os.path.getsize(save_path)
    print(f"[upload] 已保存到: {save_path} ({file_size} bytes)")

    _stop_event.clear()
    _status.update({"source": "video", "is_running": True,
                    "fall_count": 0, "fps": 0, "video_progress": 0,
                    "video_frame": 0, "video_total": 0})
    _log(f"文件已保存: {filename} ({file_size//1024}KB)，准备启动...")
    _worker_thread = threading.Thread(
        target=_video_worker, args=(save_path,), daemon=True
    )
    _worker_thread.start()
    print(f"[upload] Worker 线程已启动, thread={_worker_thread.name}")
    return jsonify({"ok": True, "msg": f"开始处理: {filename} ({file_size//1024}KB)"})


@app.route("/stop", methods=["POST"])
def stop():
    _stop_event.set()
    _status["is_running"] = False
    _log("用户手动停止")
    return jsonify({"ok": True, "msg": "已停止"})


@app.route("/test_alert", methods=["POST"])
def test_alert():
    threading.Thread(target=alert_module.trigger_alert, daemon=True).start()
    _log("测试报警已触发")
    return jsonify({"ok": True, "msg": "测试报警已触发"})


@app.route("/status")
def status():
    return jsonify({**_status, "log": _event_log[:20]})


@app.route("/debug")
def debug():
    thread_alive = _worker_thread.is_alive() if _worker_thread else False
    upload_files = os.listdir(UPLOAD_FOLDER) if os.path.exists(UPLOAD_FOLDER) else []
    return jsonify({
        "status": _status,
        "worker_thread_alive": thread_alive,
        "stop_event_set": _stop_event.is_set(),
        "upload_folder": os.path.abspath(UPLOAD_FOLDER),
        "upload_files": upload_files,
        "detector_loaded": _detector is not None,
        "log": _event_log,
    })


# ─── Main ──────────────────────────────────────────────────────
def _open_browser():
    time.sleep(1.5)
    webbrowser.open("http://127.0.0.1:5000")


if __name__ == "__main__":
    print("[AI智护银龄] Web 服务启动中，浏览器将自动打开...")
    print("[AI智护银龄] 访问地址: http://127.0.0.1:5000")
    threading.Thread(target=_open_browser, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
