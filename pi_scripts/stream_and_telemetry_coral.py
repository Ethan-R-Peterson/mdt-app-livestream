#!/usr/bin/env python3
"""
Lepton Camera Stream + Telemetry Server with Google Coral TPU
Uses Edge TPU for accelerated YOLO inference

Usage:
    python stream_and_telemetry_coral.py --tflite best_edgetpu.tflite --dev 0 --classes "person"
"""

from flask import Flask, Response
import cv2
import numpy as np
import time
import argparse
import threading
import queue
import asyncio
import websockets
import json
from pycoral.utils import edgetpu
from pycoral.adapters import common
from pycoral.adapters import detect

# Lepton 3.5 native resolution
W, H = 160, 120

# Global state
frame_queue = queue.Queue(maxsize=2)
detection_lock = threading.Lock()
current_detections = []

# Web server
app = Flask(__name__)

# Configuration
HTTP_PORT = 5001
WS_PORT = 8765
JPEG_QUALITY = 80
TELEMETRY_RATE = 5  # Hz


# ============================================================================
# LEPTON CAMERA FUNCTIONS
# ============================================================================

def read_lepton_frame(cap):
    """Read and process Lepton Y16 frame to 8-bit grayscale"""
    ok, frame = cap.read()
    if not ok or frame is None or frame.size == 0:
        return None, False

    if frame.ndim == 2 and frame.shape[0] == 1:
        flat = frame.reshape(-1)
        if flat.size != W * H * 2:
            return None, False
        frame = flat.reshape(H, W, 2)

    if frame.ndim != 3 or frame.shape[2] != 2:
        return None, False

    y16 = frame.view('<u2').reshape(H, W)
    u8 = cv2.normalize(y16, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return u8, True


def letterbox(img, new_size, color=(114, 114, 114)):
    """Letterbox resize for YOLO"""
    h, w = img.shape[:2]
    r = min(new_size / h, new_size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    nh, nw = max(1, nh), max(1, nw)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_size, new_size, 3), color, dtype=np.uint8)
    top = (new_size - nh) // 2
    left = (new_size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, left, top


# ============================================================================
# CORAL TPU DETECTION
# ============================================================================

def preprocess_for_coral(image, input_size):
    """Preprocess image for Coral TPU input"""
    # Resize to model input size
    resized = cv2.resize(image, (input_size, input_size))

    # Convert BGR to RGB (if needed)
    if len(resized.shape) == 2:  # Grayscale
        rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    return rgb


def run_coral_inference(interpreter, image, conf_threshold=0.25):
    """
    Run inference on Coral Edge TPU

    Args:
        interpreter: PyCoral interpreter
        image: Preprocessed RGB image (already resized)
        conf_threshold: Confidence threshold

    Returns:
        detections: List of [class_id, score, x1, y1, x2, y2]
    """
    # Set input tensor
    common.set_input(interpreter, image)

    # Run inference
    interpreter.invoke()

    # Get detection results
    # Note: This assumes your model outputs detection format
    # You may need to adjust based on your specific model output format

    detections = []

    # For YOLO models, typically you get:
    # - boxes: [N, 4] (x1, y1, x2, y2)
    # - scores: [N]
    # - classes: [N]

    # Get output tensors (adjust indices based on your model)
    output_tensor = common.output_tensor(interpreter, 0)

    # Parse YOLO output (adjust based on your model format)
    # Assuming output shape: (1, N, 5+num_classes) where N is number of predictions
    # [x, y, w, h, obj_conf, class1_conf, class2_conf, ...]

    for detection in output_tensor:
        # Extract box coordinates
        x, y, w, h = detection[0:4]
        obj_conf = detection[4]

        if obj_conf < conf_threshold:
            continue

        # Get class scores
        class_scores = detection[5:]
        class_id = int(np.argmax(class_scores))
        class_conf = float(class_scores[class_id])
        final_score = obj_conf * class_conf

        if final_score < conf_threshold:
            continue

        # Convert from center format to corner format
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2

        detections.append({
            'class_id': class_id,
            'score': float(final_score),
            'box': [float(x1), float(y1), float(x2), float(y2)]
        })

    return detections


# ============================================================================
# DETECTION THREAD (CORAL TPU)
# ============================================================================

def detection_loop_coral(args):
    """Main detection loop using Coral TPU"""
    global current_detections

    class_names = [s.strip() for s in args.classes.split(",")] if args.classes else None

    # Load model on Edge TPU
    print(f"[CORAL] Loading model: {args.tflite}")
    interpreter = edgetpu.make_interpreter(args.tflite)
    interpreter.allocate_tensors()

    # Get input details
    input_details = interpreter.get_input_details()
    input_size = input_details[0]['shape'][1]  # Assuming square input

    print(f"[CORAL] Model input size: {input_size}x{input_size}")
    print(f"[CORAL] Edge TPU initialized successfully")

    # Open camera
    cap = cv2.VideoCapture(args.dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open /dev/video{args.dev}")

    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS, 8)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', '1', '6', ' '))

    print("[CORAL] Camera initialized")
    t0, frames = time.time(), 0

    while True:
        u8, ok = read_lepton_frame(cap)
        if not ok:
            time.sleep(0.01)
            continue

        # Prepare model input
        model_img = cv2.merge([u8, u8, u8])
        preprocessed = preprocess_for_coral(model_img, input_size)

        # Run inference on Coral TPU
        t_infer = time.time()
        detections = run_coral_inference(interpreter, preprocessed, args.conf)
        infer_ms = (time.time() - t_infer) * 1000.0

        # Convert detections to normalized format for WebSocket
        normalized_dets = []
        for det in detections:
            x1, y1, x2, y2 = det['box']
            label = class_names[det['class_id']] if class_names and 0 <= det['class_id'] < len(class_names) else f"id:{det['class_id']}"

            normalized_dets.append({
                "x": ((x1 + x2) / 2) / input_size,
                "y": ((y1 + y2) / 2) / input_size,
                "w": (x2 - x1) / input_size,
                "h": (y2 - y1) / input_size,
                "label": label,
                "conf": det['score']
            })

        # Update global state
        with detection_lock:
            current_detections = normalized_dets

        # Visualization
        color = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
        vis_lb, _, _, _ = letterbox(color, input_size)
        out = vis_lb.copy()

        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det['box']
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det['score']:.2f}"
            if class_names and 0 <= det['class_id'] < len(class_names):
                label = f"{class_names[det['class_id']]} {label}"
            cv2.putText(out, label, (x1, max(0, y1 - 5)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        frames += 1
        if frames % 10 == 0:
            fps = frames / max(1e-6, (time.time() - t0))
            cv2.putText(out, f"FPS~{fps:.1f}", (10, 42),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(out, f"Coral {infer_ms:.1f}ms", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        try:
            frame_queue.put_nowait(out)
        except queue.Full:
            pass


# ============================================================================
# MJPEG HTTP STREAM
# ============================================================================

def generate_mjpeg():
    """MJPEG stream generator"""
    while True:
        try:
            frame = frame_queue.get(timeout=1.0)
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n'
                       b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n'
                       b'\r\n' + frame_bytes + b'\r\n')
        except queue.Empty:
            time.sleep(0.01)


@app.route('/stream.mjpg')
def stream():
    return Response(generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return f'''
    <html>
        <head><title>Lepton + Coral TPU</title></head>
        <body style="background:#000; color:#0f0; font-family:monospace; padding:20px;">
            <h1>🔥 Lepton + Google Coral TPU Stream</h1>
            <img src="/stream.mjpg" style="width:100%; max-width:800px; border:2px solid #0f0;" />
            <div style="margin-top:20px;">
                <p>📹 MJPEG Stream: <code>http://&lt;pi-ip&gt;:{HTTP_PORT}/stream.mjpg</code></p>
                <p>📡 WebSocket: <code>ws://&lt;pi-ip&gt;:{WS_PORT}</code></p>
                <p>⚡ Accelerated by Google Coral Edge TPU</p>
            </div>
        </body>
    </html>
    '''


# ============================================================================
# WEBSOCKET TELEMETRY
# ============================================================================

async def telemetry_handler(websocket):
    """WebSocket handler"""
    client_addr = websocket.remote_address
    print(f"[WS] Client connected: {client_addr}")

    try:
        while True:
            with detection_lock:
                detections = current_detections.copy()

            message = {
                "time": time.time(),
                "lat": 0.0,
                "lon": 0.0,
                "alt": 0.0,
                "gs": 0.0,
                "batt": 100.0,
                "yaw": 0.0,
                "pitch": 0.0,
                "roll": 0.0,
                "dets": detections
            }

            await websocket.send(json.dumps(message))
            await asyncio.sleep(1.0 / TELEMETRY_RATE)

    except websockets.exceptions.ConnectionClosed:
        print(f"[WS] Client disconnected: {client_addr}")


async def websocket_server():
    """Start WebSocket server"""
    print(f"[WS] Starting WebSocket server on port {WS_PORT}")
    async with websockets.serve(telemetry_handler, "0.0.0.0", WS_PORT):
        await asyncio.Future()


def run_websocket_server():
    """Run WebSocket server in asyncio event loop"""
    asyncio.run(websocket_server())


# ============================================================================
# MAIN
# ============================================================================

def get_args():
    p = argparse.ArgumentParser(description="Lepton + Coral TPU Stream Server")
    p.add_argument("--tflite", required=True, help="Edge TPU TFLite model path (*_edgetpu.tflite)")
    p.add_argument("--dev", type=int, default=0, help="Video device")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--classes", default="", help="Class names (comma-separated)")
    return p.parse_args()


if __name__ == '__main__':
    args = get_args()

    print("=" * 70)
    print("🚀 Lepton Camera + Google Coral TPU Stream Server")
    print("=" * 70)
    print(f"📹 Camera: /dev/video{args.dev}")
    print(f"🧠 Model: {args.tflite}")
    print(f"⚡ Accelerator: Google Coral Edge TPU")
    print(f"⚙️  Confidence: {args.conf}")
    print(f"🏷️  Classes: {args.classes if args.classes else 'All'}")
    print()
    print(f"🌐 MJPEG Stream:  http://0.0.0.0:{HTTP_PORT}/stream.mjpg")
    print(f"📡 WebSocket:     ws://0.0.0.0:{WS_PORT}")
    print("=" * 70)

    # Start detection thread
    print("[MAIN] Starting Coral TPU detection thread...")
    detection_thread = threading.Thread(target=detection_loop_coral, args=(args,), daemon=True)
    detection_thread.start()

    # Start WebSocket server
    print("[MAIN] Starting WebSocket server...")
    ws_thread = threading.Thread(target=run_websocket_server, daemon=True)
    ws_thread.start()

    time.sleep(2)

    # Start Flask
    print("[MAIN] Starting HTTP server...")
    app.run(host='0.0.0.0', port=HTTP_PORT, threaded=True, debug=False)
