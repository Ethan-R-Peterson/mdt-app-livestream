#!/usr/bin/env python3
"""
Combined Lepton Stream + Telemetry Server
Single script that runs both MJPEG HTTP stream and WebSocket telemetry

This combines:
- Your connect_lepton.py detection logic
- MJPEG streaming server (port 5001)
- WebSocket telemetry server (port 8765)

Usage:
    python stream_and_telemetry.py --onnx best.onnx --dev 0 --classes "person"

Then on MacBook:
    - MJPEG stream: http://<pi-ip>:5001/stream.mjpg
    - WebSocket: ws://<pi-ip>:8765
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

# Lepton 3.5 native resolution
W, H = 160, 120

# Global state - shared between threads
frame_queue = queue.Queue(maxsize=2)  # Latest visualization frames
detection_lock = threading.Lock()
current_detections = []  # Detection results for WebSocket

# Web server
app = Flask(__name__)

# Configuration
HTTP_PORT = 5001
WS_PORT = 8765
JPEG_QUALITY = 80
TELEMETRY_RATE = 5  # Hz


# ============================================================================
# LEPTON + YOLO FUNCTIONS
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


def nms_xywh(boxes, scores, iou_th):
    """NMS for xywh boxes"""
    if not boxes:
        return []
    idxs = cv2.dnn.NMSBoxes(boxes, scores, 0.0, iou_th)
    if idxs is None or len(idxs) == 0:
        return []

    if isinstance(idxs, (list, tuple)):
        flat = []
        for v in idxs:
            if isinstance(v, (list, tuple, np.ndarray)):
                flat.append(int(v[0]))
            else:
                flat.append(int(v))
        return flat
    else:
        return np.array(idxs).reshape(-1).astype(int).tolist()


# ============================================================================
# DETECTION THREAD
# ============================================================================

def detection_loop(args):
    """Main detection loop - captures, detects, updates global state"""
    global current_detections

    class_names = [s.strip() for s in args.classes.split(",")] if args.classes else None

    # Load YOLO
    print(f"[DETECTION] Loading model: {args.onnx}")
    net = cv2.dnn.readNetFromONNX(args.onnx)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Open camera
    cap = cv2.VideoCapture(args.dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open /dev/video{args.dev}")

    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS, 8)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', '1', '6', ' '))

    print("[DETECTION] Camera initialized")
    t0, frames = time.time(), 0

    while True:
        u8, ok = read_lepton_frame(cap)
        if not ok:
            time.sleep(0.01)
            continue

        # YOLO inference
        model_img = cv2.merge([u8, u8, u8])
        lb, r, padw, padh = letterbox(model_img, args.imgsz)

        blob = cv2.dnn.blobFromImage(lb, 1 / 255.0, (args.imgsz, args.imgsz),
                                     swapRB=True, crop=False)
        net.setInput(blob)

        t_infer = time.time()
        preds = net.forward().squeeze(0)
        infer_ms = (time.time() - t_infer) * 1000.0

        # Parse detections
        boxes, scores, cls_ids = [], [], []
        for det in preds:
            x, y, w, h = det[0:4]
            obj = float(det[4])
            if obj < args.conf:
                continue
            cls_scores = det[5:]
            cid = int(np.argmax(cls_scores))
            sc = obj * float(cls_scores[cid])
            if sc < args.conf:
                continue

            boxes.append([int(x - w / 2), int(y - h / 2), int(w), int(h)])
            scores.append(sc)
            cls_ids.append(cid)

        keep = nms_xywh(boxes, scores, args.nms)

        # Convert detections to normalized format for WebSocket
        # Normalized coordinates (0-1) relative to letterboxed image
        normalized_dets = []
        for i in keep:
            bx, by, bw, bh = boxes[i]
            label = class_names[cls_ids[i]] if class_names and 0 <= cls_ids[i] < len(class_names) else f"id:{cls_ids[i]}"

            normalized_dets.append({
                "x": (bx + bw / 2) / args.imgsz,  # center x
                "y": (by + bh / 2) / args.imgsz,  # center y
                "w": bw / args.imgsz,
                "h": bh / args.imgsz,
                "label": label,
                "conf": float(scores[i])
            })

        # Update global state
        with detection_lock:
            current_detections = normalized_dets

        # Visualization for MJPEG stream
        color = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
        vis_lb, _, _, _ = letterbox(color, args.imgsz)
        out = vis_lb.copy()

        for i in keep:
            x, y, w, h = boxes[i]
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{scores[i]:.2f}"
            if class_names and 0 <= cls_ids[i] < len(class_names):
                label = f"{class_names[cls_ids[i]]} {label}"
            cv2.putText(out, label, (x, max(0, y - 5)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        frames += 1
        if frames % 10 == 0:
            fps = frames / max(1e-6, (time.time() - t0))
            cv2.putText(out, f"FPS~{fps:.1f}", (10, 42),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(out, f"Infer {infer_ms:.1f}ms", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

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
        <head><title>Lepton Stream</title></head>
        <body style="background:#000; color:#0f0; font-family:monospace; padding:20px;">
            <h1>🔥 Lepton + YOLO Live Stream</h1>
            <img src="/stream.mjpg" style="width:100%; max-width:800px; border:2px solid #0f0;" />
            <div style="margin-top:20px;">
                <p>📹 MJPEG Stream: <code>http://&lt;pi-ip&gt;:{HTTP_PORT}/stream.mjpg</code></p>
                <p>📡 WebSocket Telemetry: <code>ws://&lt;pi-ip&gt;:{WS_PORT}</code></p>
            </div>
        </body>
    </html>
    '''


# ============================================================================
# WEBSOCKET TELEMETRY
# ============================================================================

async def telemetry_handler(websocket):
    """WebSocket handler - sends detections + telemetry to browser"""
    client_addr = websocket.remote_address
    print(f"[WS] Client connected: {client_addr}")

    try:
        while True:
            with detection_lock:
                detections = current_detections.copy()

            message = {
                "time": time.time(),
                "lat": 0.0,  # Add real GPS if available
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
    p = argparse.ArgumentParser(description="Lepton Stream + Telemetry Server")
    p.add_argument("--onnx", required=True, help="YOLOv5 ONNX model path")
    p.add_argument("--dev", type=int, default=0, help="Video device")
    p.add_argument("--imgsz", type=int, default=640, help="YOLO input size")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--nms", type=float, default=0.45, help="NMS threshold")
    p.add_argument("--classes", default="", help="Class names (comma-separated)")
    return p.parse_args()


if __name__ == '__main__':
    args = get_args()

    print("=" * 70)
    print("🚀 Lepton Camera Stream + Telemetry Server")
    print("=" * 70)
    print(f"📹 Camera: /dev/video{args.dev}")
    print(f"🧠 ONNX Model: {args.onnx}")
    print(f"⚙️  Config: imgsz={args.imgsz}, conf={args.conf}, nms={args.nms}")
    print(f"🏷️  Classes: {args.classes if args.classes else 'All'}")
    print()
    print(f"🌐 MJPEG Stream:  http://0.0.0.0:{HTTP_PORT}/stream.mjpg")
    print(f"📡 WebSocket:     ws://0.0.0.0:{WS_PORT}")
    print(f"🧪 Test Page:     http://0.0.0.0:{HTTP_PORT}/")
    print("=" * 70)

    # Start detection thread
    print("[MAIN] Starting detection thread...")
    detection_thread = threading.Thread(target=detection_loop, args=(args,), daemon=True)
    detection_thread.start()

    # Start WebSocket server in separate thread
    print("[MAIN] Starting WebSocket server...")
    ws_thread = threading.Thread(target=run_websocket_server, daemon=True)
    ws_thread.start()

    # Give threads time to initialize
    time.sleep(2)

    # Start Flask (blocking)
    print("[MAIN] Starting HTTP server...")
    app.run(host='0.0.0.0', port=HTTP_PORT, threaded=True, debug=False)
