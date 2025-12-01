# python connect_lepton.py --onnx best.onnx --show --dev 0 --imgsz 640 --conf 0.25 --nms 0.45 --classes "person"

#!/usr/bin/env python3
import cv2
import numpy as np
import time
import argparse

W, H = 160, 120   # Lepton 3.5 native resolution

# ----------------- ARGS -----------------
def get_args():
    p = argparse.ArgumentParser(description="FLIR Lepton 3.5 + PureThermal3 + YOLOv5 ONNX on Pi 5")
    p.add_argument("--onnx", required=True, help="Path to YOLOv5 ONNX (e.g., best.onnx)")
    p.add_argument("--dev", type=int, default=0, help="Video device index (/dev/videoX)")
    p.add_argument("--imgsz", type=int, default=640, help="YOLO input size (same as export)")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--nms", type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--classes", default="", help="Comma-separated class names (optional)")
    p.add_argument("--show", action="store_true", help="Show live window")
    return p.parse_args()

# ------------- CAMERA → Y16 → 8-BIT -------------
def read_lepton_frame(cap):
    ok, frame = cap.read()
    if not ok or frame is None or frame.size == 0:
        return None, False

    # Case we saw: (1, 38400) uint8  → reshape to (H, W, 2)
    if frame.ndim == 2 and frame.shape[0] == 1:
        flat = frame.reshape(-1)
        if flat.size != W * H * 2:
            print(f"[WARN] Unexpected flat size: {flat.size}, expected {W*H*2}")
            return None, False
        frame = flat.reshape(H, W, 2)

    # Expect (H, W, 2) uint8: Y16 = 16-bit little-endian
    if frame.ndim != 3 or frame.shape[2] != 2:
        print("[WARN] Unexpected frame shape:", frame.shape)
        return None, False

    # View as uint16 little-endian and reshape to (H, W)
    y16 = frame.view('<u2').reshape(H, W)

    # Normalize this frame to 0–255 (same logic as your working test script)
    u8 = cv2.normalize(y16, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return u8, True

# ------------- LETTERBOX (YOLO STYLE) -------------
def letterbox(img, new_size, color=(114,114,114)):
    h, w = img.shape[:2]
    r = min(new_size / h, new_size / w)
    nh, nw = int(round(h*r)), int(round(w*r))
    nh = max(1, nh)
    nw = max(1, nw)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_size, new_size, 3), color, dtype=np.uint8)
    top = (new_size - nh) // 2
    left = (new_size - nw) // 2
    canvas[top:top+nh, left:left+nw] = resized
    return canvas, r, left, top

# ------------- ROBUST NMS WRAPPER -------------
def nms_xywh(boxes, scores, iou_th):
    """
    boxes: [x, y, w, h] in letterboxed coords
    scores: list of scores
    Returns: list of kept indices
    """
    if not boxes:
        return []
    idxs = cv2.dnn.NMSBoxes(boxes, scores, 0.0, iou_th)
    if idxs is None or len(idxs) == 0:
        return []

    # Normalize to flat list of ints
    if isinstance(idxs, (list, tuple)):
        flat = []
        for v in idxs:
            if isinstance(v, (list, tuple, np.ndarray)):
                flat.append(int(v[0]))
            else:
                flat.append(int(v))
        return flat
    else:
        arr = np.array(idxs).reshape(-1)
        return arr.astype(int).tolist()

# ----------------- MAIN -----------------
def main():
    args = get_args()

    class_names = [s.strip() for s in args.classes.split(",")] if args.classes else None

    # Load YOLOv5 ONNX
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
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y','1','6',' '))

    print("[INFO] Press ESC to quit.")
    printed_info = False
    t0, frames = time.time(), 0

    while True:
        u8, ok = read_lepton_frame(cap)
        if not ok:
            continue

        if not printed_info:
            print(f"[INFO] got 8-bit frame shape={u8.shape}, dtype={u8.dtype}")
            printed_info = True

        # ----- Prepare model input -----
        # Use the same grayscale for all 3 channels
        model_img = cv2.merge([u8, u8, u8])
        lb, r, padw, padh = letterbox(model_img, args.imgsz)

        blob = cv2.dnn.blobFromImage(
            lb, 1/255.0,
            (args.imgsz, args.imgsz),
            swapRB=True,
            crop=False
        )
        net.setInput(blob)

        t_infer = time.time()
        preds = net.forward().squeeze(0)  # (N, 5+nc)
        infer_ms = (time.time() - t_infer) * 1000.0

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

            # xywh in letterboxed image coords (assumes ONNX outputs absolute pixels)
            x1 = x - w/2
            y1 = y - h/2
            boxes.append([int(x1), int(y1), int(w), int(h)])
            scores.append(sc)
            cls_ids.append(cid)

        keep = nms_xywh(boxes, scores, args.nms)

        # ----- Build colorized view for display -----
        color = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
        vis_lb, _, _, _ = letterbox(color, args.imgsz)
        out = vis_lb.copy()

        for i in keep:
            x, y, w, h = boxes[i]
            cv2.rectangle(out, (x, y), (x+w, y+h), (0, 255, 0), 2)
            label = f"{scores[i]:.2f}"
            if class_names and 0 <= cls_ids[i] < len(class_names):
                label = f"{class_names[cls_ids[i]]} {label}"
            else:
                label = f"id:{cls_ids[i]} {label}"
            cv2.putText(out, label, (x, max(0, y-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

        frames += 1
        if frames % 10 == 0:
            fps = frames / max(1e-6, (time.time()-t0))
            cv2.putText(out, f"FPS~{fps:.1f}", (10, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, 1)
        cv2.putText(out, f"Infer {infer_ms:.1f} ms", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, 1)

        if args.show:
            cv2.imshow("Thermal YOLOv5 (ONNX, Lepton 3.5)", out)
            if (cv2.waitKey(1) & 0xFF) == 27:  # ESC
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
