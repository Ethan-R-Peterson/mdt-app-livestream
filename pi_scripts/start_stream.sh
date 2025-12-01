#!/bin/bash
# Quick start script for Lepton stream + telemetry server
#
# Usage:
#   ./start_stream.sh best.onnx
#   ./start_stream.sh best.onnx 0 person,car,dog
#
# Arguments:
#   $1: ONNX model path (required)
#   $2: Video device number (default: 0)
#   $3: Class names comma-separated (default: person)

set -e

# Check if model file provided
if [ -z "$1" ]; then
    echo "Error: ONNX model path required"
    echo "Usage: $0 <model.onnx> [device] [classes]"
    echo "Example: $0 best.onnx 0 person"
    exit 1
fi

ONNX_MODEL="$1"
VIDEO_DEV="${2:-0}"
CLASSES="${3:-person}"

# Check if model exists
if [ ! -f "$ONNX_MODEL" ]; then
    echo "Error: Model file not found: $ONNX_MODEL"
    exit 1
fi

echo "=========================================="
echo "Starting Lepton Stream + Telemetry Server"
echo "=========================================="
echo "Model:   $ONNX_MODEL"
echo "Device:  /dev/video$VIDEO_DEV"
echo "Classes: $CLASSES"
echo ""

# Get Pi IP address
PI_IP=$(hostname -I | awk '{print $1}')
echo "Pi IP Address: $PI_IP"
echo ""
echo "Access from MacBook:"
echo "  MJPEG Stream: http://$PI_IP:5001/stream.mjpg"
echo "  WebSocket:    ws://$PI_IP:8765"
echo "  Test Page:    http://$PI_IP:5001/"
echo ""
echo "Press Ctrl+C to stop"
echo "=========================================="
echo ""

# Run the integrated server
python3 stream_and_telemetry.py \
    --onnx "$ONNX_MODEL" \
    --dev "$VIDEO_DEV" \
    --classes "$CLASSES" \
    --imgsz 640 \
    --conf 0.25 \
    --nms 0.45
