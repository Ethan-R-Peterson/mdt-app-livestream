#!/bin/bash
# Quick start script for Lepton + Coral TPU stream server
#
# Usage:
#   ./start_stream_coral.sh best_edgetpu.tflite
#   ./start_stream_coral.sh best_edgetpu.tflite 0 person,car
#
# Arguments:
#   $1: Edge TPU TFLite model path (required, must end in _edgetpu.tflite)
#   $2: Video device number (default: 0)
#   $3: Class names comma-separated (default: person)

set -e

# Check if model file provided
if [ -z "$1" ]; then
    echo "Error: Edge TPU model path required"
    echo "Usage: $0 <model_edgetpu.tflite> [device] [classes]"
    echo "Example: $0 best_edgetpu.tflite 0 person"
    exit 1
fi

TFLITE_MODEL="$1"
VIDEO_DEV="${2:-0}"
CLASSES="${3:-person}"

# Check if model exists
if [ ! -f "$TFLITE_MODEL" ]; then
    echo "Error: Model file not found: $TFLITE_MODEL"
    exit 1
fi

# Check if it's an Edge TPU model
if [[ ! "$TFLITE_MODEL" =~ _edgetpu\.tflite$ ]]; then
    echo "Warning: Model file should end with '_edgetpu.tflite'"
    echo "Make sure you're using an Edge TPU compiled model!"
fi

echo "=================================================="
echo "🚀 Lepton + Google Coral TPU Stream Server"
echo "=================================================="
echo "Model:   $TFLITE_MODEL"
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
echo "⚡ Powered by Google Coral Edge TPU"
echo "Press Ctrl+C to stop"
echo "=================================================="
echo ""

# Run the Coral server
python3 stream_and_telemetry_coral.py \
    --tflite "$TFLITE_MODEL" \
    --dev "$VIDEO_DEV" \
    --classes "$CLASSES" \
    --conf 0.25
