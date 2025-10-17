# App-Livestream (MDT) [Last Updated: Oct 17, 2025]

Template for web app to display a live drone feed + incoming telemetry, with room for overlays and a map panel.

## Run (local)
- VS Code → Live Server (right-click `index.html` → Open with Live Server), or
- Python: `python -m http.server 8000` then open http://localhost:8000

## Using it
- Enter HLS URL (e.g., http://localhost:8888/drone/index.m3u8) → **Play HLS**
- Enter Telemetry WebSocket URL (e.g., ws://localhost:8765) → **Connect Telemetry**
- Overlay canvas draws detection boxes if `dets: [{x,y,w,h,label,conf}]` are present.

## Stream architecture (proposed)
- RGB camera → RTSP → MediaMTX → HLS/WebRTC → browser
- MLX90640 (thermal) → Python frame capture → (A) MJPEG over HTTP **or** (B) pipe to ffmpeg → RTSP → MediaMTX → HLS/WebRTC
- Telemetry → WebSocket (JSON) → browser

## Map
A placeholder `<div id="map">` and module hook will be provided so the map can mount independently.

## Notes
- Styling intentionally minimal per request. Clean module boundaries to add features later.

## ----------------------------------------------------------------------------------------------------------------------
## Quick start
# 1) Static server
python -m http.server 8000
# open http://localhost:8000

# 2) Telemetry (optional)
pip install websockets
python telemetry_server.py
# in UI, WS is auto-loaded from config.json

# 3) Test HLS (optional)
docker run --rm -it -p 8554:8554 -p 8888:8888 bluenviron/mediamtx:latest
ffmpeg -re -stream_loop -1 -i sample.mp4 -c copy -f rtsp rtsp://localhost:8554/drone
# UI -> Source: RGB (HLS)

# 4) Thermal MJPEG dummy (optional)
pip install flask opencv-python-headless numpy
- or -
conda install -c conda-forge flask opencv numpy
python thermal_mjpeg_dummy.py
# UI -> Source: Thermal (MJPEG)

