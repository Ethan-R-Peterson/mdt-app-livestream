# Raspberry Pi Setup Guide

This folder contains scripts to run on your Raspberry Pi for streaming Lepton thermal camera feed and telemetry data to the web application.

## Overview

The Pi runs two servers:
1. **MJPEG Stream Server** (`lepton_stream.py`) - Streams thermal camera video over HTTP
2. **Telemetry WebSocket Server** (`telemetry_server_pi.py`) - Sends detection and sensor data over WebSocket

## Prerequisites

### Hardware
- Raspberry Pi (3/4/5 or Zero 2 W recommended)
- FLIR Lepton thermal camera
- Camera connected via PureThermal board or direct I2C/SPI

### Software
- Raspberry Pi OS (Bullseye or later)
- Python 3.7+

## Installation

### 1. Clone this repository on your Pi

```bash
cd ~
git clone <your-repo-url>
cd App-LiveStream/pi_scripts
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Find your Pi's IP address

```bash
hostname -I
```

Note this IP address (e.g., `192.168.1.100`) - you'll need it to configure the MacBook.

## Configuration

### Integrate with your Lepton camera code

#### Option A: Using PureThermal board (V4L2)

Edit `lepton_stream.py`, replace `get_lepton_frame()`:

```python
# Initialize camera once at module level
cap = cv2.VideoCapture('/dev/video0')
cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)  # Get raw thermal data

def get_lepton_frame():
    ret, frame = cap.read()
    if not ret:
        return None

    # Normalize thermal data to 0-255
    normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Apply color map (optional, for visualization)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)

    # Resize to display size
    resized = cv2.resize(colored, (800, 450), interpolation=cv2.INTER_NEAREST)

    return resized
```

#### Option B: Using direct I2C/SPI (pylepton)

```python
from pylepton import Lepton

lepton = Lepton()

def get_lepton_frame():
    frame = lepton.capture()

    # Convert to 8-bit
    normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    resized = cv2.resize(colored, (800, 450), interpolation=cv2.INTER_NEAREST)

    return resized
```

### Integrate with your detection model

Edit `telemetry_server_pi.py`:

1. **Update `get_detection_results()`** with your model inference code
2. **Start a detection thread** that continuously updates `current_detections`

Example with YOLOv5:

```python
import torch
import cv2
import threading

# Load model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Global state
current_detections = []

def detection_loop():
    global current_detections

    cap = cv2.VideoCapture('/dev/video0')

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Run inference
        results = model(frame)

        # Parse detections
        detections = []
        img_h, img_w = frame.shape[:2]

        for *box, conf, cls in results.xyxy[0]:
            x1, y1, x2, y2 = box
            detections.append({
                "x": float((x1 + x2) / 2) / img_w,
                "y": float((y1 + y2) / 2) / img_h,
                "w": float(x2 - x1) / img_w,
                "h": float(y2 - y1) / img_h,
                "label": model.names[int(cls)],
                "conf": float(conf)
            })

        current_detections = detections

# Start detection thread
threading.Thread(target=detection_loop, daemon=True).start()
```

## Running the Servers

### Start MJPEG stream server

```bash
python lepton_stream.py
```

This starts the video stream server on port **5001**.

Test it: Open `http://<pi-ip>:5001/` in a browser on your MacBook.

### Start telemetry WebSocket server

In a separate terminal:

```bash
python telemetry_server_pi.py
```

This starts the telemetry server on port **8765**.

### Run both at startup (optional)

Create a systemd service or add to `/etc/rc.local`:

```bash
# Add to /etc/rc.local before 'exit 0'
cd /home/pi/App-LiveStream/pi_scripts
python lepton_stream.py &
python telemetry_server_pi.py &
```

## MacBook Configuration

On your MacBook, update the web app configuration:

### 1. Edit `config.json`

```json
{
  "sources": {
    "rgb_hls": "",
    "thermal_hls": "",
    "thermal_mjpeg": "http://192.168.1.100:5001/stream.mjpg",
    "test_hls": ""
  },
  "telemetry_ws": "ws://192.168.1.100:8765"
}
```

Replace `192.168.1.100` with your Pi's actual IP!

### 2. Edit `index.html` allowlist

Find this line (~line 91):

```javascript
const ALLOWLIST_HOSTS = ["localhost","127.0.0.1"];
```

Update to:

```javascript
const ALLOWLIST_HOSTS = ["localhost","127.0.0.1","192.168.1.100"];
```

Again, use your Pi's IP address.

### 3. Run the web app

```bash
python -m http.server 8000
```

Open `http://localhost:8000` and:
1. Select **"Thermal (MJPEG)"** source
2. Click **"Play"**
3. Click **"Connect Telemetry"**

You should see the Lepton camera feed with detection boxes overlaid!

## Troubleshooting

### Stream not loading
- Check Pi is on same WiFi network
- Verify Pi's IP address: `hostname -I`
- Test stream directly: `http://<pi-ip>:5001/` in browser
- Check firewall on Pi: `sudo ufw allow 5001` and `sudo ufw allow 8765`

### Telemetry not connecting
- Check WebSocket URL in browser console
- Verify telemetry server is running: `ps aux | grep telemetry`
- Test with `wscat`: `wscat -c ws://<pi-ip>:8765`

### Low frame rate
- Reduce JPEG quality in `lepton_stream.py` (line 12)
- Lower frame rate (line 11)
- Check CPU usage: `htop`

### Camera not detected
- Check camera connection
- For V4L2: `ls /dev/video*`
- For I2C: `i2cdetect -y 1`
- Check camera permissions: `sudo usermod -a -G video $USER`

## Network Setup Tips

### Static IP for Pi (recommended)

Edit `/etc/dhcpcd.conf`:

```bash
interface wlan0
static ip_address=192.168.1.100/24
static routers=192.168.1.1
static domain_name_servers=192.168.1.1 8.8.8.8
```

Reboot Pi: `sudo reboot`

### Hotspot mode (for field use)

Install hostapd to make Pi a WiFi access point, then connect MacBook directly to Pi's WiFi.

## Performance Optimization

### Enable hardware encoding (H.264)

For better quality/bandwidth, consider using HLS with hardware encoding:

```bash
# Install MediaMTX
wget https://github.com/bluenviron/mediamtx/releases/latest/download/mediamtx_*_linux_arm64v8.tar.gz
tar -xzf mediamtx_*.tar.gz
./mediamtx &

# Stream with hardware encoder
ffmpeg -f v4l2 -i /dev/video0 -c:v h264_v4l2m2m -b:v 2M -f rtsp rtsp://localhost:8554/lepton
```

Then use HLS URL: `http://<pi-ip>:8888/lepton/index.m3u8`

## Additional Resources

- [FLIR Lepton Documentation](https://www.flir.com/products/lepton/)
- [PureThermal Guide](https://groupgets.com/manufacturers/getlab/products/purethermal-2-flir-lepton-dev-kit)
- [MediaMTX Documentation](https://github.com/bluenviron/mediamtx)
- [YOLOv5 on Raspberry Pi](https://github.com/ultralytics/yolov5/wiki/Raspberry-Pi)
