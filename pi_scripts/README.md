# Raspberry Pi Setup Guide

This folder contains scripts to run on your Raspberry Pi for streaming Lepton thermal camera feed with YOLO detection to the web application.

## 🚀 Quick Start (Recommended)

**Use the all-in-one server that combines video streaming + detection + telemetry:**

```bash
# Simple startup with your ONNX model
./start_stream.sh best.onnx

# Or with custom device and classes
./start_stream.sh best.onnx 0 "person,car,dog"
```

That's it! Access the stream from your MacBook at `http://<pi-ip>:5001/stream.mjpg`

---

## 📁 Files in this Directory

### **`stream_and_telemetry.py`** ⭐ MAIN SERVER
All-in-one server that runs:
- MJPEG video stream (port 5001)
- WebSocket telemetry (port 8765)
- YOLO detection on Lepton camera

**Usage:**
```bash
python stream_and_telemetry.py --onnx best.onnx --dev 0 --classes "person"
```

### **`start_stream.sh`**
Convenience wrapper script for quick startup with automatic IP display

### **`connect_lepton.py`**
Your original script - standalone YOLO detection with local display (no streaming, kept for reference)

### **`requirements.txt`**
Python dependencies

### **`README.md`**
This file - complete setup guide

---

## 📋 Prerequisites

### Hardware
- Raspberry Pi (tested on Pi 5, should work on 3/4/Zero 2 W)
- FLIR Lepton 3.5 thermal camera
- PureThermal 3 board (or compatible V4L2 interface)

### Software
- Raspberry Pi OS (Bullseye or later)
- Python 3.7+
- Your trained YOLO model exported to ONNX format

---

## 🔧 Installation

### 1. Clone repository on your Pi

```bash
cd ~
git clone <your-repo-url>
cd App-LiveStream/pi_scripts
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Get your Pi's IP address

```bash
hostname -I
# Example output: 192.168.1.100
```

Save this IP - you'll need it for MacBook configuration!

---

## 🎯 Running the Server

### Method 1: Using startup script (easiest)

```bash
chmod +x start_stream.sh
./start_stream.sh best.onnx
```

### Method 2: Direct Python command

```bash
python stream_and_telemetry.py \
    --onnx best.onnx \
    --dev 0 \
    --classes "person" \
    --imgsz 640 \
    --conf 0.25 \
    --nms 0.45
```

### Command-line arguments:
- `--onnx`: Path to your ONNX model (required)
- `--dev`: Video device number (default: 0 = /dev/video0)
- `--classes`: Comma-separated class names (e.g., "person,car,dog")
- `--imgsz`: YOLO input size (default: 640)
- `--conf`: Confidence threshold (default: 0.25)
- `--nms`: NMS IoU threshold (default: 0.45)

---

## 🌐 Accessing the Stream

Once running, you'll see output like:

```
🚀 Lepton Camera Stream + Telemetry Server
================================================
📹 Camera: /dev/video0
🧠 ONNX Model: best.onnx
🌐 MJPEG Stream:  http://0.0.0.0:5001/stream.mjpg
📡 WebSocket:     ws://0.0.0.0:8765
🧪 Test Page:     http://0.0.0.0:8765/
================================================
```

### Test on Pi (optional):
```bash
# Test page with live video
firefox http://localhost:5001/
```

### Access from MacBook:
Use the Pi's IP address from step 3 above.

---

## 💻 MacBook Configuration

On your MacBook, navigate to the `App-LiveStream` directory and update:

### 1. Edit `config.json`
Replace `192.168.1.100` with your Pi's IP:

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

### 2. Edit `index.html`
Find line ~91 and add your Pi's IP to the allowlist:

```javascript
const ALLOWLIST_HOSTS = ["localhost","127.0.0.1","192.168.1.100"];
```

### 3. Run the web app

```bash
python -m http.server 8000
# Open http://localhost:8000
```

### 4. Use the web interface

1. Select source: **"Thermal (MJPEG)"**
2. Click **"Play"** - you should see the Lepton camera feed with detection boxes!
3. Click **"Connect Telemetry"** - detection boxes will now overlay on the video

---

## ✅ Verification

### Test stream directly on Pi (optional)

```bash
# View test page
firefox http://localhost:5001/

# Or check if stream is working
curl -I http://localhost:5001/stream.mjpg
```

### Test from MacBook

```bash
# Test MJPEG stream
curl -I http://192.168.1.100:5001/stream.mjpg

# Test WebSocket (requires wscat: npm install -g wscat)
wscat -c ws://192.168.1.100:8765
```

---

## 🔄 Auto-start on Boot (Optional)

To start the server automatically when Pi boots:

### Method 1: systemd service (recommended)

Create `/etc/systemd/system/lepton-stream.service`:

```ini
[Unit]
Description=Lepton Camera Stream Server
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/App-LiveStream/pi_scripts
ExecStart=/usr/bin/python3 stream_and_telemetry.py --onnx /home/pi/best.onnx --classes person
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable lepton-stream
sudo systemctl start lepton-stream
sudo systemctl status lepton-stream
```

### Method 2: crontab

```bash
crontab -e
# Add this line:
@reboot cd /home/pi/App-LiveStream/pi_scripts && /usr/bin/python3 stream_and_telemetry.py --onnx /home/pi/best.onnx --classes person
```

---

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
