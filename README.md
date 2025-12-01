# App-Livestream (MDT) [Last Updated: Dec 1, 2025]

Template for web app to display a live drone feed + incoming telemetry, with room for overlays and a map panel.

## Quick Start

### MacBook (Web App)
```bash
python -m http.server 8000
# Open http://localhost:8000
```

### Raspberry Pi (Stream Server)
See **[pi_scripts/README.md](pi_scripts/README.md)** for complete Pi setup instructions.

```bash
# On your Raspberry Pi
cd pi_scripts
pip install -r requirements.txt
./start_stream.sh best.onnx  # Single command runs everything!
```

## Using the Web App

1. **Select source**: Choose "Thermal (MJPEG)" for Lepton camera
2. **Stream URL**: Should auto-load from `config.json` (e.g., `http://192.168.1.100:5001/stream.mjpg`)
3. **Click "Play"**: Starts video stream
4. **Click "Connect Telemetry"**: Connects to WebSocket for detection overlays

Detection boxes will automatically overlay on the video if your Pi sends detection data.

## Stream Architecture

**Current (Lepton Camera):**
```
Raspberry Pi                          MacBook Browser
├─ Lepton Camera ──→ MJPEG/HTTP ────→ <img> or <video>
└─ Detection Model ─→ WebSocket ────→ Canvas Overlay
```

**Future (RGB + Advanced):**
```
Raspberry Pi                          MacBook Browser
├─ RGB Camera ────→ RTSP → MediaMTX → HLS ──→ Video Player (HLS.js)
├─ Thermal Camera → RTSP → MediaMTX → HLS ──→ Video Player
└─ Telemetry ─────→ WebSocket ──────────────→ Canvas Overlay
```

## Configuration

### MacBook: `config.json`
```json
{
  "sources": {
    "thermal_mjpeg": "http://192.168.1.100:5001/stream.mjpg"
  },
  "telemetry_ws": "ws://192.168.1.100:8765"
}
```

### MacBook: `index.html` (allowlist)
Update line ~91 with your Pi's IP:
```javascript
const ALLOWLIST_HOSTS = ["localhost","127.0.0.1","192.168.1.100"];
```

## Project Structure

```
App-LiveStream/
├── index.html                       # Web UI (MacBook)
├── config.json                      # Stream URLs (MacBook)
├── telemetry_server.py              # Test telemetry (MacBook - dummy data)
├── thermal_mjpeg_dummy.py           # Test thermal stream (MacBook - dummy)
└── pi_scripts/                      # Raspberry Pi code
    ├── README.md                    # Pi setup guide ⭐
    ├── requirements.txt             # Pi dependencies
    ├── stream_and_telemetry.py      # All-in-one server (stream + detection + telemetry)
    ├── start_stream.sh              # Quick startup script
    └── connect_lepton.py            # Original detection script (reference)
```

## Map Integration
A placeholder `<div id="map">` is provided for future map integration (Leaflet, Mapbox, etc.).

---

## Testing Without Raspberry Pi

### Local Testing with Dummy Servers

#### 1. Test Telemetry
```bash
pip install websockets
python telemetry_server.py
```
Generates simulated GPS, altitude, battery, and detection data.

#### 2. Test Thermal Stream
```bash
pip install flask pillow numpy
python thermal_mjpeg_dummy.py
```
Generates animated thermal imagery with moving "hot spot".

#### 3. Run Web App
```bash
python -m http.server 8000
# Open http://localhost:8000
# Select "Thermal (MJPEG)" → Play
# Click "Connect Telemetry"
```

---

## Deployment Checklist

### On Raspberry Pi:
- [ ] Connect Lepton camera
- [ ] Install dependencies: `pip install -r pi_scripts/requirements.txt`
- [ ] Copy your ONNX model to Pi
- [ ] Get Pi IP address: `hostname -I`
- [ ] Start server: `./start_stream.sh best.onnx`

### On MacBook:
- [ ] Update `config.json` with Pi's IP
- [ ] Update `index.html` ALLOWLIST_HOSTS with Pi's IP
- [ ] Start web server: `python -m http.server 8000`
- [ ] Open browser: `http://localhost:8000`
- [ ] Select source, Play, and Connect Telemetry

---

## Troubleshooting

**Stream not loading?**
- Check both devices on same WiFi
- Verify Pi IP: `hostname -I`
- Test stream directly: `http://<pi-ip>:5001/`

**Telemetry not connecting?**
- Check WebSocket URL in browser console
- Verify server running: `ps aux | grep telemetry`

**Detection boxes not showing?**
- Check "Draw overlay" is enabled
- Verify detections in telemetry output
- Open browser console for errors

---

## Notes
- Styling intentionally minimal. Clean module boundaries for future features.
- No auto-play/auto-connect for security and user control.
- URL allowlist prevents loading untrusted streams.

