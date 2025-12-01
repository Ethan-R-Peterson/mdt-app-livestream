# Google Coral TPU Setup Guide

Quick guide to set up Google Coral Edge TPU for accelerated YOLO inference.

## 📋 **Prerequisites**

- Google Coral USB Accelerator or M.2/PCIe Accelerator
- Raspberry Pi 5 (or Pi 4/3B+)
- Your YOLO model in Edge TPU TFLite format (`*_edgetpu.tflite`)

---

## 🔧 **Installation on Raspberry Pi**

### **1. Install Edge TPU Runtime**

```bash
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install libedgetpu1-std
```

**Note:** Use `libedgetpu1-std` for standard performance or `libedgetpu1-max` for maximum performance (runs hotter).

### **2. Install Python Libraries**

```bash
pip3 install pycoral tflite-runtime
```

### **3. Verify Coral Connection**

Plug in your Coral USB Accelerator and verify it's detected:

```bash
lsusb | grep "Google"
```

You should see:
```
Bus 00X Device 00X: ID 1a6e:089a Global Unichip Corp.
```

### **4. Install Other Dependencies**

```bash
cd ~/App-LiveStream/pi_scripts
pip install -r requirements.txt
```

---

## 🚀 **Running with Coral TPU**

### **Quick Start**

```bash
./start_stream_coral.sh your_model_edgetpu.tflite
```

### **With Custom Options**

```bash
python stream_and_telemetry_coral.py \
    --tflite your_model_edgetpu.tflite \
    --dev 0 \
    --classes "person,car,dog" \
    --conf 0.25
```

---

## ⚡ **Expected Performance Improvement**

### **Before (OpenCV DNN on CPU):**
- Inference time: **50-100ms** per frame
- FPS: ~10-15 fps
- CPU usage: ~80-100%

### **After (Coral Edge TPU):**
- Inference time: **5-15ms** per frame ⚡
- FPS: ~60+ fps
- CPU usage: ~20-30%

**That's 5-10x faster!** 🚀

---

## 📝 **Model Format Requirements**

Your model **must** be in Edge TPU TFLite format (ends with `_edgetpu.tflite`).

### **If you only have ONNX:**

You need to convert: **ONNX → TFLite → Edge TPU**

```bash
# Install conversion tools
pip install onnx2tf tensorflow

# Install Edge TPU Compiler
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
sudo apt-get update
sudo apt-get install edgetpu-compiler

# Convert your model (on your MacBook or Pi)
python convert_to_coral.py --onnx best.onnx --imgsz 640
```

This creates `best_int8_edgetpu.tflite` ready for Coral.

---

## 🔍 **Troubleshooting**

### **Coral not detected:**
```bash
# Check USB connection
lsusb | grep Google

# Check permissions
sudo usermod -a -G plugdev $USER
# Log out and back in
```

### **"No EdgeTPU device found" error:**
```bash
# Reinstall runtime
sudo apt-get install --reinstall libedgetpu1-std

# Check dmesg for errors
dmesg | grep apex
```

### **Slow inference (not using TPU):**
- Make sure model ends with `_edgetpu.tflite`
- Verify model was compiled with Edge TPU Compiler
- Check that pycoral is using the TPU: Look for "Edge TPU initialized" in logs

### **Model quantization errors:**
Your model must be INT8 quantized for Edge TPU. If conversion fails:
- Use a representative dataset during quantization
- Some operations may not be TPU-compatible (will fall back to CPU)
- Check Edge TPU Compiler output for unsupported ops

---

## 📊 **Performance Monitoring**

Watch the inference time in the video stream overlay:

```
Coral 8.5ms     ← This should be <20ms
```

If it's >50ms, the TPU might not be working correctly.

---

## 🆚 **Coral vs CPU Comparison**

| Metric | CPU (OpenCV DNN) | Coral Edge TPU |
|--------|------------------|----------------|
| Inference | 50-100ms | 5-15ms ⚡ |
| FPS | 10-15 | 60+ |
| CPU Usage | 80-100% | 20-30% |
| Power | High | Low |
| Latency | ~500ms | ~100ms |

**Winner:** Coral Edge TPU 🏆

---

## 🔗 **Additional Resources**

- [Coral USB Accelerator](https://coral.ai/products/accelerator/)
- [Edge TPU Compiler](https://coral.ai/docs/edgetpu/compiler/)
- [PyCoral API](https://coral.ai/docs/reference/py/)
- [Model Compatibility](https://coral.ai/docs/edgetpu/models-intro/)

---

## ✅ **Verification Checklist**

- [ ] Coral USB Accelerator connected
- [ ] Edge TPU runtime installed (`libedgetpu1-std`)
- [ ] Python libraries installed (`pycoral`, `tflite-runtime`)
- [ ] Model in Edge TPU format (`*_edgetpu.tflite`)
- [ ] Inference time <20ms in stream overlay
- [ ] Server starts without errors

---

## 🎯 **Next Steps**

Once Coral is working:
1. **Monitor inference time** - Should be <15ms
2. **Optimize confidence threshold** - Adjust `--conf` for your needs
3. **Consider WebRTC** - Lower latency streaming (next upgrade)
4. **Add more sensors** - GPS, IMU for full drone telemetry

Enjoy your 10x faster detection! 🚀
