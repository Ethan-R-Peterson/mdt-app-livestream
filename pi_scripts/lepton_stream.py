#!/usr/bin/env python3
"""
Lepton Camera MJPEG Stream Server
Streams Lepton thermal camera feed over HTTP as MJPEG
Access at: http://<pi-ip>:5001/stream.mjpg
"""

from flask import Flask, Response
import cv2
import numpy as np
import time
import io
from PIL import Image

app = Flask(__name__)

# Configuration
STREAM_PORT = 5001
FRAME_RATE = 15  # fps
JPEG_QUALITY = 80

# Lepton camera initialization
# TODO: Replace with your actual Lepton camera initialization
# Example for PureThermal/FLIR Lepton:
# from pylepton import Lepton
# lepton = Lepton()

def get_lepton_frame():
    """
    Capture a frame from the Lepton camera

    Replace this function with your actual Lepton camera capture code.
    Should return a numpy array (grayscale or RGB image)

    Examples:
    ---------
    # For PureThermal board with V4L2:
    # cap = cv2.VideoCapture('/dev/video0')
    # ret, frame = cap.read()
    # return frame

    # For direct I2C/SPI Lepton:
    # frame = lepton.capture()
    # return frame

    # For thermal image processing:
    # raw_frame = lepton.capture()
    # normalized = cv2.normalize(raw_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    # return colored
    """

    # PLACEHOLDER: Replace with actual camera code
    # For testing, generates a dummy thermal image
    print("WARNING: Using dummy frame generator. Replace get_lepton_frame() with actual camera code!")

    # Generate test pattern
    h, w = 120, 160
    frame = np.random.randint(0, 255, (h, w), dtype=np.uint8)

    # Optional: Apply colormap for thermal visualization
    colored = cv2.applyColorMap(frame, cv2.COLORMAP_JET)

    # Resize to display size
    resized = cv2.resize(colored, (800, 450), interpolation=cv2.INTER_NEAREST)

    return resized


def generate_mjpeg():
    """
    Generator function for MJPEG stream
    Continuously captures and yields JPEG frames
    """
    frame_time = 1.0 / FRAME_RATE

    while True:
        start_time = time.time()

        try:
            # Capture frame from Lepton
            frame = get_lepton_frame()

            if frame is not None:
                # Encode as JPEG
                ret, buffer = cv2.imencode('.jpg', frame,
                                          [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

                if ret:
                    frame_bytes = buffer.tobytes()

                    # Yield MJPEG frame with proper headers
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n'
                           b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n'
                           b'\r\n' + frame_bytes + b'\r\n')

            # Maintain frame rate
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_time - elapsed)
            time.sleep(sleep_time)

        except Exception as e:
            print(f"Error capturing frame: {e}")
            time.sleep(0.1)


@app.route('/stream.mjpg')
def stream():
    """MJPEG stream endpoint"""
    return Response(generate_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Simple test page"""
    return '''
    <html>
        <head><title>Lepton Stream</title></head>
        <body>
            <h1>Lepton Thermal Camera Stream</h1>
            <img src="/stream.mjpg" width="800" />
            <p>Stream URL: <code>http://&lt;pi-ip&gt;:5001/stream.mjpg</code></p>
        </body>
    </html>
    '''


if __name__ == '__main__':
    print("=" * 60)
    print("Lepton Camera MJPEG Stream Server")
    print("=" * 60)
    print(f"Starting server on port {STREAM_PORT}...")
    print(f"Frame rate: {FRAME_RATE} fps")
    print(f"JPEG quality: {JPEG_QUALITY}")
    print()
    print("IMPORTANT: Replace get_lepton_frame() with your actual camera code!")
    print()
    print("Access stream at:")
    print(f"  - Test page: http://localhost:{STREAM_PORT}/")
    print(f"  - MJPEG stream: http://localhost:{STREAM_PORT}/stream.mjpg")
    print("=" * 60)

    app.run(host='0.0.0.0', port=STREAM_PORT, threaded=True, debug=False)
