#!/usr/bin/env python3
"""
Telemetry WebSocket Server for Raspberry Pi
Sends real-time telemetry and object detection data to web UI
Connect at: ws://<pi-ip>:8765
"""

import asyncio
import json
import websockets
import time
import math

# Configuration
WS_PORT = 8765
UPDATE_RATE = 5  # Hz (updates per second)

# Global state for detection results
# Your detection code should update this
current_detections = []
current_telemetry = {
    "lat": 0.0,
    "lon": 0.0,
    "alt": 0.0,
    "gs": 0.0,  # ground speed
    "batt": 100.0,
    "yaw": 0.0,
    "pitch": 0.0,
    "roll": 0.0
}


def get_detection_results():
    """
    Get current object detection results

    Replace this with your actual detection code.
    Should return a list of detections in this format:

    Returns:
    --------
    list of dict: [
        {
            "x": 0.25,      # normalized x (0-1) - center of box
            "y": 0.30,      # normalized y (0-1) - center of box
            "w": 0.20,      # normalized width (0-1)
            "h": 0.15,      # normalized height (0-1)
            "label": "person",  # class name
            "conf": 0.92    # confidence score (0-1)
        },
        ...
    ]

    Example integration with YOLOv5:
    ---------------------------------
    # results = model(frame)  # Your YOLO model inference
    # detections = []
    # for *box, conf, cls in results.xyxy[0]:
    #     x1, y1, x2, y2 = box
    #     img_h, img_w = frame.shape[:2]
    #     detections.append({
    #         "x": ((x1 + x2) / 2) / img_w,
    #         "y": ((y1 + y2) / 2) / img_h,
    #         "w": (x2 - x1) / img_w,
    #         "h": (y2 - y1) / img_h,
    #         "label": model.names[int(cls)],
    #         "conf": float(conf)
    #     })
    # return detections
    """

    # Return the global detection list updated by your detection thread
    return current_detections


def get_telemetry_data():
    """
    Get current telemetry data

    Replace with actual sensor readings if available:
    - GPS module for lat/lon
    - Barometer for altitude
    - IMU for yaw/pitch/roll
    - Battery monitor for battery level

    Returns:
    --------
    dict: Telemetry data
    """

    # Return the global telemetry updated by your sensor reading code
    return current_telemetry.copy()


async def telemetry_handler(websocket):
    """
    WebSocket handler - sends telemetry updates to connected clients
    """
    client_addr = websocket.remote_address
    print(f"[+] Client connected: {client_addr}")

    try:
        while True:
            # Get current detections and telemetry
            detections = get_detection_results()
            telemetry = get_telemetry_data()

            # Build message
            message = {
                "time": time.time(),
                "lat": telemetry["lat"],
                "lon": telemetry["lon"],
                "alt": telemetry["alt"],
                "gs": telemetry["gs"],
                "batt": telemetry["batt"],
                "yaw": telemetry["yaw"],
                "pitch": telemetry["pitch"],
                "roll": telemetry["roll"],
                "dets": detections
            }

            # Send to client
            await websocket.send(json.dumps(message))

            # Wait for next update
            await asyncio.sleep(1.0 / UPDATE_RATE)

    except websockets.exceptions.ConnectionClosed:
        print(f"[-] Client disconnected: {client_addr}")
    except Exception as e:
        print(f"[!] Error in telemetry handler: {e}")


async def main():
    """Start WebSocket server"""
    print("=" * 60)
    print("Telemetry WebSocket Server")
    print("=" * 60)
    print(f"Starting server on port {WS_PORT}...")
    print(f"Update rate: {UPDATE_RATE} Hz")
    print()
    print("IMPORTANT: Update get_detection_results() and get_telemetry_data()")
    print("           with your actual detection/sensor code!")
    print()
    print(f"WebSocket URL: ws://localhost:{WS_PORT}")
    print("=" * 60)

    async with websockets.serve(telemetry_handler, "0.0.0.0", WS_PORT):
        await asyncio.Future()  # run forever


# ============================================================================
# Example: How to update detections from your detection thread
# ============================================================================

def example_detection_loop():
    """
    Example of how to update detections from your detection code
    Run this in a separate thread alongside the WebSocket server

    Example:
    --------
    import threading
    detection_thread = threading.Thread(target=example_detection_loop, daemon=True)
    detection_thread.start()
    """
    global current_detections

    # Your camera/detection setup
    # cap = cv2.VideoCapture(0)
    # model = load_your_model()

    while True:
        # Capture frame
        # ret, frame = cap.read()

        # Run detection
        # results = model(frame)

        # Parse results into required format
        # current_detections = parse_detections(results, frame.shape)

        # Example dummy data (replace with actual)
        current_detections = [
            {
                "x": 0.5,
                "y": 0.5,
                "w": 0.2,
                "h": 0.15,
                "label": "person",
                "conf": 0.92
            }
        ]

        time.sleep(0.1)  # Adjust based on your detection speed


def example_telemetry_loop():
    """
    Example of how to update telemetry from sensor readings
    Run this in a separate thread alongside the WebSocket server
    """
    global current_telemetry

    while True:
        # Read sensors
        # gps_data = read_gps()
        # imu_data = read_imu()
        # battery = read_battery()

        # Update global telemetry
        # current_telemetry["lat"] = gps_data.latitude
        # current_telemetry["lon"] = gps_data.longitude
        # current_telemetry["alt"] = gps_data.altitude
        # current_telemetry["yaw"] = imu_data.yaw
        # current_telemetry["pitch"] = imu_data.pitch
        # current_telemetry["roll"] = imu_data.roll
        # current_telemetry["batt"] = battery.percentage

        # Example dummy data (replace with actual)
        current_telemetry = {
            "lat": 42.293,
            "lon": -83.715,
            "alt": 30.0,
            "gs": 5.0,
            "batt": 95.0,
            "yaw": 0.0,
            "pitch": 0.0,
            "roll": 0.0
        }

        time.sleep(0.2)


# ============================================================================
# Main entry point
# ============================================================================

if __name__ == '__main__':
    # Optional: Start detection/telemetry update threads here
    # import threading
    # threading.Thread(target=example_detection_loop, daemon=True).start()
    # threading.Thread(target=example_telemetry_loop, daemon=True).start()

    # Start WebSocket server
    asyncio.run(main())
