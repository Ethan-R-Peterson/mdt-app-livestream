from flask import Flask, Response
import numpy as np, cv2, time, io

app = Flask(__name__)

def generate():
    w,h = 160,120
    t = 0.0
    while True:
        t += 0.05
        img = np.zeros((h,w), dtype=np.uint8)
        cx = int((np.sin(t)*0.4+0.5)*w)
        cy = int((np.cos(t)*0.4+0.5)*h)
        cv2.circle(img, (cx,cy), 20, 200, -1)
        color = cv2.applyColorMap(img, cv2.COLORMAP_INFERNO)
        ret, jpeg = cv2.imencode('.jpg', color, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        frame = jpeg.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(1/15)

@app.route('/stream.mjpg')
def stream():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
