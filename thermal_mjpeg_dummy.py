from flask import Flask, Response
import time, io, numpy as np
from PIL import Image

app = Flask(__name__)

def generate():
    w, h = 160, 120
    t = 0.0
    while True:
        t += 0.05
        yy, xx = np.mgrid[0:h, 0:w]
        cx = int((np.sin(t)*0.4+0.5)*w)
        cy = int((np.cos(t)*0.4+0.5)*h)
        r2 = (xx-cx)**2 + (yy-cy)**2
        img = np.clip(255*np.exp(-r2/(2*20*20)), 0, 255).astype(np.uint8)  # faux hot spot
        # upscale + simple grayscale (you can tint if you want)
        img = Image.fromarray(img, mode="L").resize((800, 450), Image.NEAREST)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        frame = buf.getvalue()

        yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
               + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n")
        time.sleep(1/15)

@app.route("/stream.mjpg")
def stream():
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, threaded=True)

