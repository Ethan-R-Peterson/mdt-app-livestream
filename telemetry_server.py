# telemetry_server.py
import asyncio, json, websockets, random, time, math
async def handler(ws):
    t0 = time.time()
    while True:
        t = time.time() - t0
        msg = {
            "time": time.time(),
            "alt": 30 + 2*math.sin(t/3),
            "gs": 5 + 0.5*math.cos(t/2),
            "batt": 92 - t*0.01,
            "lat": 42.293, "lon": -83.715,
            "yaw": (t*20)%360, "pitch": 2*math.sin(t/4), "roll": 3*math.cos(t/5),
            "dets": [{"x":0.2+0.1*math.sin(t/2), "y":0.2, "w":0.2, "h":0.15, "label":"car", "conf":0.87}]
        }
        await ws.send(json.dumps(msg))
        await asyncio.sleep(0.2)
async def main():
    async with websockets.serve(handler, "0.0.0.0", 8765):
        await asyncio.Future()
asyncio.run(main())
