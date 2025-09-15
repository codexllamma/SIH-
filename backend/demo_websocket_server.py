
import asyncio
import json
import time
import websockets

PORT = 8765

CONNECTED = set()

TRAINS = [
    {"id": "T1", "label": "T1", "colour": "aqua", "x": 0, "y": 80, "speed": 3, "delay": 0},
    {"id": "T2", "label": "T2", "colour": "white", "x": 0, "y": 200, "speed": 3, "delay": 0},
    {"id": "T3", "label": "T3", "colour": "yellow", "x": 0, "y": 320, "speed": 3, "delay": 0},
]

SUGGESTIONS = [
    {
        "id": "s1",
        "trainId": "T1",
        "title": "Predicted 12 min delay at J1",
        "message": "Expected congestion at junction J1. Suggest reroute via Track 4",
        "priority": "high",
        "outcome": "reroute to Track 4",
        "delayImpact": 12,
        "confidence": 0.87,
        "status": "pending",
    },
    {
        "id": "s2",
        "trainId": "T2",
        "title": "Hold 5 min at Station B",
        "message": "Temporary hold to reduce downstream conflicts.",
        "priority": "medium",
        "outcome": "hold 5 min",
        "delayImpact": 5,
        "confidence": 0.74,
        "status": "pending",
    },
    {
        "id": "s3",
        "trainId": "T3",
        "title": "Preemptive reroute to avoid cascade",
        "message": "Small reroute to avoid cascade — improves ETA by ~8 min.",
        "priority": "critical",
        "outcome": "reroute (benefit)",
        "delayImpact": -8,
        "confidence": 0.79,
        "status": "pending",
    },
]

async def broadcast_state():
    if not CONNECTED:
        return
    msg = {
        "type": "state_update",
        "data": {
            "timestamp": time.time(),
            "trains": TRAINS,
            "suggestions": SUGGESTIONS,
            "metrics": {"demo": True},
        },
    }
    text = json.dumps(msg)
    await asyncio.wait([ws.send(text) for ws in CONNECTED])

async def ticker():
    while True:
        # update anything dynamic (simulate motion)
        for t in TRAINS:
            t["x"] = (t.get("x", 0) + (t.get("speed", 1) * 2)) % 1000
        await broadcast_state()
        await asyncio.sleep(1.0)

async def handler(ws, path):
    print("[demo-ws] client connected")
    CONNECTED.add(ws)
    try:
        # Send immediate state on connect
        await broadcast_state()
        async for msg in ws:
            try:
                parsed = json.loads(msg)
            except Exception:
                continue
            # client sends { type: "user_action", data: {...} }
            if parsed.get("type") == "user_action":
                data = parsed.get("data", {})
                # map trainId parsing (frontend sends train_id as number sometimes)
                # Accept message format variations
                sid = data.get("id") or data.get("suggestion_id")
                train_id = data.get("train_id") or data.get("trainId") or data.get("train_id_str")
                action = data.get("action")
                decision = data.get("decision")
                # Find suggestion and mark it
                for s in SUGGESTIONS:
                    if s["id"] == sid:
                        s["status"] = "accepted" if decision == "accept" else "rejected"
                        s["handled_at"] = time.time()
                        s["handled_by"] = "demo"
                        # Apply a simple deterministic change to TRAINS when accepted
                        if decision == "accept":
                            for t in TRAINS:
                                if t["id"] == (train_id if isinstance(train_id, str) else (f"T{train_id}" if isinstance(train_id,int) else train_id)):
                                    # apply effects based on suggestion outcome text
                                    a = (action or s.get("outcome","")).lower()
                                    if "reroute" in a:
                                        t["y"] = t.get("y",0) + 40  # visually move to another track
                                        t["colour"] = "orange"
                                    elif "hold" in a or "delay" in a:
                                        t["delay"] = t.get("delay",0) + (s.get("delayImpact", 5))
                                        t["colour"] = "red"
                                    elif s.get("delayImpact",0) < 0:
                                        # beneficial reroute: green highlight
                                        t["colour"] = "lime"
            # after processing, rebroadcast update
            await broadcast_state()
    finally:
        CONNECTED.remove(ws)
        print("[demo-ws] client disconnected")

async def main():
    print(f"[demo-ws] starting on ws://0.0.0.0:{PORT}")
    server = await websockets.serve(handler, "0.0.0.0", PORT)
    # start ticker broadcaster
    tick = asyncio.create_task(ticker())
    await server.wait_closed()
    tick.cancel()

if __name__ == "__main__":
    asyncio.run(main())
