# websocket_server.py (add or modify accordingly)

import asyncio
import json
import random
import websockets
from datetime import datetime

# --- Fake Train State for Testing ---
trains = [
    {"id": 1, "position": [0, 0], "speed": 30, "delay": 0},
    {"id": 2, "position": [1, 1], "speed": 25, "delay": 1.2}
]

clients = set()

async def send_state():
    """Periodically send fake state + suggestions to all clients."""
    while True:
        state = {
            "type": "state_update",
            "data": {
                "timestamp": datetime.now().isoformat(),
                "step": 0,
                "trains": trains,
                "signals": {},
                "metrics": {
                    "conflicts_prevented": 0,
                    "delays_mitigated": 0,
                    "on_time_arrivals": 0,
                    "total_delays": 0,
                    "system_efficiency": 1,
                },
                "suggestions": generate_fake_suggestions(),
                "delay_predictions": {},
                "network": {"congestion": 0.1, "throughput": 0, "cascade_factor": 1},
            },
        }
        if clients:
            msg = json.dumps(state)
            await asyncio.gather(*(client.send(msg) for client in clients))
        await asyncio.sleep(3)  # every 3 seconds

def generate_fake_suggestions():
    """Return random fake suggestions for trains."""
    actions = ["slow_down", "speed_up", "hold_at_signal"]
    return [
        {
            "train_id": random.choice(trains)["id"],
            "action": random.choice(actions),
            "confidence": round(random.uniform(0.6, 0.99), 2)
        }
        for _ in range(random.randint(1, 3))
    ]

async def handler(websocket):
    clients.add(websocket)
    print("Client connected:", websocket.remote_address)
    try:
        async for message in websocket:
            data = json.loads(message)
            print("[Server] Received:", data)

            if data.get("type") == "user_action":
                # Apply action (just log for now)
                print(f"User action received: {data}")
                # Later: update trains, call PPO, etc.

            await websocket.send(json.dumps({"type": "ack", "data": {"status": "ok"}}))
    finally:
        clients.remove(websocket)

async def main():
    async with websockets.serve(handler, "localhost", 8765):
        print("WebSocket server started on ws://localhost:8765")
        await send_state()

if __name__ == "__main__":
    asyncio.run(main())
