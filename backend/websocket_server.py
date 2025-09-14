# backend/websocket_server.py
import asyncio
import json
import logging
import websockets
from datetime import datetime

logging.basicConfig(level=logging.INFO)

HEARTBEAT_INTERVAL = 15  # seconds

def get_dummy_state():
    return {
        "type": "state_update",
        "data": {
            "timestamp": datetime.now().isoformat(),
            "step": 0,
            "trains": [
                {"id": 1, "position": [0, 0], "speed": 30, "delay": 0.0},
                {"id": 2, "position": [1, 1], "speed": 25, "delay": 1.2}
            ],
            "signals": {},
            "metrics": {
                "conflicts_prevented": 0,
                "delays_mitigated": 0,
                "on_time_arrivals": 0,
                "total_delays": 0.0,
                "system_efficiency": 1.0
            },
            "suggestions": [],
            "delay_predictions": {},
            "network": {"congestion": 0.1, "throughput": 0.0, "cascade_factor": 1.0}
        }
    }

connected_clients = set()

async def handler(websocket):
    connected_clients.add(websocket)
    logging.info(f"Client connected: {websocket.remote_address}")

    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                if data.get("type") == "ping":
                    # Respond to heartbeat ping
                    await websocket.send(json.dumps({"type": "pong"}))
                    continue

                logging.info(f"Received message: {data}")
                if data.get("type") == "user_action":
                    logging.info(f"User action received: {data['data']}")
                    await websocket.send(json.dumps({"type": "ack", "data": {"status": "ok"}}))
            except json.JSONDecodeError:
                logging.warning(f"Received invalid JSON: {message}")
    finally:
        connected_clients.remove(websocket)
        logging.info("Client disconnected")

async def broadcast_state():
    while True:
        if connected_clients:
            state = get_dummy_state()
            message = json.dumps(state)
            await asyncio.gather(*(client.send(message) for client in connected_clients))
        await asyncio.sleep(1)

async def main():
    async with websockets.serve(handler, "localhost", 8765, ping_interval=None):
        logging.info("WebSocket server started on ws://localhost:8765")
        await broadcast_state()

if __name__ == "__main__":
    asyncio.run(main())
