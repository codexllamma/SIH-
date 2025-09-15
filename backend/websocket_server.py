import asyncio
import json
import websockets
from model.dispatcher4 import create_delay_aware_dispatcher

dispatcher = create_delay_aware_dispatcher()
clients = set()

async def send_state():
    """Periodically send real state + suggestions from dispatcher to all clients."""
    while True:
        try:
            state = await dispatcher.get_comprehensive_status()
            message = {
                "type": "state_update",
                "data": state
            }
            if clients:
                msg = json.dumps(message)
                await asyncio.gather(*(client.send(msg) for client in clients))
        except Exception as e:
            print(f"[Server] Failed to get/send state: {e}")
        await asyncio.sleep(10)

async def handler(websocket):
    clients.add(websocket)
    print("Client connected:", websocket.remote_address)
    try:
        async for message in websocket:
            data = json.loads(message)
            print("[Server] Received:", data)

            if data.get("type") == "user_action":
                # Forward accepted actions to dispatcher
                result = await dispatcher.accept_action(data["data"])
                await websocket.send(json.dumps({"type": "ack", "data": result}))
    finally:
        clients.remove(websocket)

async def main():
    async with websockets.serve(handler, "localhost", 8765):
        print("WebSocket server started on ws://localhost:8765")
        # Run dispatcher loop and send_state concurrently
        await asyncio.gather(
            dispatcher_loop(),
            send_state()
        )

async def dispatcher_loop():
    update_interval = 1.0 / dispatcher.config.get("update_rate", 2)
    while True:
        try:
            await dispatcher.step()
            await asyncio.sleep(update_interval)
        except Exception as e:
            print(f"[Server] Dispatcher loop error: {e}")
            await asyncio.sleep(10)

if __name__ == "__main__":
    asyncio.run(main())
