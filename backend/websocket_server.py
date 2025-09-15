# Paste into your Python websocket server (temporary debug mode)
import asyncio, json, time
import websockets
from model.dispatcher4 import create_delay_aware_dispatcher

dispatcher = create_delay_aware_dispatcher()
clients = set()

async def send_state():
    """Aggressive debug sender every 1s and guaranteed demo suggestions if none."""
    while True:
        try:
            # Try to get dispatcher state if function exists, otherwise build minimal state
            try:
                state = await dispatcher.get_comprehensive_status()
            except Exception:
                # lightweight fallback state
                state = {
                    "step": getattr(dispatcher, "env", None).current_step if getattr(dispatcher, "env", None) else 0,
                    "trains": [],
                    "suggestions": [],
                    "delay_predictions": getattr(dispatcher, "delay_predictions", {})
                }

            # Inject demo suggestions if none (guaranteed visible output)
            if not state.get("suggestions"):
                state["suggestions"] = [
                    {
                        "id": f"debug-{int(time.time())}",
                        "trainId": "T1",
                        "title": "DEBUG: demo reroute",
                        "message": "Bridge/Python debug suggestion to show in UI",
                        "priority": "high",
                        "outcome": "reroute to Track 2",
                        "delayImpact": -5,
                        "confidence": 0.75
                    }
                ]

            message = {"type": "state_update", "data": state}
            if clients:
                msgtxt = json.dumps(message)
                print("[Python] Broadcasting state (len suggestions):", len(state["suggestions"]))
                await asyncio.gather(*(c.send(msgtxt) for c in clients))
        except Exception as e:
            print("[Python] send_state error:", e)
        await asyncio.sleep(1)  # 1s for demo/debug
        

async def handler(websocket):
    clients.add(websocket)
    print("[Python] Client connected:", websocket.remote_address)
    # send immediate hello state
    try:
        hello = {"type": "hello", "data": {"msg": "welcome from python", "time": time.time()}}
        await websocket.send(json.dumps(hello))
    except Exception as e:
        print("[Python] hello send failed:", e)

    try:
        async for message in websocket:
            print("[Python] Received from client:", message)
            try:
                data = json.loads(message)
            except Exception:
                continue
            # echo back ack for ping/test
            if data.get("type") == "ping":
                await websocket.send(json.dumps({"type": "pong", "time": time.time()}))
            if data.get("type") == "user_action":
                # forward to dispatcher if available
                try:
                    result = await dispatcher.accept_action(data["data"])
                    await websocket.send(json.dumps({"type": "ack", "data": result}))
                except Exception as e:
                    await websocket.send(json.dumps({"type": "error", "message": str(e)}))
    finally:
        clients.remove(websocket)
        print("[Python] Client disconnected:", websocket.remote_address)

async def main():
    async with websockets.serve(handler, "localhost", 8765):
        print("[Python] WebSocket server started on ws://localhost:8765")
        await asyncio.gather(send_state())

if __name__ == "__main__":
    asyncio.run(main())
