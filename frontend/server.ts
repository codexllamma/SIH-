import WebSocket, { WebSocketServer } from "ws";

const PYTHON_WS_URL = "ws://localhost:8765";
const BRIDGE_PORT = 8080; // for your UI client

let pythonSocket: WebSocket | null = null;
let uiWss: WebSocketServer | null = null;

function connectToPython() {
  pythonSocket = new WebSocket(PYTHON_WS_URL);

  pythonSocket.on("open", () => {
    console.log("[Bridge] Connected to Python backend.");
  });

  pythonSocket.on("message", (raw) => {
    try {
      const msg = JSON.parse(raw.toString());
      if (msg.type === "state_update") {
        if (msg.data?.suggestions) {
          console.log(
            "[Bridge] Suggestions received:",
            JSON.stringify(msg.data.suggestions, null, 2)
          );
        }
        // forward to all UI clients
        broadcastToUI(msg);
      } else {
        console.log("[Bridge] Incoming from Python:", msg);
      }
    } catch (err) {
      console.error("[Bridge] Failed to parse Python message:", err);
    }
  });

  pythonSocket.on("close", () => {
    console.warn("[Bridge] Python connection closed. Reconnecting in 3s...");
    setTimeout(connectToPython, 3000);
  });

  pythonSocket.on("error", (err) => {
    console.error("[Bridge] Python connection error:", err.message);
  });
}

function startBridge() {
  uiWss = new WebSocketServer({ port: BRIDGE_PORT });

  uiWss.on("connection", (ws) => {
    console.log("[Bridge] UI client connected.");

    ws.on("message", (raw) => {
      if (pythonSocket && pythonSocket.readyState === WebSocket.OPEN) {
        pythonSocket.send(raw.toString());
      }
    });

    ws.on("close", () => {
      console.log("[Bridge] UI client disconnected.");
    });
  });

  console.log(`[Bridge] UI WebSocket server started on ws://localhost:${BRIDGE_PORT}`);
}

function broadcastToUI(msg: any) {
  if (!uiWss) return;
  uiWss.clients.forEach((client) => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(JSON.stringify(msg));
    }
  });
}

// boot
connectToPython();
startBridge();
