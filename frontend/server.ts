import WebSocket, { WebSocketServer } from "ws";

const PYTHON_WS_URL = "ws://localhost:8765";
const HEARTBEAT_INTERVAL = 10000; // 10s

let pythonSocket: WebSocket | null = null;
let reconnectTimeout: NodeJS.Timeout | null = null;

function connectToPython() {
  console.log("[Bridge] Connecting to Python backend...");
  pythonSocket = new WebSocket(PYTHON_WS_URL);

  pythonSocket.on("open", () => {
    console.log("[Bridge] Connected to Python backend");
    startHeartbeat();
  });

  pythonSocket.on("message", (msg) => {
    wss.clients.forEach((client) => {
      if (client.readyState === WebSocket.OPEN) {
        client.send(msg.toString());
      }
    });
  });

  pythonSocket.on("close", () => {
    console.log("[Bridge] Lost connection to Python, retrying...");
    scheduleReconnect();
  });

  pythonSocket.on("error", (err) => {
    console.error("[Bridge] Python WS error:", err);
    scheduleReconnect();
  });
}

function scheduleReconnect() {
  if (reconnectTimeout) return; // avoid multiple reconnect attempts
  reconnectTimeout = setTimeout(() => {
    reconnectTimeout = null;
    connectToPython();
  }, 3000);
}

function startHeartbeat() {
  setInterval(() => {
    if (pythonSocket && pythonSocket.readyState === WebSocket.OPEN) {
      pythonSocket.send(JSON.stringify({ type: "ping" }));
    }
  }, HEARTBEAT_INTERVAL);
}

const wss = new WebSocketServer({ port: 3001 });

wss.on("connection", (ws) => {
  console.log("[Bridge] React client connected");

  ws.on("message", (message) => {
    if (pythonSocket && pythonSocket.readyState === WebSocket.OPEN) {
      pythonSocket.send(message.toString());
    }
  });

  ws.on("close", () => console.log("[Bridge] React client disconnected"));
});

connectToPython();
console.log("[Bridge] Bridge server running on ws://localhost:3001");
