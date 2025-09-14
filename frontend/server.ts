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
  const parsed = JSON.parse(msg.toString());

  if (parsed.type === "state_update") {
    // Log suggestions for debugging
    if (parsed.data?.suggestions?.length) {
      console.log("[Bridge] Suggestions from backend:", parsed.data.suggestions);

      // Send suggestions separately to frontend
      const suggestionMsg = JSON.stringify({
        type: "suggestion",
        data: parsed.data.suggestions,
      });

      wss.clients.forEach((client) => {
        if (client.readyState === WebSocket.OPEN) {
          client.send(suggestionMsg);
        }
      });
    }
  }

  // Still forward the original message (for TrainMap, metrics, etc.)
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
  if (reconnectTimeout) return; // Avoid multiple reconnect attempts
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

// --- WebSocket server for React frontend ---
const wss = new WebSocketServer({ port: 3001 });

wss.on("connection", (ws) => {
  console.log("[Bridge] React client connected");

  ws.on("message", (message) => {
    console.log("[Bridge] Forwarding action to Python:", message.toString());
    if (pythonSocket && pythonSocket.readyState === WebSocket.OPEN) {
      pythonSocket.send(message.toString());
    }
  });

  ws.on("close", () => console.log("[Bridge] React client disconnected"));
});

connectToPython();
console.log("[Bridge] Bridge server running on ws://localhost:3001");
