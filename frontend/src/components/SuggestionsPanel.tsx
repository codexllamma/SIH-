import { useEffect, useState } from "react";


type Suggestion = {
  train_id: number;
  action: string;
  confidence: number;
};

export default function SuggestionsPanel() {
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  const [socket, setSocket] = useState<WebSocket | null>(null);

  useEffect(() => {
    const ws = new WebSocket("ws://localhost:3001");
    setSocket(ws);

    ws.onopen = () => console.log("[SuggestionsPanel] Connected to bridge");
    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === "suggestion") {
          console.log("[Frontend] Received suggestions:", msg.data); 
          setSuggestions(msg.data || []);
        }
      } catch (e) {
        console.error("[SuggestionsPanel] WS parse error:", e);
      }
    };
    ws.onclose = () => console.log("[SuggestionsPanel] Disconnected");

    return () => ws.close();
  }, []);

  function sendUserAction(train_id: number, action: string, decision: "accept" | "reject") {
    if (!socket || socket.readyState !== WebSocket.OPEN) return;
    socket.send(
      JSON.stringify({
        type: "user_action",
        data: { train_id, action, decision },
      })
    );
  }

  return (
    <div className="bg-gray-900 text-white p-4 rounded-2xl shadow-lg w-full max-w-md mx-auto">
      <h2 className="text-xl font-bold mb-3">AI Suggestions</h2>
      {suggestions.length === 0 && (
        <p className="text-gray-400 text-sm">No suggestions right now.</p>
      )}
      <div className="space-y-3">
        {suggestions.map((s, i) => (
          <div
            key={i}
            className="bg-gray-800 p-3 rounded-xl shadow-md flex justify-between items-center"
          >
            <div>
              <p className="text-lg font-semibold">
                🚆 Train {s.train_id} → {s.action.replace("_", " ")}
              </p>
              <p className="text-gray-400 text-xs">
                Confidence: {(s.confidence * 100).toFixed(1)}%
              </p>
            </div>
            <div className="flex gap-2">
              <button
                className="bg-green-600 hover:bg-green-700 px-3 py-1 rounded-xl text-sm"
                onClick={() => sendUserAction(s.train_id, s.action, "accept")}
              >
                Accept
              </button>
              <button
                className="bg-red-600 hover:bg-red-700 px-3 py-1 rounded-xl text-sm"
                onClick={() => sendUserAction(s.train_id, s.action, "reject")}
              >
                Reject
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
