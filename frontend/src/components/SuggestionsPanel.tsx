import { useEffect, useState } from "react";

type Suggestion = {
  train_id: number;
  action: string;
  confidence: number;
};

export default function SuggestionsBox() {
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  const [socket, setSocket] = useState<WebSocket | null>(null);

  useEffect(() => {
    const ws = new WebSocket("ws://localhost:3001");
    setSocket(ws);

    ws.onopen = () => console.log("[SuggestionsBox] Connected to bridge");
    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === "suggestion") {
          setSuggestions(msg.data || []);
        }
      } catch (e) {
        console.error("[SuggestionsBox] WS parse error:", e);
      }
    };
    ws.onclose = () => console.log("[SuggestionsBox] Disconnected");

    return () => ws.close();
  }, []);

  function sendUserAction(
    train_id: number,
    action: string,
    decision: "accept" | "reject"
  ) {
    if (!socket || socket.readyState !== WebSocket.OPEN) return;
    socket.send(
      JSON.stringify({
        type: "user_action",
        data: { train_id, action, decision },
      })
    );
  }

  return (
    <div className="w-1/4 h-[80vh] bg-gray-800 p-4 rounded-xl shadow-lg border border-gray-700 flex flex-col">
      {/* Heading */}
      <h2 className="text-xl font-semibold text-teal-300 mb-4">AI Suggestions</h2>

      {/* Scrollable content */}
      <div className="flex-grow overflow-y-auto pr-2 space-y-3 custom-scrollbar">
        {suggestions.length === 0 && (
          <p className="text-gray-400 text-sm">No suggestions right now.</p>
        )}

        {suggestions.map((s, i) => (
          <div
            key={i}
            className="bg-gray-700 p-3 rounded-xl shadow-md flex flex-col"
          >
            {/* Train & action */}
            <div>
              <p className="text-lg font-semibold text-white">
                🚆 Train {s.train_id} → {s.action.replace("_", " ")}
              </p>
              <p className="text-gray-400 text-xs">
                Confidence: {(s.confidence * 100).toFixed(1)}%
              </p>
            </div>

            {/* Buttons */}
            <div className="mt-3 flex space-x-2">
        <button className="flex-1 bg-green-600 hover:bg-green-700 text-white text-sm font-semibold py-2 rounded-lg transition-colors duration-200"
        >
          Accept
        </button>
        <button className="flex-1 bg-red-600 hover:bg-red-700 text-white text-sm font-semibold py-2 rounded-lg transition-colors duration-200"
        >
          Dismiss
        </button>
      </div>
          </div>
        ))}
      </div>
    </div>
  );
}