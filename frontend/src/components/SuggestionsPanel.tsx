import { useEffect, useState } from "react";
import bus from "../utils/eventBus";

type Suggestion = {
  id: string;
  trainId: string;
  title: string;
  message: string;
  priority: string;
  outcome: string;
  delayImpact: number;
  confidence?: number;
};

export default function SuggestionsBox() {
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  const [socket, setSocket] = useState<WebSocket | null>(null);
  
  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8765"); // match websocket_server.py port
    setSocket(ws);

    ws.onopen = () => console.log("[SuggestionsBox] Connected to WebSocket");
    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);

        if (msg.type === "state_update" || msg.type === "delay_aware_update") {
          // Pull suggestions from dispatcher data
          const suggestionsFromServer = msg.data?.suggestions || [];
          setSuggestions(suggestionsFromServer);
        }
      } catch (e) {
        console.error("[SuggestionsBox] WS parse error:", e);
      }
    };
    ws.onclose = () => console.log("[SuggestionsBox] Disconnected");

    return () => ws.close();
  }, []);
    
    useEffect(() => {
    const fallback = setTimeout(() => {
      if (!suggestions || suggestions.length === 0) {
        setSuggestions([
          {
            id: "s1",
            trainId: "T1",
            title: "Predicted 12 min delay at J1",
            message: "Expected congestion at junction J1. Suggest reroute via Track 4",
            priority: "high",
            outcome: "reroute to Track 4",
            delayImpact: 12,
            confidence: 0.87,
          },
          {
            id: "s2",
            trainId: "T2",
            title: "Hold 5 min at Station B",
            message: "Temporary hold to reduce downstream conflicts.",
            priority: "medium",
            outcome: "hold 5 min",
            delayImpact: 5,
            confidence: 0.74,
          },
        ]);
      }
    }, 600);

    return () => clearTimeout(fallback);
  }, []);

  function sendUserAction(id: string, trainId: string, action: string, decision: "accept" | "reject") {
  // send to server if socket healthy
  if (socket && socket.readyState === WebSocket.OPEN) {
    try {
      socket.send(
        JSON.stringify({
          type: "user_action",
          data: { id, train_id: parseInt(trainId.replace("T", "")) || trainId, action, decision },
        })
      );
    } catch (e) {
      console.warn("[SuggestionsBox] ws send failed:", e);
    }
  }

  // local immediate UI update so demo feels instant
  setSuggestions((prev) =>
    prev.map((s) => (s.id === id ? { ...s, status: decision === "accept" ? "accepted" : "rejected" } : s))
  );

  // broadcast locally for the train animation to update immediately
  bus.dispatchEvent(
    new CustomEvent("applySuggestion", {
      detail: { id, trainId, action, decision },
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

        {suggestions.map((s) => (
          <div
            key={s.id}
            className="bg-gray-700 p-3 rounded-xl shadow-md flex flex-col"
          >
            {/* Train & action */}
            <div>
              <p className="text-lg font-semibold text-white">
                {s.trainId} → {s.title}
              </p>
              {s.confidence && (
                <p className="text-gray-400 text-xs">
                  Confidence: {(s.confidence * 100).toFixed(1)}%
                </p>
              )}
              <p className="text-gray-300 text-sm mt-1">{s.message}</p>
              <p className="text-gray-400 text-xs italic">
                Outcome: {s.outcome}
              </p>
            </div>

            {/* Buttons */}
            <div className="mt-3 flex space-x-2">
              <button
                className="flex-1 bg-green-600 hover:bg-green-700 text-white text-sm font-semibold py-2 rounded-lg transition-colors duration-200"
                onClick={() => sendUserAction(s.id, s.trainId, s.title, "accept")}
              >
                Accept
              </button>
              <button
                className="flex-1 bg-red-600 hover:bg-red-700 text-white text-sm font-semibold py-2 rounded-lg transition-colors duration-200"
                onClick={() => sendUserAction(s.id, s.trainId, s.title, "reject")}
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
