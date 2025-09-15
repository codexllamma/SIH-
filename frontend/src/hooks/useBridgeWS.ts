
import { useEffect, useRef, useState } from "react";

export function useBridgeWS() {
  const wsRef = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<any>(null);

  useEffect(() => {
    wsRef.current = new WebSocket("ws://localhost:3001");

    wsRef.current.onopen = () => {
      console.log("[Frontend] Connected to bridge");
      setConnected(true);
    };

    wsRef.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        console.log("[Frontend] Received:", data);
        setLastMessage(data);
        
        if (data.type === "suggestion") {
        (data.data || []).forEach((s: any) => {
          window.dispatchEvent(
            new CustomEvent("applySuggestion", {
              detail: {
                trainId: s.trainId,
                action: s.type,
                decision: "accept", // or let user trigger accept later
              },
            })
          );
        });
       }

      } catch (err) {
        console.error("[Frontend] Failed to parse message:", err);
      }
    };

    wsRef.current.onclose = () => {
      console.log("[Frontend] Disconnected from bridge");
      setConnected(false);
    };

    return () => wsRef.current?.close();
  }, []);

  function sendAction(data: Record<string, unknown>) {
    wsRef.current?.send(JSON.stringify({ type: "user_action", data }));
  }

  return { connected, lastMessage, sendAction };
}
