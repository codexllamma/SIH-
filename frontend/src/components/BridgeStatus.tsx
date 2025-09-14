
import { useBridgeWS } from "../hooks/useBridgeWS";

export default function BridgeStatus() {
  const { connected, lastMessage, sendAction } = useBridgeWS();

  return (
    <div className="p-4 text-white bg-gray-800 rounded-lg shadow-lg">
      <h2 className="text-lg font-bold">
        Bridge Status:{" "}
        <span className={connected ? "text-green-400" : "text-red-400"}>
          {connected ? "Connected" : "Disconnected"}
        </span>
      </h2>

      <pre className="text-xs bg-gray-900 p-2 mt-2 rounded">
        {JSON.stringify(lastMessage, null, 2)}
      </pre>

      <button
        onClick={() => sendAction({ trainId: 1, action: "HOLD" })}
        className="mt-3 px-3 py-1 bg-blue-600 hover:bg-blue-500 rounded"
      >
        Send Test Action
      </button>
    </div>
  );
}
