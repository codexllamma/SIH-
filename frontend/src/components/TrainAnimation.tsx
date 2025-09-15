// src/components/TrainAnimation.tsx
import { useState, useEffect } from "react";
import { Stage, Layer, Line } from "react-konva";
import Platform from "./platform";
import Signal from "./Signal";
import Train from "./Train";
import bus from "../utils/eventBus";
import { trains as initialTrains } from "../data/traindata";

import {
  tracksData,
  platforms,
  signals,
  trains,
  trainRoutes,
} from "../data/traindata";

export default function TrainAnimation() {
  const [signalStates, setSignalStates] = useState<Record<number, boolean>>(
    Object.fromEntries(signals.map((s) => [s.id, false]))
  );
  const [localTrains, setLocalTrains] = useState(initialTrains);

  const toggleSignal = (id: number) => {
    setSignalStates((prev) => {
      const newState = !prev[id];
      if (newState) {
        setTimeout(() => {
          setSignalStates((p) => ({ ...p, [id]: false }));
        }, 5000);
      }
      return { ...prev, [id]: newState };
    });
  };

  useEffect(() => {
  const handler = (ev: Event) => {
    const custom = ev as CustomEvent;
    const { trainId, action, decision } = custom.detail || {};
    if (!trainId || decision !== "accept") return;

    setLocalTrains((prev) =>
      prev.map((t) => {
        if (t.id !== trainId) return t;
        const nt = { ...t };
        const a = (action || "").toLowerCase();
        if (a.includes("reroute") || a.includes("track")) {
          nt.y = (nt.y || 0) + 40; // visually shift to a different track
          nt.colour = "orange";
        } else if (a.includes("hold") || a.includes("delay")) {
          nt.delay = (nt.delay || 0) + 5;
          nt.colour = "red";
        } else if ((nt.delay || 0) + 0 < 0) {
          nt.colour = "lime";
        } else {
          nt.colour = "teal";
        }
        return nt;
      })
    );
  };

    bus.addEventListener("applySuggestion", handler as EventListener);
    return () => bus.removeEventListener("applySuggestion", handler as EventListener);
  }, []);


  return (
    <Stage width={883} height={384}>
      <Layer>
        {/* Render tracks */}
        {tracksData.map((track, index) => (
          <Line
            key={`track-${index}`}
            points={track.points}
            stroke="grey"
            strokeWidth={6}
            lineCap="round"
          />
        ))}

        {/* Render platforms */}
        {platforms.map((platform) => (
          <Platform
            key={platform.label}
            x={platform.x}
            y={platform.y}
            label={platform.label}
            colour={platform.colour}
          />
        ))}

        {/* Render signals */}
        {signals.map((s) => (
          <Signal
            key={s.id}
            x={s.x}
            y={s.y}
            active={signalStates[s.id]}
            toggle={() => toggleSignal(s.id)}
          />
        ))}

        {/* Render trains */}
        {localTrains.map((train) => (
          <Train
          key={train.id}
          x={train.x}
          y={train.y}
          label={train.label}
          colour={train.colour}
          signals={signals}
          signalStates={signalStates}
          assignedSignals={signals.map((s) => s.id)}
          route={trainRoutes[train.id]}
          tracksData={tracksData}
          speed={train.speed ?? 0}
          detectionDistance={40}
          debug={false}
        />

        ))}
      </Layer>
    </Stage>
  );
}
