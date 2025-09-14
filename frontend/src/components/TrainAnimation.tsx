// src/components/TrainAnimation.tsx
import { useState } from "react";
import { Stage, Layer, Line } from "react-konva";
import Platform from "./platform";
import Signal from "./Signal";
import Train from "./Train";

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
        {trains.map((train) => (
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
