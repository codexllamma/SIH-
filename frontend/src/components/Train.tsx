// src/components/Train.tsx
import React from "react";
import { Group, Rect, Text } from "react-konva";
import type { TrainState } from "../types";

interface TrainProps {
  train: TrainState; // Pass the entire train state object
  debug?: boolean;
}

const Train: React.FC<TrainProps> = ({ train, debug = false }) => {
  const trainWidth = 40;
  const trainHeight = 20;

  // The color changes based on its state
  const getTrainColor = () => {
    if (train.haltTime > 0) return 'orange'; // Halted at station
    if (train.speed < 0.1) return 'grey'; // Stopped
    return 'royalblue';
  };

  return (
    <Group x={train.x} y={train.y - trainHeight / 2}>
      {/* Collision Risk Indicator (a red halo) */}
      {train.collisionRisk > 0.3 && (
        <Rect
          width={trainWidth + 10}
          height={trainHeight + 10}
          x={-5}
          y={-5}
          fill="red"
          cornerRadius={8}
          opacity={train.collisionRisk * 0.7} // Becomes more visible with higher risk
          shadowColor="red"
          shadowBlur={15}
        />
      )}
      
      {/* Train Body */}
      <Rect
        width={trainWidth}
        height={trainHeight}
        fill={getTrainColor()}
        cornerRadius={4}
        stroke="black"
        strokeWidth={1}
      />
      
      {/* Label */}
      <Text
        text={train.label}
        fontSize={12}
        fontStyle="bold"
        fill="white"
        width={trainWidth}
        height={trainHeight}
        align="center"
        verticalAlign="middle"
      />
      
      {/* Debug Info */}
      {debug && (
        <Text
          text={`Spd: ${train.speed.toFixed(1)} | Trg: ${train.targetSpeed}`}
          fontSize={10}
          fill="white"
          y={-15}
        />
      )}
    </Group>
  );
};

export default Train;