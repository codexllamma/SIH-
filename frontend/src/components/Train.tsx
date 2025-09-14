
import React from "react";
import { Group, Rect, Text } from "react-konva";

interface TrainProps {
  x: number;
  y: number;
  label: string;
  colour: string;
  signals: { id: number; x: number; y: number }[];
  signalStates: Record<number, boolean>;
  assignedSignals: number[];
  route: string[];
  tracksData: { id: string; points: number[] }[];
  speed: number;
  detectionDistance: number;
  debug?: boolean;
}

const Train: React.FC<TrainProps> = ({
  x,
  y,
  label,
  colour,
  signals,
  signalStates,
  assignedSignals,
  route,
  tracksData,
  speed,
  detectionDistance,
  debug = false,
}) => {
  const trainWidth = 40;
  const trainHeight = 20;

  const getTrainColor = () => {
    if (speed < 0.1) return "grey";
    return colour || "royalblue";
  };

  return (
    <Group x={x} y={y - trainHeight / 2}>
      {/* Train Body */}
      <Rect
        width={trainWidth}
        height={trainHeight}
        fill={getTrainColor()}
        cornerRadius={4}
        stroke="black"
        strokeWidth={1}
      />

      {/* Train Label */}
      <Text
        text={label}
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
          text={`Spd: ${speed.toFixed(1)} | Dist: ${detectionDistance}`}
          fontSize={10}
          fill="white"
          y={-15}
        />
      )}
    </Group>
  );
};

export default Train;
