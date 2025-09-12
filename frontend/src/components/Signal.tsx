// src/components/Signal.tsx
import React from "react";
import { Group, Circle, Line } from "react-konva";

interface SignalProps {
  x: number;
  y: number;
  state: 'red' | 'green';
}

const Signal: React.FC<SignalProps> = ({ x, y, state }) => {
  const radius = 8;
  const stickLength = 20;

  return (
    <Group x={x} y={y}>
      {/* Stick */}
      <Line points={[0, 0, 0, -stickLength]} stroke="gray" strokeWidth={2} />
      {/* Light */}
      <Circle
        x={0}
        y={-stickLength}
        radius={radius}
        fill={state}
        stroke="black"
        strokeWidth={1}
        shadowColor="black"
        shadowBlur={state === 'red' ? 10 : 0}
        shadowOpacity={0.8}
      />
    </Group>
  );
};

export default Signal;