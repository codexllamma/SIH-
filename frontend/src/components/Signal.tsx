
import React from "react";
import { Group, Circle, Line } from "react-konva";

interface SignalProps {
  x: number;
  y: number;
  active: boolean; 
  toggle: () => void;
}

const Signal: React.FC<SignalProps> = ({ x, y, active, toggle }) => {
  const radius = 8;
  const stickLength = 20;

  // Compute color based on active state
  const fillColor = active ? "red" : "green";

  return (
    <Group x={x} y={y} onClick={toggle}>
      {/* Stick */}
      <Line points={[0, 0, 0, -stickLength]} stroke="gray" strokeWidth={2} />
      {/* Light */}
      <Circle
        x={0}
        y={-stickLength}
        radius={radius}
        fill={fillColor}
        stroke="black"
        strokeWidth={1}
        shadowColor="black"
        shadowBlur={!active ? 10 : 0} // glow when red
        shadowOpacity={0.8}
      />
    </Group>
  );
};

export default Signal;
