import React from "react";
import { Group, Circle, Line, Rect } from "react-konva";

interface SignalProps {
  x: number;
  y: number;
  active: boolean; // true = red
  toggle: () => void; // callback
}

const Signal: React.FC<SignalProps> = ({ x, y, active, toggle }) => {
  const radius = 10;
  const stickLength = 25;

  // actual circle center (not the stick base)
  const circleY = y - stickLength;

  return (
    <Group x={x} y={y} onClick={toggle} cursor="pointer">
      {/* Invisible hitbox */}
      <Rect
        x={-radius - 5}
        y={-stickLength - radius - 5}
        width={radius * 2 + 10}
        height={stickLength + radius + 10}
        fill="transparent"
      />

      {/* stick */}
      <Line points={[0, 0, 0, -stickLength]} stroke="gray" strokeWidth={3} />

      {/* light */}
      <Circle
        x={0}
        y={-stickLength}
        radius={radius}
        fill={active ? "red" : "green"}
        stroke="black"
        strokeWidth={2}
      />

      {/* helper rect: store circle coords in Konva attrs */}
      <Rect
        x={-1}
        y={-stickLength - 1}
        width={2}
        height={2}
        fill="transparent"
        name="signal-circle"
        listening={false}
      />
    </Group>
  );
};

export default Signal;
