import React, { useState } from "react";
import { Group, Circle, Line, Rect } from "react-konva";

interface SignalProps {
  x: number;
  y: number;
}

const Signal: React.FC<SignalProps> = ({ x, y }) => {
  const [active, setActive] = useState(false); // true = red, false = green
  const radius = 10;
  const stickLength = 25;

  return (
    <Group
      x={x}
      y={y}
      onClick={() => setActive(!active)}
      cursor="pointer"
    >
      {/* Invisible hitbox (covers circle + stick) */}
      <Rect
        x={-radius - 5}
        y={-stickLength - radius - 5}
        width={radius * 2 + 10}
        height={stickLength + radius + 10}
        fill="transparent"
      />

      {/* push-pin stick */}
      <Line points={[0, 0, 0, -stickLength]} stroke="gray" strokeWidth={3} />

      {/* signal head */}
      <Circle
        x={0}
        y={-stickLength}
        radius={radius}
        fill={active ? "red" : "green"}
        stroke="black"
        strokeWidth={2}
      />
    </Group>
  );
};

export default Signal;
