import React, { useEffect, useState } from "react";
import { Group, Rect, Text } from "react-konva";

interface TrainProps {
  x: number;           // starting x
  y: number;           // fixed y (track level)
  label: string;       // train label
  colour?: string;     // train color
  speed?: number;      // pixels per frame
  trackLength?: number;// how far to go before resetting
}

const Train: React.FC<TrainProps> = ({ 
  x, 
  y, 
  label, 
  colour = "blue", 
  speed = 2, 
  trackLength = 850 
}) => {
  const [posX, setPosX] = useState(x);

  useEffect(() => {
    let anim: number;

    const move = () => {
      setPosX((prev) => {
        if (prev > trackLength) {
          return x; // reset to start
        }
        return prev + speed;
      });
      anim = requestAnimationFrame(move);
    };

    anim = requestAnimationFrame(move);
    return () => cancelAnimationFrame(anim);
  }, [x, speed, trackLength]);

  const trainWidth = 50;
  const trainHeight = 25;

  return (
    <Group x={posX} y={y-12}>
      {/* train body */}
      <Rect
        width={trainWidth}
        height={trainHeight}
        fill={colour}
        cornerRadius={4}
      />

      {/* label centered on train */}
      <Text
        text={label}
        fontSize={12}
        fontStyle="bold" 
        fill="black"
        width={trainWidth}
        height={trainHeight}
        align="center"
        verticalAlign="middle"
      />
    </Group>
  );
};

export default Train;
