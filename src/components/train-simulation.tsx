import React, { useState } from "react";
import { Stage, Layer, Line, Rect, Text } from "react-konva";

type TrainType = "freight" | "passenger" | "bullet";

interface Train {
  id: number;
  type: TrainType;
  x: number;
  y: number;
}

const trainColors: Record<TrainType, string> = {
  freight: "black",
  passenger: "blue",
  bullet: "green",
};

export default function TrainSimulation() {
  const [trains, setTrains] = useState<Train[]>([
    { id: 1, type: "freight", x: 50, y: 120 },
    { id: 2, type: "passenger", x: 200, y: 120 },
    { id: 3, type: "bullet", x: 350, y: 120 },
  ]);

  return (
    <Stage width={600} height={300} style={{ background: "white" }}>
      <Layer>
        {/* Track */}
        <Line
          points={[20, 150, 580, 150]} // x1, y1, x2, y2
          stroke="red"
          strokeWidth={4}
        />

        {/* Trains */}
        {trains.map((train) => (
          <React.Fragment key={train.id}>
            <Rect
              x={train.x}
              y={train.y - 30}
              width={60}
              height={30}
              fill={trainColors[train.type]}
              cornerRadius={4}
              draggable
            />
            <Text text={train.type} x={train.x} y={train.y} fontSize={12} />
          </React.Fragment>
        ))}
      </Layer>
    </Stage>
  );
}
