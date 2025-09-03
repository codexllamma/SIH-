import React from "react";
import { Stage, Layer, Line } from "react-konva";
import { tracksData, junctions, platforms, signals, trains} from "../data/traindata";
import Platform from "./platform";
import Signal from "./Signal";
import Train from "./Train";

export default function TrainAnimation() {
  return (
    <>
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
              x={platform.x}
              y={platform.y}
              label={platform.label}
              colour={platform.colour}
            />
          ))}

        {signals.map((s) => (
          <>
          <Signal key={s.id} x={s.x} y={s.y} />
          </>
        ))}  

        {trains.map((train) => (
          <Train
            key={train.id}
            x={train.x}
            y={train.y}
            label={train.label}
            colour={train.colour}
          />
        ))}

        </Layer>
      </Stage>
    </>
  );
}