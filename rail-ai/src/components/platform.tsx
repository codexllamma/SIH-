import React from "react";
import { Stage, Layer, Line, Rect, Text } from "react-konva";

const Platform = ({ x, y, label, colour }) => {
  return (
    <>
      <Rect
        x={x}
        y={y}
        width={100}
        height={50}
        fill={colour || "lightblue"}
        cornerRadius={5}
      />
      <Text
        x={x + 24}
        y={y + 18}
        text={label}
        fontSize={18}
        fontStyle="bold"
      />
    </>
  );
};

export default Platform;