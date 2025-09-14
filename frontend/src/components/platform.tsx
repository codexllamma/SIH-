
import { Rect, Text } from "react-konva";

interface PlatformProps {
  x: number;
  y: number;
  label: string;
  colour: string;
}

const Platform: React.FC<PlatformProps> = ({ x, y, label, colour }) => {
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