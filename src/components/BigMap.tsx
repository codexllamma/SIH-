import React, { useEffect, useRef, useState } from "react";
import { Stage, Layer, Line, Circle, Rect, Text } from "react-konva";

/* ---------------------- Type Definitions ---------------------- */
interface Segment {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  length: number;
}

interface ComputedSegments {
  segments: Segment[];
  total: number;
}

interface PosAndAngle {
  x: number;
  y: number;
  angle: number;
}

interface Track {
  id: string;
  points: number[];
}

interface Junction {
  id: string;
  x: number;
  y: number;
  connectedTracks: string[];
}

interface Train {
  id: string;
  trackId: string;
  progress: number;
  speed: number;
  color: string;
}

/* ---------------------- Helper: track math ---------------------- */
// compute segments + total length for a polyline points array
function computeSegments(points: number[]): ComputedSegments {
  const segments: Segment[] = [];
  let total = 0;
  for (let i = 0; i < points.length - 2; i += 2) {
    const x1 = points[i],
      y1 = points[i + 1],
      x2 = points[i + 2],
      y2 = points[i + 3];
    const dx = x2 - x1,
      dy = y2 - y1;
    const length = Math.hypot(dx, dy);
    segments.push({ x1, y1, x2, y2, length });
    total += length;
  }
  return { segments, total };
}

// get position+angle on polyline (points) given progress 0..1
function getPosOnPoints(points: number[], progress: number): PosAndAngle {
  const { segments, total } = computeSegments(points);
  let dist = Math.max(0, Math.min(progress, 0.999999)) * total;
  for (let seg of segments) {
    if (dist <= seg.length) {
      const ratio = seg.length === 0 ? 0 : dist / seg.length;
      const x = seg.x1 + (seg.x2 - seg.x1) * ratio;
      const y = seg.y1 + (seg.y2 - seg.y1) * ratio;
      const angle =
        Math.atan2(seg.y2 - seg.y1, seg.x2 - seg.x1) * (180 / Math.PI);
      return { x, y, angle };
    }
    dist -= seg.length;
  }
  // fallback to last point
  const last = segments[segments.length - 1];
  return {
    x: last.x2,
    y: last.y2,
    angle: Math.atan2(last.y2 - last.y1, last.x2 - last.x1) * (180 / Math.PI),
  };
}

/* -------------------------- Data setup -------------------------- */
const tracksData: Track[] = [
  { id: "T1", points: [0, 80, 760, 80] },
  { id: "T2", points: [0, 160, 760, 160] },
  { id: "T3", points: [0, 260, 760, 260] },
  { id: "T4", points: [400, 80, 520, 120, 760, 160] },
];

const junctions: Junction[] = [
  { id: "J1", x: 220, y: 80, connectedTracks: ["T1", "T2"] },
  { id: "J2", x: 400, y: 80, connectedTracks: ["T1", "T4"] },
  { id: "J3", x: 300, y: 160, connectedTracks: ["T2", "T4"] },
  { id: "J4", x: 220, y: 260, connectedTracks: ["T2", "T3"] },
  { id: "J5", x: 420, y: 260, connectedTracks: ["T3", "T4"] },
];

const initialTrains: Train[] = [
  {
    id: "A",
    trackId: "T1",
    progress: 0.05,
    speed: 0.08,
    color: "#00bfff",
  },
  {
    id: "B",
    trackId: "T2",
    progress: 0.2,
    speed: 0.05,
    color: "#ff7f50",
  },
  {
    id: "C",
    trackId: "T3",
    progress: 0.5,
    speed: 0.06,
    color: "#ad8cf7",
  },
  {
    id: "D",
    trackId: "T4",
    progress: 0.15,
    speed: 0.07,
    color: "#ffd24d",
  },
];

/* -------------------------- Component --------------------------- */
export default function BigMap() {
  const trackMap = useRef<Map<string, ComputedSegments>>(new Map());
  useEffect(() => {
    for (let t of tracksData) {
      trackMap.current.set(t.id, computeSegments(t.points));
    }
  }, []);

  const [trains, setTrains] = useState<Train[]>(initialTrains);
  const [signals, setSignals] = useState<Record<string, boolean>>(() =>
    junctions.reduce((acc, j) => ({ ...acc, [j.id]: false }), {})
  );

  const trainsRef = useRef(trains);
  const signalsRef = useRef(signals);
  useEffect(() => {
    trainsRef.current = trains;
  }, [trains]);
  useEffect(() => {
    signalsRef.current = signals;
  }, [signals]);

  const getTrack = (id: string): Track | undefined =>
    tracksData.find((t) => t.id === id);

  function trainWorldPos(train: Train): PosAndAngle | null {
    const track = getTrack(train.trackId);
    if (!track) return null;
    return getPosOnPoints(track.points, train.progress);
  }

  const lastRef = useRef(performance.now());

  useEffect(() => {
    let rafId = 0;
    const occupyRadius = 18;
    const stopDistance = 38;
    const spawnChance = 0.03;

    function step(now: number) {
      const dt = (now - lastRef.current) / 1000;
      lastRef.current = now;

      const occupiedJunctions = new Set<string>();
      trainsRef.current.forEach((t) => {
        const pos = trainWorldPos(t);
        if (pos) {
          junctions.forEach((j) => {
            if (Math.hypot(pos.x - j.x, pos.y - j.y) <= occupyRadius) {
              occupiedJunctions.add(j.id);
            }
          });
        }
      });

      const newSignals: Record<string, boolean> = {};
      junctions.forEach((j) => {
        newSignals[j.id] = occupiedJunctions.has(j.id);
      });
      setSignals(newSignals);

      const updatedTrains = trainsRef.current
        .map((t) => {
          const track = getTrack(t.trackId);
          if (!track) return null;
          const pos = getPosOnPoints(track.points, t.progress);

          let shouldStop = false;
          for (const j of junctions) {
            if (j.connectedTracks.includes(t.trackId)) {
              const d = Math.hypot(pos.x - j.x, pos.y - j.y);
              if (signalsRef.current[j.id] && d < stopDistance) {
                shouldStop = true;
                break;
              }
            }
          }

          if (shouldStop) {
            return t;
          } else {
            const newProgress = t.progress + t.speed * dt;
            if (newProgress >= 1) {
              return null;
            }
            return { ...t, progress: newProgress };
          }
        })
        .filter((t): t is Train => t !== null);

      setTrains(updatedTrains);

      if (Math.random() < dt * spawnChance && updatedTrains.length < 10) {
        const tracks = ["T1", "T2", "T3", "T4"];
        const trackId = tracks[Math.floor(Math.random() * tracks.length)];
        setTrains((prev) => [
          ...prev,
          {
            id: Math.random().toString(36).slice(2, 7),
            trackId,
            progress: 0,
            speed: 0.05 + Math.random() * 0.05,
            color: ["#00bfff", "#ff7f50", "#ad8cf7", "#ffd24d"][
              Math.floor(Math.random() * 4)
            ],
          },
        ]);
      }

      rafId = requestAnimationFrame(step);
    }

    rafId = requestAnimationFrame(step);

    return () => cancelAnimationFrame(rafId);
  }, []);

  /* ---------------------------- Render ---------------------------- */
  return (
    <div style={{ padding: 12 }}>
      <div style={{ display: "flex" }}>
        <Stage
          width={760}
          height={360}
          style={{ background: "#f8fbfd", borderRadius: 6 }}
        >
          <Layer>
            {tracksData.map((t) => (
              <Line
                key={t.id}
                points={t.points}
                stroke="#cfcfcf"
                strokeWidth={6}
                lineCap="round"
              />
            ))}

            {junctions.map((j) => (
              <React.Fragment key={j.id}>
                <Circle
                  x={j.x}
                  y={j.y}
                  radius={10}
                  fill={signals[j.id] ? "#d33" : "#2eb82e"}
                  stroke="#333"
                  strokeWidth={1}
                />
                <Text
                  text={j.id}
                  x={j.x + 12}
                  y={j.y - 8}
                  fontSize={12}
                  fill="#222"
                />
              </React.Fragment>
            ))}

            {trains.map((t) => {
              const track = getTrack(t.trackId);
              if (!track) return null;
              const pos = getPosOnPoints(track.points, t.progress);
              const angle = pos.angle || 0;
              return (
                <React.Fragment key={t.id}>
                  <Rect
                    x={pos.x}
                    y={pos.y}
                    width={30}
                    height={14}
                    fill={t.color}
                    offsetX={15}
                    offsetY={7}
                    rotation={angle}
                    cornerRadius={3}
                    shadowBlur={2}
                  />
                  <Text
                    text={t.id}
                    x={pos.x - 6}
                    y={pos.y - 28}
                    fontSize={12}
                    fill="#222"
                  />
                </React.Fragment>
              );
            })}
          </Layer>
        </Stage>
      </div>

      <div style={{ color: "#ddd", marginTop: 12 }}>
        <small>
          Trains stop before red junctions. Speeds vary per train. Signals only
          at junctions — not along every track.
        </small>
      </div>
    </div>
  );
}
