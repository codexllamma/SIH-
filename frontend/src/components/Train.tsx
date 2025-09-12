// src/components/Train.tsx
import React, { useEffect, useRef, useState } from "react";
import { Group, Rect, Text } from "react-konva";

type Pt = { x: number; y: number };

interface SignalType {
  id: number;
  x: number;
  y: number;
}

interface TrackDef {
  id: string;
  points: number[]; // [x1,y1,x2,y2]
}

interface TrainProps {
  x: number;
  y: number;
  label: string;
  colour?: string;
  speed?: number; // px/sec
  signals: SignalType[];
  signalStates: Record<number, boolean>; // true = RED
  assignedSignals?: (string | number)[];
  detectionDistance?: number; // px
  verticalTolerance?: number;
  debug?: boolean;
  route?: string[]; // sequence of track ids, e.g. ["T1","T4","T2"]
  tracksData?: TrackDef[];
}

const EPS = 1e-6;

function dist(a: Pt, b: Pt) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

function projectPointOnSegment(p: Pt, a: Pt, b: Pt) {
  const vx = b.x - a.x;
  const vy = b.y - a.y;
  const len2 = vx * vx + vy * vy;
  if (len2 < EPS) return { point: { ...a }, t: 0 };
  const tRaw = ((p.x - a.x) * vx + (p.y - a.y) * vy) / len2;
  const t = Math.max(0, Math.min(1, tRaw));
  return { point: { x: a.x + vx * t, y: a.y + vy * t }, t };
}

function pointEquals(a: Pt | null, b: Pt | null) {
  if (!a || !b) return false;
  return Math.abs(a.x - b.x) < 0.5 && Math.abs(a.y - b.y) < 0.5;
}

const Train: React.FC<TrainProps> = ({
  x,
  y,
  label,
  colour = "blue",
  speed = 100, // px per second
  signals,
  signalStates,
  assignedSignals,
  detectionDistance = 40,
  debug = false,
  route = [],
  tracksData = [],
}) => {
  const trainWidth = 50;
  const trainHeight = 25;

  const polyRef = useRef<Pt[]>([]);
  const segIndexRef = useRef<number>(0);
  const posRef = useRef<Pt>({ x, y });
  const pausedRef = useRef<boolean>(false);
  const pausedForSignalRef = useRef<number | null>(null);

  const [, setTick] = useState(0);
  const [pausedState, setPausedState] = useState(false);

  const rafRef = useRef<number | null>(null);
  const lastTsRef = useRef<number | null>(null);

  // Build polyline from route & tracks
  const buildPolyline = () => {
    if (
      !route ||
      route.length === 0 ||
      !tracksData ||
      tracksData.length === 0
    ) {
      polyRef.current = [
        { x, y },
        { x: x + 1000, y },
      ];
      segIndexRef.current = 0;
      posRef.current = { x, y };
      return;
    }

    const getTrack = (id: string) =>
      tracksData.find((t) => t.id === id) || null;

    const built: Pt[] = [];
    let prevPoint: Pt = { x, y };
    built.push(prevPoint);

    for (let i = 0; i < route.length; i++) {
      const tid = route[i];
      const track = getTrack(tid);
      if (!track) {
        if (debug) console.warn(`[Train ${label}] track ${tid} not found`);
        continue;
      }
      const [ax, ay, bx, by] = track.points;
      const a: Pt = { x: ax, y: ay };
      const b: Pt = { x: bx, y: by };

      const projEntry = projectPointOnSegment(prevPoint, a, b);
      const entry = projEntry.point;

      let exit: Pt = b;
      if (i < route.length - 1) {
        const nextTrack = getTrack(route[i + 1]);
        if (nextTrack) {
          const [nax, nay, nbx, nby] = nextTrack.points;
          const n1: Pt = { x: nax, y: nay };
          const n2: Pt = { x: nbx, y: nby };

          const p1 = projectPointOnSegment(n1, a, b);
          const p2 = projectPointOnSegment(n2, a, b);

          const candidates = [
            { proj: p1, orig: n1 },
            { proj: p2, orig: n2 },
          ];

          candidates.sort((cA, cB) => {
            const dA = dist(cA.proj.point, cA.orig);
            const dB = dist(cB.proj.point, cB.orig);
            if (Math.abs(dA - dB) > EPS) return dA - dB;
            return cA.proj.t - cB.proj.t;
          });

          const best = candidates[0];
          if (dist(best.proj.point, best.orig) <= 5) {
            if (best.proj.t >= projEntry.t - 1e-3) {
              exit = best.proj.point;
            } else {
              const dA = dist(entry, a);
              const dB = dist(entry, b);
              exit = dA > dB ? a : b;
            }
          } else {
            const dA = dist(entry, a);
            const dB = dist(entry, b);
            exit = dA > dB ? a : b;
          }
        } else {
          const dA = dist(entry, a);
          const dB = dist(entry, b);
          exit = dA > dB ? a : b;
        }
      } else {
        const dA = dist(entry, a);
        const dB = dist(entry, b);
        exit = dA > dB ? a : b;
      }

      if (!pointEquals(built[built.length - 1], entry)) built.push(entry);
      if (!pointEquals(built[built.length - 1], exit)) built.push(exit);
      prevPoint = exit;
    }

    const compact: Pt[] = [];
    for (const p of built) {
      if (compact.length === 0 || !pointEquals(compact[compact.length - 1], p))
        compact.push(p);
    }

    if (debug) {
      console.log(`[Train ${label}] built polyline:`, compact);
    }

    polyRef.current = compact;
    segIndexRef.current = 0;
    posRef.current = compact[0] || { x, y };
  };

  useEffect(() => {
    buildPolyline();
    setTick((t) => t + 1);
  }, [route?.join("|"), JSON.stringify(tracksData), x, y]);

  function currentSegmentDir(): Pt {
    const poly = polyRef.current;
    const idx = segIndexRef.current;
    if (!poly || poly.length < 2) return { x: 1, y: 0 };
    const a = poly[Math.max(0, Math.min(idx, poly.length - 2))];
    const b = poly[Math.max(1, Math.min(idx + 1, poly.length - 1))];
    const vx = b.x - a.x;
    const vy = b.y - a.y;
    const len = Math.sqrt(vx * vx + vy * vy) || 1;
    return { x: vx / len, y: vy / len };
  }

  function detectRedSignalAhead(): number | null {
    const poly = polyRef.current;
    if (!poly || poly.length < 2) return null;
    const dir = currentSegmentDir();
    const frontOffset = 0.5 * trainWidth;
    const front: Pt = {
      x: posRef.current.x + dir.x * frontOffset,
      y: posRef.current.y + dir.y * frontOffset,
    };

    for (const s of signals) {
      if (
        assignedSignals &&
        !assignedSignals.map(String).includes(String(s.id))
      )
        continue;
      const vx = s.x - front.x;
      const vy = s.y - front.y;
      const dot = vx * dir.x + vy * dir.y;
      if (dot < 0) continue;
      const d = Math.sqrt(vx * vx + vy * vy);
      if (d <= detectionDistance && signalStates[s.id]) {
        return s.id;
      }
    }
    return null;
  }

  // main loop (RAF)
  useEffect(() => {
    function step(ts: number) {
      if (lastTsRef.current == null) lastTsRef.current = ts;
      const deltaMs = ts - (lastTsRef.current || ts);
      lastTsRef.current = ts;

      const redSig = detectRedSignalAhead();
      if (redSig != null) {
        if (!pausedRef.current) {
          pausedRef.current = true;
          pausedForSignalRef.current = redSig;
          setPausedState(true);
          if (debug) {
            console.log(`[Train ${label}] STOPPED for signal ${redSig}`);
          }
        }
      } else {
        if (pausedRef.current) {
          pausedRef.current = false;
          pausedForSignalRef.current = null;
          setPausedState(false);
          if (debug) {
            console.log(`[Train ${label}] RESUMED`);
          }
        }
      }

      if (!pausedRef.current) {
        const dRemaining = (speed * deltaMs) / 1000;
        if (dRemaining > 0) {
          let remaining = dRemaining;
          const poly = polyRef.current;

          // Check for end of route BEFORE attempting to move
          if (segIndexRef.current >= poly.length - 1) {
            posRef.current = poly[0];
            segIndexRef.current = 0;
            setTick((t) => t + 1); // Force a re-render
            rafRef.current = requestAnimationFrame(step);
            return;
          }

          while (remaining > 0 && segIndexRef.current < poly.length - 1) {
            const a = poly[segIndexRef.current];
            const b = poly[segIndexRef.current + 1];
            const toEnd = dist(posRef.current, b);

            if (remaining >= toEnd) {
              remaining -= toEnd;
              posRef.current = b;
              segIndexRef.current += 1;
            } else {
              const vx = b.x - a.x;
              const vy = b.y - a.y;
              const segLen = dist(a, b) || 1;
              const ux = vx / segLen;
              const uy = vy / segLen;
              posRef.current = {
                x: posRef.current.x + ux * remaining,
                y: posRef.current.y + uy * remaining,
              };
              remaining = 0;
            }
          }
        }
      }

      setTick((t) => t + 1);
      rafRef.current = requestAnimationFrame(step);
    }

    rafRef.current = requestAnimationFrame(step);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
      lastTsRef.current = null;
    };
  }, [
    speed,
    JSON.stringify(route),
    JSON.stringify(tracksData),
    JSON.stringify(signals),
    detectionDistance,
    label,
    debug,
    assignedSignals,
  ]);

  const pos = posRef.current;
  return (
    <Group x={pos.x} y={pos.y - trainHeight / 2}>
      <Rect
        width={trainWidth}
        height={trainHeight}
        fill={colour}
        cornerRadius={4}
      />
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
      {debug && (
        <Text
          text={pausedState ? "PAUSED" : ""}
          fontSize={10}
          fill="white"
          y={-18}
          x={-10}
        />
      )}
    </Group>
  );
};

export default Train;
