import React, { useEffect, useRef, useState } from "react";
import { Stage, Layer, Line, Circle, Rect, Text } from "react-konva";

/* ---------------------- Helper: track math ---------------------- */
// compute segments + total length for a polyline points array
function computeSegments(int: points) {
  const segments = [];
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
function getPosOnPoints(points, progress) {
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
const tracksData = [
  // top line
  { id: "T1", points: [0, 80, 760, 80] },
  // middle line
  { id: "T2", points: [0, 160, 760, 160] },
  // bottom line
  { id: "T3", points: [0, 260, 760, 260] },
  // branch
  { id: "T4", points: [400, 80, 520, 120, 760, 160] },
];

// junctions (only these have signals). connectedTracks lists IDs of tracks meeting here
const junctions = [
  { id: "J1", x: 220, y: 80, connectedTracks: ["T1", "T2"] },
  { id: "J2", x: 400, y: 80, connectedTracks: ["T1", "T4"] },
  { id: "J3", x: 300, y: 160, connectedTracks: ["T2", "T4"] },
  { id: "J4", x: 220, y: 260, connectedTracks: ["T2", "T3"] },
  { id: "J5", x: 420, y: 260, connectedTracks: ["T3", "T4"] }, // optional overlap
];

// trains: each train is bound to a track (by trackId) and has a progress (0..1) and speed (fraction per second)
const initialTrains = [
  { id: "A", trackId: "T1", progress: 0.05, speed: 0.08, color: "#00bfff" },
  { id: "B", trackId: "T2", progress: 0.2, speed: 0.05, color: "#ff7f50" },
  { id: "C", trackId: "T3", progress: 0.5, speed: 0.06, color: "#ad8cf7" },
  { id: "D", trackId: "T4", progress: 0.15, speed: 0.07, color: "#ffd24d" },
];

/* -------------------------- Component --------------------------- */

export default function BigMap() {
  // compute and cache track lengths to avoid recomputing every frame
  const trackMap = useRef(new Map());
  useEffect(() => {
    for (let t of tracksData) {
      trackMap.current.set(t.id, computeSegments(t.points));
    }
  }, []);

  const [trains, setTrains] = useState(initialTrains);

  // signals state: for each junction id we keep boolean occupied (true => red)
  const [signals, setSignals] = useState(() =>
    junctions.reduce((acc, j) => {
      acc[j.id] = false;
      return acc;
    }, {})
  );

  useEffect(() => {
    const intervalIds = [];

    junctions.forEach((j) => {
      // random initial delay (0–3s)
      const initialDelay = Math.random() * 3000;

      const id = setTimeout(() => {
        // toggle immediately after initial delay, then every 3s
        setSignals((prev) => ({ ...prev, [j.id]: !prev[j.id] }));

        const intervalId = setInterval(() => {
          setSignals((prev) => ({ ...prev, [j.id]: !prev[j.id] }));
        }, 3000);

        intervalIds.push(intervalId); // store interval for cleanup
      }, initialDelay);

      intervalIds.push(id); // store timeout for cleanup
    });

    // cleanup all timers on unmount
    return () => {
      intervalIds.forEach((id) => {
        clearTimeout(id);
        clearInterval(id);
      });
    };
  }, []);

  // helper: get track by id
  const getTrack = (id) => tracksData.find((t) => t.id === id);

  // Helper: compute train's world position and angle given its track and progress
  function trainWorldPos(train) {
    const track = getTrack(train.trackId);
    if (!track) return null;
    return getPosOnPoints(track.points, train.progress);
  }

  // main loop - requestAnimationFrame style using timestamp diffs so speed is consistent
  const lastRef = useRef(performance.now());
  useEffect(() => {
    let rafId = 0;

    function step(now) {
      const dt = (now - lastRef.current) / 1000; // seconds
      lastRef.current = now;

      // First: determine which junctions are currently occupied (train physically within occupyRadius)
      const occupyRadius = 18; // px - if a train is within this many px of junction center, count it as occupying
      const newSignals = { ...signals }; // start from previous, we'll set true/false

      // Occupation overrides
      junctions.forEach((j) => {
        const trainNear = trains.some((t) => {
          const pos = trainWorldPos(t);
          if (!pos) return false;
          return Math.hypot(pos.x - j.x, pos.y - j.y) <= occupyRadius;
        });

        // If a train is near, signal is red; otherwise let the timer govern
        newSignals[j.id] = trainNear ? true : newSignals[j.id];
      });

      // Second: update trains' progress considering signals & stops
      const stopDistance = 38; // px: how far before junction to begin stopping (braking distance)
      let updatedTrains = trains.map((t) => {
        const track = getTrack(t.trackId);
        if (!track) return t;
        const pos = getPosOnPoints(track.points, t.progress);

        // find the nearest upcoming junctions along the train direction
        let shouldStop = false;
        for (let j of junctions) {
          if (!j.connectedTracks.includes(t.trackId)) continue;
          const d = Math.hypot(pos.x - j.x, pos.y - j.y);
          if (newSignals[j.id] && d < stopDistance + 2) {
            shouldStop = true;
            break;
          }
        }

        if (shouldStop) {
          return { ...t }; // stopped
        }

        let newProgress = t.progress + t.speed * dt;

        // if train finishes → remove
        if (newProgress >= 1) {
          return null;
        }

        return { ...t, progress: newProgress };
      });

      // 1️⃣ remove finished trains
      updatedTrains = updatedTrains.filter(Boolean);

      // 2️⃣ chance to spawn new train
      if (Math.random() < dt * 0.03 && trains.length < 10) {
        const tracks = ["T1", "T2", "T3", "T4"];
        const trackId = tracks[Math.floor(Math.random() * tracks.length)];
        updatedTrains.push({
          id: Math.random().toString(36).slice(2, 7), // random short id
          trackId,
          progress: 0,
          speed: 0.05 + Math.random() * 0.05,
          color: ["#00bfff", "#ff7f50", "#ad8cf7", "#ffd24d"][
            Math.floor(Math.random() * 4)
          ],
        });
      }

      setSignals(newSignals);
      setTrains(updatedTrains);

      rafId = requestAnimationFrame(step);
    }

    rafId = requestAnimationFrame(step);
    return () => cancelAnimationFrame(rafId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trains]); // we intentionally include trains as dependency because step reads trains; alternative would be refs

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
            {/* Draw tracks */}
            {tracksData.map((t) => (
              <Line
                key={t.id}
                points={t.points}
                stroke="#cfcfcf"
                strokeWidth={6}
                lineCap="round"
              />
            ))}

            {/* Draw junction signals (circles) */}
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

            {/* Draw trains */}
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
