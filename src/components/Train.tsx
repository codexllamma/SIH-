// Train.tsx (NEW: detection inside animation loop + robust debug)
import React, { useEffect, useRef, useState } from "react";
import { Group, Rect, Text } from "react-konva";

interface SignalType {
  id: string | number;
  x: number;
  y: number;
}

interface TrainProps {
  x: number;
  y: number;
  label: string;
  colour?: string;
  speed?: number;
  trackLength?: number;
  signals: SignalType[];
  signalStates: { [id: string]: boolean }; // true = red
  assignedSignals?: (string | number)[];
  detectionDistance?: number; // px
  verticalTolerance?: number; // px
  debug?: boolean;
}

const Train: React.FC<TrainProps> = ({
  x,
  y,
  label,
  colour = "blue",
  speed = 2,
  trackLength = 850,
  signals,
  signalStates,
  assignedSignals,
  detectionDistance = 30,
  verticalTolerance = 30,
  debug = true,
}) => {
  const trainWidth = 50;
  const trainHeight = 25;

  // position state + ref (posRef used inside RAF loop for up-to-date value)
  const [posX, setPosX] = useState(x);
  const posRef = useRef<number>(x);
  // paused refs + state
  const pausedRef = useRef<boolean>(false);
  const [paused, setPaused] = useState(false);
  const pausedForSignalRef = useRef<string | number | null>(null);

  const rafRef = useRef<number | null>(null);

  // if x prop changes (reset start), sync pos
  useEffect(() => {
    posRef.current = x;
    setPosX(x);
  }, [x]);

  useEffect(() => {
    if (debug) {
      console.log(`[Train ${label}] mounted. startX=${x}, y=${y}`);
      console.log(
        `[Train ${label}] signals passed:`,
        signals.map((s) => ({ id: s.id, x: s.x, y: s.y }))
      );
      console.log(`[Train ${label}] assignedSignals:`, assignedSignals);
    }
    // no cleanup logs necessary
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    // The animation loop performs detection + movement in one place.
    const step = () => {
      const frontX = posRef.current + trainWidth;

      // choose which signals to consider
      const mySignals: SignalType[] = Array.isArray(assignedSignals)
        ? signals.filter((s) =>
            assignedSignals!.map(String).includes(String(s.id))
          )
        : signals.slice();

      // find nearest AHEAD signal (dx >= 0) — we only care about signals on this track
      let nearest: SignalType | null = null;
      let nearestDx = Infinity;

      for (const s of mySignals) {
        const dx = s.x - frontX; // positive if ahead
        const dy = Math.abs(s.y - y);
        if (dx >= 0 && dy <= verticalTolerance) {
          if (dx < nearestDx) {
            nearestDx = dx;
            nearest = s;
          }
        }
      }

      // DEBUG: show nearest + basic info
      if (debug) {
        if (nearest) {
          console.log(
            `[Train ${label}] frontX=${frontX.toFixed(1)} nearest=${String(
              nearest.id
            )} dx=${nearestDx.toFixed(1)} red=${!!signalStates[
              String(nearest.id)
            ]}`
          );
        } else {
          if (debug)
            console.log(
              `[Train ${label}] frontX=${frontX.toFixed(1)} nearest=none`
            );
        }
      }

      let shouldStop = false;
      let stopSignalId: string | number | null = null;

      if (nearest) {
        const dx = nearest.x - frontX;
        const isRed = !!signalStates[String(nearest.id)];
        // if train front is inside detectionDistance and signal is red → stop
        if (dx <= detectionDistance && dx >= 0 && isRed) {
          shouldStop = true;
          stopSignalId = nearest.id;
        }
      }

      // If paused due to a signal, keep track and only resume when condition lifted
      if (shouldStop) {
        // Stop the train (don't advance)
        if (!pausedRef.current) {
          pausedRef.current = true;
          setPaused(true);
          pausedForSignalRef.current = stopSignalId;
          if (debug)
            console.log(
              `[Train ${label}] STOPPED for signal ${String(
                stopSignalId
              )} (within ${detectionDistance}px)`
            );
        }
        // no movement this frame
      } else {
        // Not required to stop => advance train
        // If we were paused previously for a signal, decide if we should resume
        if (pausedRef.current) {
          const pausedSig = pausedForSignalRef.current;
          const pausedStillRed = pausedSig
            ? !!signalStates[String(pausedSig)]
            : false;

          // Resume if the paused signal is no longer red OR train is no longer within detectionDistance of that signal
          if (!pausedStillRed) {
            pausedRef.current = false;
            setPaused(false);
            pausedForSignalRef.current = null;
            if (debug)
              console.log(
                `[Train ${label}] RESUMED — paused signal ${String(
                  pausedSig
                )} turned GREEN`
              );
          } else {
            // still red: ensure we calculate distance to that paused signal and only resume if out of detectionDistance
            // find that paused signal in the signals list (if present)
            const pausedSignal = signals.find(
              (s) => String(s.id) === String(pausedSig)
            );
            if (pausedSignal) {
              const pausedDx = pausedSignal.x - frontX;
              if (pausedDx > detectionDistance) {
                pausedRef.current = false;
                setPaused(false);
                pausedForSignalRef.current = null;
                if (debug)
                  console.log(
                    `[Train ${label}] RESUMED — moved out of ${detectionDistance}px for ${String(
                      pausedSig
                    )}`
                  );
              } else {
                // still inside detection zone and still red -> remain stopped (shouldn't get here because shouldStop would be true)
              }
            } else {
              // paused signal not found (edge case) -> resume
              pausedRef.current = false;
              setPaused(false);
              pausedForSignalRef.current = null;
              if (debug)
                console.log(`[Train ${label}] RESUMED — paused signal missing`);
            }
          }
        }

        // move (only when not paused)
        if (!pausedRef.current) {
          posRef.current = posRef.current + speed;
          if (posRef.current > trackLength) posRef.current = x; // loop
        }
      }

      // sync state for rendering
      setPosX(posRef.current);

      // next frame
      rafRef.current = requestAnimationFrame(step);
    };

    // start loop
    rafRef.current = requestAnimationFrame(step);

    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
    // Recreate the loop when any of these change so the closure has latest props
  }, [
    signals,
    signalStates,
    assignedSignals,
    detectionDistance,
    verticalTolerance,
    speed,
    trackLength,
    x,
    y,
    label,
    debug,
  ]);

  return (
    <Group x={posX} y={y - trainHeight / 2}>
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
    </Group>
  );
};

export default Train;
