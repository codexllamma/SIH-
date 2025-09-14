// src/components/Simulation.tsx
import React, { useState, useEffect, useCallback } from 'react';
import { Stage, Layer, Line } from 'react-konva';
import type { TrainState, SignalState, AIUpdate, Action } from '../types';
import Train from './Train';
import Signal from './Signal';

const API_URL = 'http://127.0.0.1:8000/get-update';
const TRACK_Y_POSITIONS = [50, 80, 110, 140];
const TRACK_LENGTH = 1200;

const Simulation: React.FC = () => {
  const [trains, setTrains] = useState<TrainState[]>([]);
  const [signals, setSignals] = useState<{ id: number; position: number; active: boolean }[]>([]);
  const [suggestions, setSuggestions] = useState<AIUpdate['suggestions']>([]);
  const [lastUpdateTime, setLastUpdateTime] = useState<string>('Never');
  const [simulationStep, setSimulationStep] = useState(0);

  // Initialize Trains
  useEffect(() => {
    const COLOURS = ["aqua", "yellow", "white", "lime", "orange", "pink", "cyan", "violet", "gold", "red"];
    const initialTrains: TrainState[] = Array.from({ length: 10 }, (_, i) => ({
      id: i,
      label: `T${i}`,
      track: i % 4,
      colour: COLOURS[i % COLOURS.length],
      x: 50 + i * 20,
      y: TRACK_Y_POSITIONS[i % 4],
      speed: 0,
      targetSpeed: 2,
      wantsToSwitch: 0,
      collisionRisk: 0,
      haltTime: 0,
    }));
    setTrains(initialTrains);
  }, []);

  // Fetch AI Updates
  useEffect(() => {
    const getAIUpdate = async () => {
      if (trains.length === 0) return;

      const currentState = {
        current_step: simulationStep,
        tracks: trains.map(t => t.track),
        positions: trains.map(t => t.x),
        speeds: trains.map(t => Math.round(t.speed)),
        started: trains.map(() => true),
      };

      try {
        const response = await fetch(API_URL, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(currentState),
        });
        const data: AIUpdate = await response.json();

        setSuggestions(data.suggestions);

        // Convert signal state into boolean active
        const newSignals = Object.entries(data.signals).map(([pos, state]) => ({
          id: parseInt(pos),
          position: parseInt(pos),
          active: state === 'green', // ✅ convert to boolean
        }));
        setSignals(newSignals);
        setLastUpdateTime(new Date().toLocaleTimeString());
      } catch (error) {
        console.error("AI connection error:", error);
      }
    };

    const intervalId = setInterval(getAIUpdate, 2000);
    return () => clearInterval(intervalId);
  }, [trains, simulationStep]);

  // Allow toggling signal manually
  const toggleSignal = useCallback((id: number) => {
    setSignals(prev =>
      prev.map(signal =>
        signal.id === id ? { ...signal, active: !signal.active } : signal
      )
    );
  }, []);

  // Main Animation Loop
  useEffect(() => {
    const anim = () => {
      setTrains(prevTrains =>
        prevTrains.map(train => {
          let { speed, targetSpeed, x, y, track, wantsToSwitch } = train;

          // Accelerate/decelerate towards targetSpeed
          if (speed < targetSpeed) speed = Math.min(targetSpeed, speed + 0.02);
          if (speed > targetSpeed) speed = Math.max(targetSpeed, speed - 0.02);

          // Stop at red signals
          const frontX = x + 20;
          const redSignalAhead = signals.find(
            s => !s.active && frontX > s.position - 10 && frontX < s.position + 10
          );
          if (redSignalAhead) speed = 0;

          x += speed;
          if (x > TRACK_LENGTH) x = 0;

          // Handle switching
          if (wantsToSwitch !== 0) {
            const isAtJunction = [320, 680, 1450].some(j => Math.abs(x - j) < 10);
            if (isAtJunction) {
              track = Math.max(0, Math.min(3, track + wantsToSwitch));
              y = TRACK_Y_POSITIONS[track];
              wantsToSwitch = 0;
            }
          }

          return { ...train, x, y, speed, track, wantsToSwitch };
        })
      );
      setSimulationStep(s => s + 1);
      requestAnimationFrame(anim);
    };

    const animId = requestAnimationFrame(anim);
    return () => cancelAnimationFrame(animId);
  }, [signals]);

  return (
    <div className="app-container">
      <div className="simulation-view">
        <Stage width={TRACK_LENGTH} height={200}>
          <Layer>
            {/* Tracks */}
            {TRACK_Y_POSITIONS.map((y, i) => (
              <Line key={i} points={[0, y, TRACK_LENGTH, y]} stroke="grey" strokeWidth={2} />
            ))}

            {/* Signals */}
            {signals.map(signal => (
              <Signal
                key={signal.id}
                x={signal.position}
                y={TRACK_Y_POSITIONS[0] - 10}
                active={signal.active} // ✅ boolean now
                toggle={() => toggleSignal(signal.id)}
              />
            ))}

            {/* Trains */}
            {trains.map(train => (
            <Train
                key={train.id}
                x={train.x}
                y={train.y}
                label={train.label}
                colour={train.colour || "royalblue"}
                signals={signals.map((s) => ({ id: s.id, x: s.position, y: TRACK_Y_POSITIONS[train.track] }))} 
                signalStates={Object.fromEntries(signals.map(s => [s.id, s.active === true]))}
                assignedSignals={signals.map(s => s.id)}
                route={[]} // or train.route if you have one
                tracksData={[]} // or your tracks array
                speed={train.speed}
                detectionDistance={40}
                debug
            />
            ))}


          </Layer>
        </Stage>
      </div>
    </div>
  );
};

export default Simulation;
