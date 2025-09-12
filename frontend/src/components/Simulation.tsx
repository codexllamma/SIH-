// src/components/Simulation.tsx
import React, { useState, useEffect, useCallback } from 'react';
import { Stage, Layer, Line } from 'react-konva';
import type{ TrainState, SignalState, AIUpdate, Action } from '../types';
import Train from './Train';
import Signal from './Signal';
import SuggestionsPanel from './SuggestionsPanel';

// --- Constants ---
const API_URL = 'http://127.0.0.1:8000/get-update';
const TRACK_Y_POSITIONS = [50, 80, 110, 140]; // Y position for each track
const TRACK_LENGTH = 1200;

// --- Main Component ---
const Simulation: React.FC = () => {
    // --- Central State Management ---
    const [trains, setTrains] = useState<TrainState[]>([]); // All train data
    const [signals, setSignals] = useState<SignalState[]>([]); // All signal data
    const [suggestions, setSuggestions] = useState<AIUpdate['suggestions']>([]);
    const [lastUpdateTime, setLastUpdateTime] = useState<string>('Never');
    const [simulationStep, setSimulationStep] = useState(0);

    // --- Initialization Effect ---
    useEffect(() => {
        // Initialize 10 trains on different tracks
        const initialTrains: TrainState[] = Array.from({ length: 10 }, (_, i) => ({
            id: i,
            label: `T${i}`,
            track: i % 4,
            x: 50 + i * 20,
            y: TRACK_Y_POSITIONS[i % 4],
            speed: 0,
            targetSpeed: 2, // Give them an initial speed to start moving
            wantsToSwitch: 0,
            collisionRisk: 0,
            haltTime: 0,
        }));
        setTrains(initialTrains);
    }, []);

    // --- API Communication Effect ---
    useEffect(() => {
        const getAIUpdate = async () => {
            if (trains.length === 0) return;

            const currentState = {
                current_step: simulationStep,
                tracks: trains.map(t => t.track),
                positions: trains.map(t => t.x),
                speeds: trains.map(t => Math.round(t.speed)), // Model expects integers
                started: trains.map(t => true),
            };

            try {
                const response = await fetch(API_URL, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(currentState),
                });
                const data: AIUpdate = await response.json();
                
                setSuggestions(data.suggestions);
                
                // Convert backend signal data into the format our state uses
                const newSignals = Object.entries(data.signals).map(([pos, state]) => ({
                    id: parseInt(pos),
                    position: parseInt(pos),
                    state: state,
                }));
                setSignals(newSignals);
                setLastUpdateTime(new Date().toLocaleTimeString());

            } catch (error) {
                console.error("AI connection error:", error);
            }
        };

        const intervalId = setInterval(getAIUpdate, 2000); // Poll every 2 seconds
        return () => clearInterval(intervalId);
    }, [trains, simulationStep]);

    // --- User Action Handler ---
    const handlePlanSelect = useCallback((actions: Action) => {
        console.log("Applying selected plan:", actions);
        setTrains(currentTrains =>
            currentTrains.map((train, index) => {
                const action = actions[`train_${index}`];
                if (!action) return train;

                const newTrain = { ...train };
                switch (action) {
                    case "Accelerate": newTrain.targetSpeed = Math.min(4, train.speed + 1); break;
                    case "Decelerate": newTrain.targetSpeed = Math.max(0, train.speed - 1); break;
                    case "Hold": newTrain.targetSpeed = 0; break;
                    case "Emergency Brake": newTrain.targetSpeed = 0; newTrain.speed = 0; break; // Immediate stop
                    case "Switch Left": newTrain.wantsToSwitch = -1; break;
                    case "Switch Right": newTrain.wantsToSwitch = 1; break;
                }
                return newTrain;
            })
        );
    }, []);

    // --- Main Animation Loop ---
    useEffect(() => {
        const anim = () => {
            setTrains(prevTrains => prevTrains.map(train => {
                let { speed, targetSpeed, x, y, track, wantsToSwitch } = train;

                // Simple physics: move current speed towards target speed
                if (speed < targetSpeed) speed = Math.min(targetSpeed, speed + 0.02);
                if (speed > targetSpeed) speed = Math.max(targetSpeed, speed - 0.02);
                
                // If stopped by a signal, speed is 0 regardless of target
                const frontX = x + 20;
                const redSignalAhead = signals.find(s => s.state === 'red' && frontX > s.position - 10 && frontX < s.position + 10);
                if (redSignalAhead) {
                    speed = 0;
                }

                x += speed;
                if (x > TRACK_LENGTH) x = 0; // Loop for demo

                // Handle switching logic
                if (wantsToSwitch !== 0) {
                     // Find if we are near a junction (you'll need junction data)
                     const isAtJunction = [320, 680, 1450].some(j => Math.abs(x-j) < 10);
                     if(isAtJunction) {
                        track = Math.max(0, Math.min(3, track + wantsToSwitch));
                        y = TRACK_Y_POSITIONS[track];
                        wantsToSwitch = 0; // Reset switch request
                     }
                }

                return { ...train, x, y, speed, track, wantsToSwitch };
            }));
            setSimulationStep(s => s + 1);
            requestAnimationFrame(anim);
        };
        const animId = requestAnimationFrame(anim);
        return () => cancelAnimationFrame(animId);
    }, [signals]); // Re-run if signals change to react to them

    return (
        <div className="app-container">
            <div className="simulation-view">
                <Stage width={TRACK_LENGTH} height={200}>
                    <Layer>
                        {/* Render Tracks */}
                        {TRACK_Y_POSITIONS.map((y, i) => (
                            <Line key={i} points={[0, y, TRACK_LENGTH, y]} stroke="grey" strokeWidth={2} />
                        ))}
                        {/* Render Signals */}
                        {signals.map(signal => (
                            <Signal key={signal.id} x={signal.position} y={TRACK_Y_POSITIONS[0]-10} state={signal.state} />
                        ))}
                        {/* Render Trains */}
                        {trains.map(train => (
                            <Train key={train.id} train={train} debug />
                        ))}
                    </Layer>
                </Stage>
            </div>
            <SuggestionsPanel 
                suggestions={suggestions} 
                onSelectPlan={handlePlanSelect} 
                lastUpdateTime={lastUpdateTime}
            />
        </div>
    );
};

export default Simulation;