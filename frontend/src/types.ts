// src/types.ts

export interface TrainState {
  id: number;
  label: string;
  x: number;
  y: number;
  track: number; // The index of the track it's on
  speed: number; // Current speed
  targetSpeed: number; // The speed it's trying to reach
  wantsToSwitch: 0 | -1 | 1; // 0: no, -1: left, 1: right
  collisionRisk: number; // A value from 0 to 1
  haltTime: number; // Remaining halt time
}

export interface SignalState {
  id: number;
  position: number;
  state: 'red' | 'green';
}

export interface Action {
  [trainId: string]: string; // e.g., "train_0": "Accelerate"
}

export interface SuggestionPlan {
  plan_id: string;
  annotation: string;
  actions: Action;
}

// The complete JSON structure from the backend
export interface AIUpdate {
  suggestions: SuggestionPlan[];
  signals: Record<number, 'red' | 'green'>;
}

export interface AlertCard {
  id: number;
  title : String;
  message: String;
  trainId: String;
}