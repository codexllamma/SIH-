// src/components/SuggestionsPanel.tsx
import React from 'react';
import type { SuggestionPlan } from '../types';


interface Props {
  suggestions: SuggestionPlan[];
  onSelectPlan: (actions: SuggestionPlan['actions']) => void;
  lastUpdateTime: string;
}

const SuggestionsPanel: React.FC<Props> = ({ suggestions, onSelectPlan, lastUpdateTime }) => {

  return (
    <div className="suggestions-panel">
      <h3>AI Decision Support</h3>
      <p>Last update: {lastUpdateTime}</p>
      {suggestions.map(plan => (
        <button key={plan.plan_id} onClick={() => onSelectPlan(plan.actions)}>
          <strong>{plan.plan_id}:</strong> {plan.annotation}
        </button>
      ))}
    </div>
  );
};

export default SuggestionsPanel;