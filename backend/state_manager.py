# backend/state_manager.py
import threading
from typing import Dict, Any

class StateManager:
    def __init__(self):
        self._lock = threading.Lock()
        self.state = self._default_state()

    def _default_state(self) -> Dict[str, Any]:
        return {
            "step": 0,
            "trains": [],
            "signals": {},
            "metrics": {
                "conflicts_prevented": 0,
                "delays_mitigated": 0,
                "on_time_arrivals": 0,
                "total_delays": 0.0,
                "system_efficiency": 1.0
            },
            "suggestions": [],
            "delay_predictions": {},
            "network": {
                "congestion": 0.0,
                "throughput": 0.0,
                "cascade_factor": 1.0
            }
        }

    def update(self, key: str, value: Any):
        with self._lock:
            self.state[key] = value

    def bulk_update(self, data: Dict[str, Any]):
        """Atomic update for multiple keys (prevents partial state issues)"""
        with self._lock:
            self.state.update(data)

    def append_suggestion(self, suggestion: Dict):
        with self._lock:
            self.state["suggestions"].append(suggestion)

    def clear_suggestions(self):
        with self._lock:
            self.state["suggestions"].clear()

    def get(self) -> Dict:
        with self._lock:
            return self.state.copy()

state_manager = StateManager()
