"""JSONL event trace logger for tracking agent loop execution and debugging."""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class TraceEvent:
    event_type: str
    step: int
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "step": self.step,
            "message": self.message,
            "data": self.data,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class TraceLogger:
    """Appends trace events to JSONL file and maintains in-memory history."""

    def __init__(self, log_path: Optional[Union[str, Path]] = None):
        self.log_path = Path(log_path) if log_path else None
        self.events: List[TraceEvent] = []

        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event_type: str, step: int, message: str = "", data: Optional[Dict[str, Any]] = None) -> TraceEvent:
        evt = TraceEvent(
            event_type=event_type,
            step=step,
            timestamp=time.time(),
            message=message,
            data=data or {},
        )
        self.events.append(evt)

        if self.log_path:
            try:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(evt.to_json() + "\n")
            except Exception as e:
                print(f"[TraceLogger Error] Failed to write trace to file: {e}")

        return evt

    def get_events(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self.events]
