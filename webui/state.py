"""
Thread-safe in-memory state for real-time web UI.

This module exposes a singleton `state` that main/GEPA can update during
optimization. The FastAPI server reads from the same object to serve /state.

Design goals
------------
- Keep updates cheap and lock-scoped to very small critical sections.
- Store only the high-signal fields we need for a simple dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import threading
import time


@dataclass
class CandidateView:
    """Lightweight projection of a candidate for the UI."""

    id: int
    parent: Optional[int]
    prompts: Dict[str, str]
    mean_score: Optional[float]
    created_at: float


@dataclass
class Event:
    """Append-only event log for the UI timeline."""

    t: float
    type: str
    data: Dict[str, Any]


@dataclass
class EvolutionState:
    """Shared state snapshot for the evolving run."""

    task: str = "hotpot"
    model: str = ""
    started_at: float = field(default_factory=time.time)
    pareto_size: int = 0
    minibatch_size: int = 0
    budget: int = 0
    merge_prob: float = 0.0

    best_candidate_id: Optional[int] = None
    candidates: List[CandidateView] = field(default_factory=list)
    # Matrix of scores [candidate_index][pareto_instance_index]
    pareto_scores: List[List[float]] = field(default_factory=list)
    # Most recent mutation info for quick UI access
    last_mutation: Optional[Dict[str, Any]] = None
    # Recent events (bounded list)
    events: List[Event] = field(default_factory=list)
    # Baseline outputs (first pass before optimization)
    baseline: List[Dict[str, Any]] = field(default_factory=list)
    # Live output stream across phases: baseline | minibatch-parent | minibatch-child | pareto
    outputs_stream: List[Dict[str, Any]] = field(default_factory=list)

    # Non-serialized: lock for thread-safe updates
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize safe fields for JSON responses."""
        with self._lock:
            return {
                "task": self.task,
                "model": self.model,
                "started_at": self.started_at,
                "pareto_size": self.pareto_size,
                "minibatch_size": self.minibatch_size,
                "budget": self.budget,
                "merge_prob": self.merge_prob,
                "best_candidate_id": self.best_candidate_id,
                "candidates": [asdict(c) for c in self.candidates],
                "pareto_scores": self.pareto_scores,
                "last_mutation": self.last_mutation,
                "events": [asdict(e) for e in self.events[-200:]],
                "baseline": list(self.baseline[-200:]),
                "outputs_stream": list(self.outputs_stream[-400:]),
            }

    # --- mutation helpers ---

    def reset(self, *, task: str, model: str, pareto_size: int, minibatch_size: int, budget: int, merge_prob: float) -> None:
        """Initialize a new run."""
        with self._lock:
            self.task = task
            self.model = model
            self.started_at = time.time()
            self.pareto_size = pareto_size
            self.minibatch_size = minibatch_size
            self.budget = budget
            self.merge_prob = merge_prob
            self.best_candidate_id = None
            self.candidates.clear()
            self.pareto_scores.clear()
            self.last_mutation = None
            self.events.clear()
            self.baseline.clear()
            self.outputs_stream.clear()

    def add_candidate(self, cand_id: int, parent: Optional[int], prompts: Dict[str, str], mean_score: Optional[float], scores_row: Optional[List[float]]) -> None:
        """Register a candidate and its evaluation row."""
        with self._lock:
            self.candidates.append(
                CandidateView(
                    id=cand_id,
                    parent=parent,
                    prompts=prompts,
                    mean_score=mean_score,
                    created_at=time.time(),
                )
            )
            if scores_row is not None:
                self.pareto_scores.append(list(map(float, scores_row)))
            else:
                self.pareto_scores.append([])

    def set_best(self, cand_id: int) -> None:
        with self._lock:
            self.best_candidate_id = cand_id

    def log_event(self, ev_type: str, data: Dict[str, Any]) -> None:
        with self._lock:
            self.events.append(Event(t=time.time(), type=ev_type, data=data))
            # keep list from growing without bound
            if len(self.events) > 1000:
                self.events = self.events[-500:]

    def set_last_mutation(self, info: Dict[str, Any]) -> None:
        with self._lock:
            self.last_mutation = info

    def add_baseline_output(self, item_index: int, inputs: Dict[str, Any], outputs: Dict[str, Any], score: float) -> None:
        """Record baseline evaluation output for visualization."""
        with self._lock:
            self.baseline.append({
                "t": time.time(),
                "item_index": item_index,
                "inputs": inputs,
                "outputs": outputs,
                "score": float(score),
            })
            if len(self.baseline) > 500:
                self.baseline = self.baseline[-300:]

    def add_output_stream(self, phase: str, cand_id: int, item_index: int, inputs: Dict[str, Any], outputs: Dict[str, Any], score: float, meta: Optional[Dict[str, Any]] = None) -> None:
        """Append a compact record to the live outputs stream."""
        with self._lock:
            rec = {
                "t": time.time(),
                "phase": phase,
                "candidate": cand_id,
                "item_index": item_index,
                "inputs": inputs,
                "outputs": outputs,
                "score": float(score),
                "meta": meta or {},
            }
            self.outputs_stream.append(rec)
            if len(self.outputs_stream) > 2000:
                self.outputs_stream = self.outputs_stream[-1200:]


# Global singleton state for import from both main and the server
state = EvolutionState()


