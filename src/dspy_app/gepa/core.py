"""
GEPA core implementation: algorithms 1 & 2 and optional "merge" crossover.

This file contains the `GEPA` class and all logic that operates over
`Candidate` objects to evolve module instructions using reflective mutations.
"""

from __future__ import annotations

import dataclasses
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from webui.state import state as ui_state

from .types import Candidate
from .reflect import update_instruction_via_reflection


def set_seed(seed: int = 2025) -> None:
    """Seed Python and NumPy RNGs for repeatability in small demos."""
    random.seed(seed)
    np.random.seed(seed)


class GEPA:
    """Pareto-guided instruction evolution with reflective mutation.

    Parameters mirror the legacy implementation for easy drop-in.
    """

    def __init__(
        self,
        system_factory: Callable[[Dict[str, str]], Any],
        train_items: List[Any],
        eval_metric: Callable[[str, str], float],
        minibatch_size: int = 8,
        pareto_set_size: int = 16,
        seed: int = 2025,
        merge_prob: float = 0.0,
    ) -> None:
        set_seed(seed)
        self.system_factory = system_factory
        self.train_items = train_items
        self.metric = eval_metric
        self.minibatch_size = minibatch_size
        self.pareto_set_size = pareto_set_size
        self.merge_prob = merge_prob

        idx = list(range(len(train_items)))
        random.shuffle(idx)
        self.D_pareto_idx = idx[: min(self.pareto_set_size, len(idx))]
        self.D_feedback_idx = idx[min(self.pareto_set_size, len(idx)) :] or idx

        self.candidates: List[Candidate] = []
        self.parents: List[Optional[int]] = []
        self.S: List[List[float]] = []

    # --- helpers ---
    def _evaluate_on(self, cand: Candidate, indices: Sequence[int]):
        """Run system on selected items and compute scores + compact traces."""
        sys_inst = self.system_factory(cand.prompts)
        scores: List[float] = []
        traces: List[Dict[str, Any]] = []
        for i in indices:
            item = self.train_items[i]
            io = sys_inst.run(item, capture_traces=True)
            gold = getattr(item, "answer", "")
            scr = self.metric(io.answer, gold)
            scores.append(scr)
            if dataclasses.is_dataclass(item):
                try:
                    inputs_view = dataclasses.asdict(item)
                except Exception:
                    inputs_view = {"item": repr(item)}
            else:
                inputs_view = {"item": repr(item)}
            outs = {"answer": io.answer}
            for k, v in (io.inter or {}).items():
                outs[k] = v
            issue = "ok" if scr >= 1.0 else "needs-improvement"
            traces.append({
                "score": scr,
                "inputs": inputs_view,
                "outputs": outs,
                "issue": issue,
            })
        return scores, traces

    def _initialize(self, seed_prompts: Dict[str, str]) -> None:
        base = Candidate(prompts=dict(seed_prompts))
        self.candidates = [base]
        self.parents = [None]
        scores, traces = self._evaluate_on(base, self.D_pareto_idx)
        self.S = [scores]
        for local_idx, global_idx in enumerate(self.D_pareto_idx):
            t = traces[local_idx]
            ui_state.add_output_stream(
                phase="pareto",
                cand_id=0,
                item_index=int(global_idx),
                inputs=t["inputs"],
                outputs=t["outputs"],
                score=float(scores[local_idx]),
                meta={"candidate": 0},
            )

    def _select_candidate_index(self) -> int:
        S_arr = np.array(self.S)
        s_star = S_arr.max(axis=0)
        C = set()
        for j, best in enumerate(s_star):
            winners = np.where(S_arr[:, j] == best)[0].tolist()
            C.update(winners)
        C = list(C)
        non_dom: List[int] = []
        for i in C:
            dominated = False
            for j in C:
                if i == j:
                    continue
                if np.all(S_arr[j] >= S_arr[i]) and np.any(S_arr[j] > S_arr[i]):
                    dominated = True
                    break
            if not dominated:
                non_dom.append(i)
        freqs = []
        for i in non_dom:
            f = int((S_arr[i] == s_star).sum())
            freqs.append(max(1, f))
        weights = np.array(freqs, dtype=float)
        weights = weights / weights.sum()
        return int(np.random.choice(non_dom, p=weights))

    def _select_module_name(self) -> str:
        return random.choice(list(self.candidates[0].prompts.keys()))

    def _merge(self) -> Optional[Candidate]:
        if len(self.candidates) < 2:
            return None
        i, j = random.sample(range(len(self.candidates)), 2)
        mean_i = float(np.mean(self.S[i]))
        mean_j = float(np.mean(self.S[j]))
        chosen = i if mean_i >= mean_j else j
        parent = self.candidates[chosen]
        other = self.candidates[j if chosen == i else i]
        merged: Dict[str, str] = {}
        for m in parent.prompts.keys():
            merged[m] = parent.prompts[m] if random.random() < 0.5 else other.prompts[m]
        return Candidate(prompts=merged)

    # --- main loop ---
    def optimize(self, seed_prompts: Dict[str, str], rollout_budget: int = 60):
        self._initialize(seed_prompts)
        base_mean = float(np.mean(self.S[0])) if self.S and self.S[0] else None
        ui_state.add_candidate(
            cand_id=0,
            parent=None,
            prompts=self.candidates[0].prompts,
            mean_score=base_mean,
            scores_row=self.S[0] if self.S else None,
        )
        if base_mean is not None:
            ui_state.set_best(0)
        b = max(1, self.minibatch_size)

        while rollout_budget > 0:
            rollout_budget -= 1
            do_merge = self.merge_prob > 0 and random.random() < self.merge_prob
            if do_merge:
                merged = self._merge()
                if merged is not None:
                    batch = random.sample(self.D_feedback_idx, k=min(b, len(self.D_feedback_idx)))
                    old_sc, old_tr = self._evaluate_on(self.candidates[0], batch)
                    new_sc, new_tr = self._evaluate_on(merged, batch)
                    for j, idx_global in enumerate(batch):
                        ui_state.add_output_stream(
                            phase="minibatch_parent",
                            cand_id=0,
                            item_index=int(idx_global),
                            inputs=old_tr[j]["inputs"],
                            outputs=old_tr[j]["outputs"],
                            score=float(old_sc[j]),
                            meta={"merge": True},
                        )
                        ui_state.add_output_stream(
                            phase="minibatch_child",
                            cand_id=-1,
                            item_index=int(idx_global),
                            inputs=new_tr[j]["inputs"],
                            outputs=new_tr[j]["outputs"],
                            score=float(new_sc[j]),
                            meta={"merge": True},
                        )
                    if np.mean(new_sc) > np.mean(old_sc):
                        self.candidates.append(merged)
                        self.parents.append(None)
                        S_new, tr_new = self._evaluate_on(merged, self.D_pareto_idx)
                        self.S.append(S_new)
                        cand_id = len(self.candidates) - 1
                        mean_new = float(np.mean(S_new)) if S_new else None
                        ui_state.add_candidate(
                            cand_id=cand_id,
                            parent=None,
                            prompts=merged.prompts,
                            mean_score=mean_new,
                            scores_row=S_new,
                        )
                        ui_state.log_event("merge_accept", {"child": cand_id, "mean": mean_new})
                        for local_idx, global_idx in enumerate(self.D_pareto_idx):
                            t = tr_new[local_idx]
                            ui_state.add_output_stream(
                                phase="pareto",
                                cand_id=cand_id,
                                item_index=int(global_idx),
                                inputs=t["inputs"],
                                outputs=t["outputs"],
                                score=float(S_new[local_idx]),
                                meta={"candidate": cand_id},
                            )
                        means = [float(np.mean(s)) for s in self.S]
                        best_idx = int(np.argmax(means))
                        ui_state.set_best(best_idx)
                continue

            k = self._select_candidate_index()
            module_name = self._select_module_name()

            parent = self.candidates[k]
            batch = random.sample(self.D_feedback_idx, k=min(b, len(self.D_feedback_idx)))
            parent_scores, parent_traces = self._evaluate_on(parent, batch)
            batch_traces_for_module: List[Dict[str, Any]] = []
            for pt in parent_traces:
                bt = {
                    "score": pt["score"],
                    "issue": pt["issue"],
                    "inputs": pt["inputs"],
                    "outputs": {"answer": pt["outputs"]["answer"]},
                }
                for k2 in ["summary1", "summary2", "query2"]:
                    if k2 in pt["outputs"]:
                        bt["outputs"][k2] = pt["outputs"][k2]
                batch_traces_for_module.append(bt)
            for j, idx_global in enumerate(batch):
                ui_state.add_output_stream(
                    phase="minibatch_parent",
                    cand_id=k,
                    item_index=int(idx_global),
                    inputs=parent_traces[j]["inputs"],
                    outputs=parent_traces[j]["outputs"],
                    score=float(parent_scores[j]),
                    meta={"module": module_name},
                )

            cur_instruction = parent.prompts[module_name]
            io_schema = f"Module {module_name}: keep IO schema unchanged and improve instruction."
            new_instruction = update_instruction_via_reflection(
                None,  # lm is taken from dspy.settings
                module_name,
                io_schema,
                cur_instruction,
                batch_traces_for_module,
            )
            if new_instruction.strip() == cur_instruction.strip():
                continue

            child_prompts = dict(parent.prompts)
            child_prompts[module_name] = new_instruction
            child = Candidate(prompts=child_prompts)

            child_scores, child_traces = self._evaluate_on(child, batch)
            for j, idx_global in enumerate(batch):
                ui_state.add_output_stream(
                    phase="minibatch_child",
                    cand_id=-1,
                    item_index=int(idx_global),
                    inputs=child_traces[j]["inputs"],
                    outputs=child_traces[j]["outputs"],
                    score=float(child_scores[j]),
                    meta={"module": module_name},
                )

            parent_mean = float(np.mean(parent_scores)) if parent_scores else 0.0
            child_mean = float(np.mean(child_scores)) if child_scores else 0.0
            ui_state.set_last_mutation(
                {
                    "module": module_name,
                    "parent_mean": parent_mean,
                    "child_mean": child_mean,
                    "accepted": child_mean > parent_mean,
                    "old_instruction": cur_instruction,
                    "new_instruction": new_instruction,
                }
            )
            ui_state.log_event(
                "mutation_attempt", {"module": module_name, "parent_mean": parent_mean, "child_mean": child_mean}
            )

            if child_mean > parent_mean:
                self.candidates.append(child)
                self.parents.append(k)
                S_child, tr_child = self._evaluate_on(child, self.D_pareto_idx)
                self.S.append(S_child)
                cand_id = len(self.candidates) - 1
                mean_child = float(np.mean(S_child)) if S_child else None
                ui_state.add_candidate(
                    cand_id=cand_id,
                    parent=k,
                    prompts=child.prompts,
                    mean_score=mean_child,
                    scores_row=S_child,
                )
                ui_state.log_event("mutation_accept", {"child": cand_id, "module": module_name, "mean": mean_child})
                for local_idx, global_idx in enumerate(self.D_pareto_idx):
                    t = tr_child[local_idx]
                    ui_state.add_output_stream(
                        phase="pareto",
                        cand_id=cand_id,
                        item_index=int(global_idx),
                        inputs=t["inputs"],
                        outputs=t["outputs"],
                        score=float(S_child[local_idx]),
                        meta={"candidate": cand_id},
                    )
                means = [float(np.mean(s)) for s in self.S]
                best_idx = int(np.argmax(means))
                ui_state.set_best(best_idx)

        means = [float(np.mean(s)) for s in self.S]
        best_idx = int(np.argmax(means))
        return self.candidates[best_idx], means[best_idx]


