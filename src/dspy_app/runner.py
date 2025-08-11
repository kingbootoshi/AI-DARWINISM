"""
Application runner that orchestrates logging, LM configuration, task wiring,
baseline evaluation, and optional GEPA optimization.

This module centralizes control-flow so `main.py` stays as a thin entry.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple
import threading
import numpy as np
from loguru import logger

import dspy

from config.settings import get_settings
from config.logging import init_logging
from dspy_base.lm import configure_dspy
from webui.state import state as ui_state

from .wiring import setup_task
from .tasks.horror import set_horror_judge_model
from .gepa.core import GEPA


def run(args) -> None:
    """Run the app using parsed CLI arguments.

    Parameters
    ----------
    args: argparse.Namespace
        Arguments produced by `dspy_app.cli.parse_args`.
    """
    # Initialize logging and DSPy configuration using the centralized setup
    settings = get_settings()
    init_logging(
        app_name="GEPA-in-DSPy",
        level=settings.log_level,
        log_dir=settings.log_dir,
        colorize_console=settings.colorize_console,
        file_rotation=settings.file_rotation,
        file_retention=settings.file_retention,
    )

    logger.info("Starting GEPA-in-DSPy with model: {}", settings.model)

    # Configure LM and DSPy
    dspy_random_seed = getattr(args, "seed", 2025)
    _ = dspy_random_seed  # seed is applied within GEPA; DSPy modules are pure
    lm = configure_dspy(settings)

    # Start web UI if requested
    if getattr(args, "web", False):
        try:
            from webui.server import run as run_server

            t = threading.Thread(target=run_server, kwargs={"host": "127.0.0.1", "port": 3000}, daemon=True)
            t.start()
            logger.info("Web UI running at http://127.0.0.1:3000")
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to start web UI: {}", exc)

    # Resolve task-specific factories and data
    prompts, system_factory, data, metric = setup_task(
        task=args.task,
        lm=lm,
        use_paper_prompts=args.use_paper_prompts,
        pdf_path=args.pdf,
    )

    # Configure judge model if provided (horror only)
    if args.task == "horror":
        set_horror_judge_model(args.judge_model)

    # Initialize UI header state
    ui_state.reset(
        task=args.task,
        model=settings.model,
        pareto_size=args.pareto,
        minibatch_size=args.minibatch,
        budget=args.budget,
        merge_prob=args.merge_prob,
    )

    # Baseline evaluation
    sys_inst = system_factory(prompts)
    print("\n=== Baseline evaluation ===")
    s_list: List[float] = []
    if args.task == "horror":
        for idx, it in enumerate(data):
            out = sys_inst.run(it)
            score = metric(out.answer, "")
            s_list.append(score)
            ui_state.add_baseline_output(
                item_index=idx,
                inputs={"prompt": it.prompt},
                outputs={"story": out.answer},
                score=float(score),
            )
            print(f"Prompt: {it.prompt}\nStory: {out.answer}\nScore: {score:.3f}\n")
        print(f"Mean score: {np.mean(s_list):.3f}")
    else:
        for idx, it in enumerate(data):
            out = sys_inst.run(it)
            score = metric(out.answer, it.answer)
            s_list.append(score)
            ui_state.add_baseline_output(
                item_index=idx,
                inputs={"question": it.question},
                outputs={"answer": out.answer, "gold": it.answer},
                score=float(score),
            )
            print(f"Q: {it.question}\nA: {out.answer}  | gold: {it.answer}  | EM={score:.0f}\n")
        print(f"Mean EM: {np.mean(s_list):.3f}")

    # Optional GEPA optimization
    if args.optimize:
        print("\n=== Running GEPA optimization ===")
        gepa = GEPA(
            system_factory=system_factory,
            train_items=data,
            eval_metric=metric,
            minibatch_size=args.minibatch,
            pareto_set_size=args.pareto,
            seed=args.seed,
            merge_prob=args.merge_prob,
        )
        # Attach reflection/acceptance knobs and feedback extractor for horror
        if args.task == "horror":
            from .tasks.horror import horror_feedback_extractor
            gepa.feedback_extractor = horror_feedback_extractor
        gepa.k_proposals = int(args.k)
        gepa.reflect_temp = float(args.reflect_temp)
        gepa.accept_eps = float(args.accept_eps)
        winner, best_mean = gepa.optimize(seed_prompts=prompts, rollout_budget=args.budget)
        print("\nBest Pareto mean after GEPA:", f"{best_mean:.3f}")
        print("\n--- Improved prompts (diff vs start) ---")
        for k in sorted(winner.prompts.keys()):
            if winner.prompts[k].strip() != prompts[k].strip():
                print(f"\n[{k}]")
                print(winner.prompts[k])

        # Final pass with improved prompts
        final_sys = system_factory(winner.prompts)
        s_list2: List[float] = []
        if args.task == "horror":
            for it in data:
                out = final_sys.run(it)
                s_list2.append(metric(out.answer, ""))
            print(f"\nFinal Mean score with evolved prompts: {np.mean(s_list2):.3f}")
        else:
            for it in data:
                out = final_sys.run(it)
                s_list2.append(metric(out.answer, it.answer))
            print(f"\nFinal Mean EM with evolved prompts: {np.mean(s_list2):.3f}")


