"""
Task wiring and setup helpers.

This module provides a single setup function that prepares prompts, system
factories, toy data, and metrics for the selected task. It lets the runner
remain task-agnostic.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple
from loguru import logger
import dspy

from webui.state import state as ui_state

from .gepa.pdf_extract import extract_prompts_from_pdf
from .tasks.hotpot import (
    BASE_SEED_PROMPTS as HOTPOT_SEEDS,
    system_factory_from_prompts as hotpot_system_factory,
    toy_hotpot_data,
    exact_match,
)
from .tasks.horror import (
    BASE_SEED_PROMPTS_HORROR as HORROR_SEEDS,
    system_factory_from_prompts_horror,
    horror_seed_data,
    horror_metric,
)


def setup_task(
    *, task: str, lm: dspy.LM, use_paper_prompts: bool, pdf_path: str | None
):
    """Return (prompts, system_factory, data, metric) for the selected task.

    Notes
    -----
    - When `use_paper_prompts` is True for hotpot, a PDF path is required.
    - The factories returned here are callables that accept a `prompts` dict and
      return a concrete system instance.
    """
    if task == "horror":
        if use_paper_prompts:
            logger.warning("--use-paper-prompts is only for Hotpot. Using horror seed prompts instead.")
        prompts = dict(HORROR_SEEDS)
        system_factory = system_factory_from_prompts_horror(lm, prompts)
        data = horror_seed_data()
        metric = horror_metric
        return prompts, system_factory, data, metric

    # Hotpot
    if use_paper_prompts:
        if not pdf_path:
            raise SystemExit("ERROR: --use-paper-prompts requires --pdf path to the GEPA paper.")
        print("Extracting GEPA prompts from PDF ...")
        paper_prompts = extract_prompts_from_pdf(pdf_path)
        prompts = paper_prompts
        print("Loaded modules:", ", ".join(sorted(prompts.keys())))
    else:
        prompts = dict(HOTPOT_SEEDS)

    system_factory = hotpot_system_factory(lm, prompts)
    data = toy_hotpot_data()
    metric = exact_match
    return prompts, system_factory, data, metric


