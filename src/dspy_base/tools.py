"""
Reusable tools for DSPy agents.

Includes a safe Python evaluator for simple math and a ColBERT-backed
Wikipedia retriever for lightweight ReAct examples.
"""

from __future__ import annotations

from typing import List
from loguru import logger
import dspy


def evaluate_math(expression: str) -> float:
    """Evaluate a math expression using DSPy's sandboxed Python interpreter.

    This is intentionally limited; for anything beyond arithmetic, prefer
    purpose-built libraries or tool wrappers.
    """
    logger.debug("evaluate_math() expression={}", expression)
    output = dspy.PythonInterpreter({}).execute(expression)
    logger.debug("evaluate_math() output={}", output)
    return output


def search_wikipedia(query: str) -> List[str]:
    """Retrieve a few relevant abstracts using the hosted ColBERT index.

    Returns just the text payloads to keep prompts small while teaching.
    """
    logger.debug("search_wikipedia() query={}", query)
    retriever = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")
    results = retriever(query, k=3)
    texts = [x["text"] for x in results]
    logger.debug("search_wikipedia() k={} fetched", len(texts))
    return texts


