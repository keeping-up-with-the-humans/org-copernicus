"""The counting and recording parts of the smoke-test script.

They are the token estimate, the spend ledger, the metered backend and the acceptance metrics.
The script ``smoke_two_stage.py`` holds the options, the flow of a run and the report; this
module holds the parts that count and record. Nothing here reads a key or writes a request
body: the ledger holds the backend kind, numbers and timestamps only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import threading
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, get_args

import pandas as pd

import copernicus as cop
from copernicus.coders import (
    VERIFIED_NONE,
    load_wording_pool,
    plan_cell,
    render_extraction_prompt,
    render_scoring_prompt,
    scoring_rubric_text,
    scoring_user_message,
    select,
    text_key,
    user_message_for_text,
)
from copernicus.coders.two_stage_types import Status
from copernicus.llm import Backend, BackendInfo, GenerationRequest, GenerationResult
from copernicus.references import TokenEstimate, token_counter

__all__ = [
    "MeteredBackend",
    "SPEND_FILE",
    "acceptance_metrics",
    "append_spend",
    "estimate_run",
    "estimated_cost",
    "format_acceptance",
    "format_estimate",
    "read_total",
    "spend_entry",
]

SPEND_FILE = "spend.json"
COMPLETION_PER_TEXT_TOKEN = 1.5
COMPLETION_BASE_TOKENS = 200
STAGE2_COMPLETION_TOKENS = 300
# The kind that the fake backend reports in its info; its answers cost nothing.
FAKE_BACKEND = "fake"


def estimate_run(
    table: cop.TextTable,
    rubric: cop.Rubric,
    text_ids: Sequence[Any],
    settings: cop.CodingSettings,
    use_tiktoken: bool,
) -> tuple[TokenEstimate, TokenEstimate]:
    """Estimate the tokens of the extraction stage and of the scoring stage.

    The extraction prompt of every text, run and attribute is rendered as the run will send
    it and counted. Each extraction answer is assumed to take ``min(stage1_max_new_tokens,
    1.5 x text tokens + 200)`` tokens. Each scoring call is assumed to happen (a text without
    evidence skips it), with its system prompt over all markers, the rubric and the whole text
    as the passages (the passages are verbatim parts of the text) as its prompt, and 300
    tokens as its answer. Retries and texts shorter than the minimum length are not
    discounted or added.

    Args:
        table: The texts.
        rubric: The rubric.
        text_ids: Ids of the texts to code.
        settings: The coding settings; they fix the wording rotation and the run count.
        use_tiktoken: Whether tiktoken counts the prompt tokens where it is installed;
            ``False`` counts characters / 4 and never loads an encoding from the network.

    Returns:
        The estimate of the extraction stage and of the scoring stage.
    """
    count, counter = token_counter(use_tiktoken=use_tiktoken)
    pool = select(load_wording_pool(), settings.wording_enable, settings.wording_disable)
    stage1_prompt = stage1_completion = stage2_prompt = calls = 0
    for raw_id in text_ids:
        text = table.text_of(raw_id)
        text_tokens = count(text)
        completion = min(
            settings.two_stage.stage1_max_new_tokens,
            math.ceil(COMPLETION_PER_TEXT_TOKEN * text_tokens + COMPLETION_BASE_TOKENS),
        )
        user_tokens = count(user_message_for_text(text))
        for run in range(1, settings.runs + 1):
            plan = plan_cell(
                rubric,
                pool,
                text_id=str(raw_id),
                text=text,
                text_key=text_key(raw_id),
                run_number=run,
                run_seed=settings.seed,
                shuffle_markers=settings.shuffle_markers,
            )
            for context in plan.contexts:
                names = list(context.marker_order)
                stage1_prompt += count(render_extraction_prompt(context)) + user_tokens
                stage1_completion += completion
                stage2_prompt += count(render_scoring_prompt(context, names)) + count(
                    scoring_user_message(scoring_rubric_text(context.attribute, names), [text])
                )
                calls += 1
    return (
        TokenEstimate(stage1_prompt, stage1_completion, calls, counter),
        TokenEstimate(stage2_prompt, calls * STAGE2_COMPLETION_TOKENS, calls, counter),
    )


def format_estimate(
    stages: tuple[TokenEstimate, TokenEstimate],
    options: argparse.Namespace,
    total_usd: float,
    settings: cop.CodingSettings,
) -> str:
    """Describe the estimate with its formula, and the budget it is checked against.

    The formula names the extraction budget of ``settings`` and the assumptions of
    `estimate_run`.
    """
    stage1, stage2 = stages
    prompt = stage1.prompt_tokens + stage2.prompt_tokens
    completion = stage1.completion_tokens + stage2.completion_tokens
    cost = estimated_cost(stages, options)
    budget = settings.two_stage.stage1_max_new_tokens
    return "\n".join(
        [
            f"token counter: {stage1.counter}",
            f"stage 1 (extraction): {stage1.calls} calls, {stage1.prompt_tokens} prompt tokens, "
            f"{stage1.completion_tokens} completion tokens",
            f"  completion per call = min({budget}, {COMPLETION_PER_TEXT_TOKEN} x text tokens + "
            f"{COMPLETION_BASE_TOKENS})",
            f"stage 2 (scoring): {stage2.calls} calls, {stage2.prompt_tokens} prompt tokens, "
            f"{stage2.completion_tokens} completion tokens",
            "  prompt = system prompt + rubric + the whole text as passages; completion = "
            f"{STAGE2_COMPLETION_TOKENS}",
            f"estimate (USD) = ({prompt} x {options.price_input} + {completion} x "
            f"{options.price_output}) / 1000000 = {cost:.4f}",
            f"running total in {SPEND_FILE} (USD): {total_usd:.4f}",
            f"ceiling (USD): {options.ceiling}; estimate + running total = {cost + total_usd:.4f}",
        ]
    )


def estimated_cost(
    stages: tuple[TokenEstimate, TokenEstimate], options: argparse.Namespace
) -> float:
    """Return the estimated cost of both stages in US dollars."""
    return sum(stage.cost(options.price_input, options.price_output) for stage in stages)


def read_total(path: Path) -> float:
    """Return what the ledger holds so far in US dollars; 0 when there is no ledger yet.

    Raises:
        KeyError: If the ledger lacks its entries or an entry lacks its cost.
    """
    if not path.exists():
        return 0.0
    entries = json.loads(path.read_text(encoding="utf-8"))["entries"]
    return float(sum(entry["cost_usd"] for entry in entries))


def append_spend(path: Path, entry: dict[str, Any]) -> float:
    """Add one entry to the ledger and return the new running total.

    The ledger is replaced in one step, so an interrupted write leaves the old ledger.

    Args:
        path: The ledger file.
        entry: The numbers and the timestamp of the run, as `spend_entry` builds them.

    Returns:
        The running total in US dollars.
    """
    entries = json.loads(path.read_text(encoding="utf-8"))["entries"] if path.exists() else []
    entries.append(entry)
    total = float(sum(each["cost_usd"] for each in entries))
    scratch = path.with_suffix(".tmp")
    scratch.write_text(
        json.dumps({"entries": entries, "total_usd": total}, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(scratch, path)
    return total


def spend_entry(
    options: argparse.Namespace,
    backend: str,
    n_texts: int,
    prompt_tokens: int,
    completion_tokens: int,
    answers_without_usage: int,
) -> dict[str, Any]:
    """Build the ledger entry of a run from its token counts and the prices.

    Args:
        options: The command-line options, with the prices and the number of runs.
        backend: The kind of the backend that answered, as its info reports it; the fake
            backend costs nothing, so its runs are recorded with a cost of 0.
        n_texts: Number of texts coded.
        prompt_tokens: Prompt tokens the answers reported.
        completion_tokens: Completion tokens the answers reported.
        answers_without_usage: Answers that reported no prompt or no completion count; the
            tokens of these answers are missing from the two sums.

    Returns:
        The entry: the backend kind, numbers and a UTC timestamp only.
    """
    cost = (
        0.0
        if backend == FAKE_BACKEND
        else TokenEstimate(prompt_tokens, completion_tokens, 0, "").cost(
            options.price_input, options.price_output
        )
    )
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "backend": backend,
        "texts": n_texts,
        "runs": options.runs,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "answers_without_usage": answers_without_usage,
        "price_input": options.price_input,
        "price_output": options.price_output,
        "cost_usd": cost,
    }


class MeteredBackend:
    """A backend that counts the tokens its answers report and otherwise forwards everything.

    An answer that reports no prompt count or no completion count is counted in
    ``answers_without_usage``; the counts it does report are added. A call that raises
    returns no answer and so no usage: it is not metered.
    """

    def __init__(self, backend: Backend) -> None:
        """Wrap a backend; the counts start at 0."""
        self._backend = backend
        self._lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.answers_without_usage = 0

    def _meter(self, result: GenerationResult) -> None:
        """Add the tokens of one answer to the counts."""
        with self._lock:
            if result.prompt_tokens is None or result.completion_tokens is None:
                self.answers_without_usage += 1
            if result.prompt_tokens is not None:
                self.prompt_tokens += result.prompt_tokens
            if result.completion_tokens is not None:
                self.completion_tokens += result.completion_tokens

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Forward the request and count the tokens of the answer."""
        result = self._backend.generate(request)
        self._meter(result)
        return result

    def generate_batch(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        """Answer the requests one at a time and count each answer as it arrives.

        The wrapped backend's own batch call returns all answers or none, so a batch that
        failed part way would leave the answers that came back before the failure unmetered;
        one request at a time, every answer is counted, and the requests after a failure are
        not sent.
        """
        return [self.generate(request) for request in requests]

    def info(self) -> BackendInfo:
        """Return the wrapped backend's info."""
        return self._backend.info()

    def provenance(self) -> Mapping[str, str]:
        """Return the wrapped backend's provenance, so the identity of the store is unchanged."""
        return self._backend.provenance()


def acceptance_metrics(scores: pd.DataFrame, evidence: pd.DataFrame) -> dict[str, Any]:
    """Count how well the calls and the passages came out.

    Args:
        scores: The long score table of the run store.
        evidence: The evidence table of the run store.

    Returns:
        The share of attribute calls whose status is not ``extraction_failed``, the share of
        passages that were verified (``None`` when the model returned none), the count of each
        status, and the number of scores equal to 0.
    """
    statuses = scores["status"].value_counts()
    verified = None if evidence.empty else float((evidence["verified"] != VERIFIED_NONE).mean())
    return {
        "n_attribute_calls": len(scores),
        "share_not_extraction_failed": float((scores["status"] != "extraction_failed").mean()),
        "n_passages": len(evidence),
        "share_passages_verified": verified,
        "status_counts": {
            status: int(count)
            for status, count in statuses.reindex(get_args(Status), fill_value=0).items()
        },
        "n_scores_equal_to_zero": int(scores["score"].dropna().eq(0).sum()),
    }


def format_acceptance(metrics: dict[str, Any], spend: dict[str, Any], total_usd: float) -> str:
    """Format the spend and the acceptance metrics of a run."""
    verified = metrics["share_passages_verified"]
    return "\n".join(
        [
            "== spend ==",
            f"prompt tokens {spend['prompt_tokens']}, completion tokens "
            f"{spend['completion_tokens']}",
            f"answers without usage: {spend['answers_without_usage']}",
            f"cost of this run (USD): {spend['cost_usd']:.4f}; running total (USD): "
            f"{total_usd:.4f}",
            "",
            "== acceptance ==",
            f"attribute calls: {metrics['n_attribute_calls']}",
            f"share not extraction_failed: {metrics['share_not_extraction_failed']:.3f}",
            f"passages returned: {metrics['n_passages']}; share verified: "
            + ("n/a" if verified is None else f"{verified:.3f}"),
            "statuses: " + ", ".join(f"{k} {v}" for k, v in metrics["status_counts"].items()),
            f"scores equal to 0: {metrics['n_scores_equal_to_zero']}",
        ]
    )
