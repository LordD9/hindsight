"""Retain's capabilities, graded on what the bank stores — one case per capability.

``test_03`` and ``test_04`` guard two regressions in depth (the language of a fact, a
dimension borrowed from a neighbour). This suite is the breadth: selectivity, exact
figures, coreference, who did it (world vs experience, a speaker named in context or
metadata), dates (relative, coarse, open-ended), language, faithfulness (hedges,
negation, corrections, the prompt's own examples, instructions inside content), the
bank's retain mission, the extraction modes, entities and entity labels, and inputs
that are structured or long. The cases, and the history behind each, are in
``hindsight_system_evals.retain_capabilities``.

Real extraction against the server's model; a judge independent of it grades the
facts. A single failure is a signal to re-run, as everywhere in this package — but a
trap is a stored falsehood, so it is what to look at first.
"""

from __future__ import annotations

import logging

import pytest
from hindsight_client import Hindsight

from hindsight_system_evals.pages import SettleFn
from hindsight_system_evals.report import RECORDED, EvalRecord
from hindsight_system_evals.retain_capabilities import CASES, Case, run_case

log = logging.getLogger(__name__)


async def _check(client: Hindsight, bank_id: str, settled: SettleFn, case: Case) -> None:
    outcome = await run_case(client, bank_id, settled, case)
    failed = outcome.missing + outcome.broken
    RECORDED.append(
        EvalRecord(
            kind="retain",
            question_id=case.id,
            category=case.capability,
            correct=not failed,
            hit_trap=bool(outcome.trapped),
            bank_id=bank_id,
            reason="; ".join(outcome.trapped or failed) or "ok",
        )
    )
    log.info("%s [%s]:\n%s", case.id, case.capability, outcome.rendered)
    evidence = f"bank {bank_id} ({case.origin})\nfacts:\n{outcome.rendered}"
    assert not outcome.trapped, f"{case.id}: stored a falsehood —\n" + "\n".join(outcome.trapped) + f"\n{evidence}"
    assert not failed, f"{case.id}:\n" + "\n".join(failed) + f"\n{evidence}"


@pytest.mark.parametrize("case", [c for c in CASES if c.minimum], ids=lambda c: c.id)
async def test_retain_capability_minimum_acceptance(
    client: Hindsight, bank_id: str, settled: SettleFn, case: Case
) -> None:
    await _check(client, bank_id, settled, case)


@pytest.mark.full
@pytest.mark.parametrize("case", [c for c in CASES if not c.minimum], ids=lambda c: c.id)
async def test_retain_capability_full(client: Hindsight, bank_id: str, settled: SettleFn, case: Case) -> None:
    await _check(client, bank_id, settled, case)
