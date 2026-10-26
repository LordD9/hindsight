"""Recall returns a fact once per document that stated it — unless the decision model prunes.

Recall does not merge facts; observations do. With observations and consolidation
off, the same statement retained as several documents becomes that many facts, and
a recall about it hands back every copy. That is the default, and the first test pins
it: it fails if recall starts returning the fact once, because then something
deduplicates and the reason should be known.

``HINDSIGHT_API_RECALL_DECISION_MODEL_PRUNE`` asks the decision model to drop a
candidate that repeats a better-ranked one, or that is no help at all. The second test
stresses that against pools built to tempt it into the wrong drop: a reworded repeat
must go, but a different customer signing up for the same beta, or a different detail
about the same offsite, reads alike and must stay. Run it with the decision model set up:

    HINDSIGHT_EVAL_SET_DECISION_MODEL_PROVIDER=typesafe \\
    HINDSIGHT_EVAL_SET_DECISION_MODEL_API_KEY=... \\
    HINDSIGHT_EVAL_SET_RECALL_DECISION_MODEL_PRUNE=true uv run pytest evals/recall
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import pytest
from hindsight_client import Hindsight

from hindsight_system_evals import evaluate
from hindsight_system_evals.pages import SettleFn, prepare_bank

log = logging.getLogger(__name__)

PRUNING = os.getenv("HINDSIGHT_EVAL_SET_RECALL_DECISION_MODEL_PRUNE", "").lower() == "true"

RETAIN_REPEATS = 4

#: One atomic fact, so every fact from these documents is a copy of the same statement.
CONTENT = "Marta Rossi's favourite tea is genmaicha, which she drinks every morning."
QUERY = "What tea does Marta Rossi like?"


async def test_recall_duplicates_follow_the_prune_flag(client: Hindsight, bank_id: str, settled: SettleFn) -> None:
    """Real extraction, so the copies are the model's own rewordings, dates and all."""
    await client.aupdate_bank_config(bank_id, enable_observations=False, enable_auto_consolidation=False)
    await client.aretain_batch(
        bank_id=bank_id,
        items=[{"content": CONTENT, "document_id": f"tea-{i}"} for i in range(RETAIN_REPEATS)],
    )
    await settled(bank_id)

    recalled = await client.arecall(bank_id, QUERY)
    results = "\n".join(f"- [{r.type}] (doc {r.document_id}) {r.text}" for r in recalled.results)
    documents = {r.document_id for r in recalled.results if r.document_id}
    log.info("recall returned %d results from %d documents:\n%s", len(recalled.results), len(documents), results)

    # The judge, not string equality: each document is extracted separately, so the
    # copies are paraphrases of one another rather than identical strings.
    verdict = await evaluate(
        results,
        "At least two of the listed results state the same fact about Marta Rossi's favourite tea.",
        context=f"The same sentence was retained as {RETAIN_REPEATS} separate documents: {CONTENT!r}",
    )
    where = f"{len(documents)} documents among the results — bank {bank_id}: {verdict.reasoning}\n{results}"
    if PRUNING:
        assert not verdict.meets_criteria, f"recall pruned, yet returned the fact more than once ({where})"
        assert recalled.results, f"recall pruned every copy ({where})"
    else:
        assert verdict.meets_criteria, f"recall returned no duplicates ({where})"


@dataclass(frozen=True)
class PruneCase:
    query: str
    #: Each inner tuple is one fact, reworded: one of them must come back, and only one.
    facts: tuple[tuple[str, ...], ...]
    #: Nothing to do with the query: should not come back.
    junk: tuple[str, ...]


_JUNK = (
    "The staging cluster was upgraded to Kubernetes 1.31.",
    "The office coffee machine was repaired on Tuesday.",
    "The design team switched from Figma to Penpot.",
    "Release 0.9.3 was deployed to production on 18 March 2026.",
)

_CASES = {
    "reworded-repeats": PruneCase(
        query=QUERY,
        facts=(
            (
                "Marta Rossi's favourite tea is genmaicha.",
                "Marta Rossi's favorite tea is genmaicha.",
                "Genmaicha is the tea Marta Rossi likes best.",
                "Marta Rossi likes genmaicha more than any other tea.",
            ),
            ("Marta Rossi drinks sencha in the afternoon.",),
        ),
        junk=_JUNK,
    ),
    # Alike in wording, distinct in fact: a "which customers" answer needs every one.
    "alike-but-distinct": PruneCase(
        query="Which customers signed up for the billing beta?",
        facts=(
            ("Acme Corp signed up for the billing beta.", "Acme Corp joined the billing beta."),
            ("Globex signed up for the billing beta.",),
            ("Initech signed up for the billing beta.",),
            ("Umbrella Health signed up for the billing beta.",),
        ),
        junk=_JUNK[:3] + ("Marta Rossi's favourite tea is genmaicha.",),
    ),
    # One subject, several attributes: none repeats another.
    "same-subject": PruneCase(
        query="Tell me about the Lisbon offsite.",
        facts=(
            ("The quarterly offsite is in Lisbon in June 2026.", "The June 2026 offsite will take place in Lisbon."),
            ("Everyone stays at the Memmo Alfama hotel during the Lisbon offsite.",),
            ("The Lisbon offsite budget was approved at 40,000 EUR.",),
        ),
        junk=_JUNK,
    ),
}


@pytest.mark.skipif(not PRUNING, reason="set HINDSIGHT_EVAL_SET_RECALL_DECISION_MODEL_PRUNE=true and a decision model")
@pytest.mark.parametrize("case_id", list(_CASES))
async def test_recall_prune_drops_repeats_and_junk(
    client: Hindsight, bank_id: str, settled: SettleFn, case_id: str
) -> None:
    case = _CASES[case_id]
    # Stored as written, so each label points at the exact text recall returns.
    await prepare_bank(client, bank_id)
    texts = [text for fact in case.facts for text in fact] + list(case.junk)
    await client.aretain_batch(bank_id=bank_id, items=[{"content": text} for text in texts])
    await settled(bank_id)

    recalled = await client.arecall(bank_id, case.query)
    returned = [r.text for r in recalled.results]
    log.info("%s: recall returned %d of %d:\n%s", case_id, len(returned), len(texts), "\n".join(returned))

    def came_back(text: str) -> bool:
        return any(text in r for r in returned)

    lost = [fact[0] for fact in case.facts if not any(came_back(text) for text in fact)]
    repeated = [fact[0] for fact in case.facts if sum(came_back(text) for text in fact) > 1]
    junk = [text for text in case.junk if came_back(text)]
    where = f"bank {bank_id}, returned:\n" + "\n".join(returned)
    assert not lost, f"{case_id}: prune dropped every copy of {lost} — {where}"
    assert not repeated, f"{case_id}: more than one copy of {repeated} came back — {where}"
    # Most, not all: an unrelated memory is dropped only when the model is sure, since
    # dropping a needed one costs more than showing a stray one.
    assert len(junk) <= len(case.junk) // 2, f"{case_id}: {len(junk)}/{len(case.junk)} unrelated came back — {where}"
