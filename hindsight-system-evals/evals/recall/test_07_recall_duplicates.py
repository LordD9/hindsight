"""Recall over raw facts returns the same fact once per document that stated it.

Recall does not merge facts; observations do. With observations and consolidation
off, the same statement retained as ``RETAIN_REPEATS`` separate documents becomes
that many facts, and a recall about it is expected to hand back several copies.

This pins the expected behaviour: it fails if recall starts returning the fact only
once, because then something deduplicates and the reason should be known.
"""

from __future__ import annotations

import logging

from hindsight_client import Hindsight

from hindsight_system_evals import evaluate
from hindsight_system_evals.pages import SettleFn

log = logging.getLogger(__name__)

RETAIN_REPEATS = 4

#: One atomic fact, so every fact from these documents is a copy of the same statement.
CONTENT = "Marta Rossi's favourite tea is genmaicha, which she drinks every morning."
QUERY = "What tea does Marta Rossi like?"


async def test_recall_returns_duplicates_without_observations(
    client: Hindsight, bank_id: str, settled: SettleFn
) -> None:
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
    assert verdict.meets_criteria, (
        f"recall returned no duplicates ({len(documents)} documents among the results) — "
        f"bank {bank_id}: {verdict.reasoning}\n{results}"
    )
