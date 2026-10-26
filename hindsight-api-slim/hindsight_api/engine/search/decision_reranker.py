"""Recall's reranker when a decision model is set up for recall.

It asks the decision model of each candidate separately whether showing it would help
someone answer the question, and orders the pool by that, in one batch of requests. The
order is final: recall applies none of the recency, temporal or proof-count boosts it
applies to a cross-encoder's scores.

This is not the ``typesafe`` reranker provider (``HINDSIGHT_API_RERANKER_PROVIDER``),
which is configured on its own and ranks differently. This one is configured by
``HindsightConfig.decision_model_config("recall")`` and talks to the endpoint through the
shared :class:`~hindsight_api.engine.decision_model.DecisionModel` client.
"""

from __future__ import annotations

import asyncio
import contextvars
import re
from collections.abc import Callable

from ...config import DecisionModelConfig
from ..cross_encoder import CrossEncoderModel
from ..decision_model import DecisionModel, DecisionQuestion
from ..token_encoding import count_tokens, truncate_to_tokens

PROVIDER_NAME = "decision_model"

# What every candidate in this rerank shares — the current date and the bank's own
# framing, which recall would otherwise paste onto each one. The decision model gets it in
# the question state once instead: repeated per candidate it cannot say which candidate
# beats which, and it spends the window on the same text once per candidate. Set by
# CrossEncoderReranker.rerank().
rerank_shared_context: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "hindsight_rerank_shared_context", default=None
)

# Defensive ceilings on the query and the shared context, which every request carries:
# past the request ceiling, every request would fail. Each also stays within a quarter of
# that ceiling, so a low one still leaves half of each request for the questions.
_MAX_QUERY_TOKENS = 2_000
_MAX_SHARED_CONTEXT_TOKENS = 4_000
# The same for each candidate's text, which one question carries alone.
_MAX_MEMORY_TOKENS = 4_000

# The question asked of each candidate. Four wordings were tried on the same pool; this
# one is the best calibrated, lifting the wanted candidate to 0.74 against 0.59 for
# "contains information needed to answer" while holding the pool median at 0.08. Framing it
# as help-versus-distract asks the question a keep/drop decision actually turns on.
_KEEP_INSTRUCTIONS = "Showing this memory to someone answering the question would help them, rather than distract them."

# Pruning (``prune=True``). A candidate is dropped as no help when its answer to
# _KEEP_INSTRUCTIONS falls below _PRUNE_BELOW, or below _PRUNE_BELOW_BEST of the best
# candidate's. The relative cut is needed because the answers move with the pool: asked
# alongside four memories answering at 0.9, an unrelated "staging cluster upgraded" came
# back 0.26, against 0.07 among weaker ones. On a labelled bench of pools mixing needed
# facts, related context and unrelated facts, over 6 shuffles, these two cuts lost no
# needed fact and dropped 58/132 unrelated ones (0.2 alone: 42; half the best: lost two).
# Reusing the ranking answer costs no extra question. Asking separately whether a memory
# "has nothing to do with the question" was worse than useless: the model answers on topic
# overlap and ignores the negation, so it dropped the needed facts and kept the junk.
_PRUNE_BELOW = 0.2
_PRUNE_BELOW_BEST = 0.4
# A candidate that repeats one kept above it is dropped. Asked of the pair with both
# memories in the state: with both in the question's instructions the model answered
# ~0.9 for any pair, related or not, and asked of one candidate against the whole list
# above it, it dropped distinct customers in a "which customers" pool as repeats. Pairwise
# it dropped 72/72 repeats and no needed fact.
_REPEAT_INSTRUCTIONS = "Memory B is a repeat of Memory A."
_REPEAT_ABOVE = 0.5
# Pairs sharing fewer words than this (over the shorter memory) are not asked: a reworded
# repeat still shares its names and values. On the bench it asked a quarter of the pairs
# and still caught every repeat; 0.4 missed some.
_REPEAT_MIN_OVERLAP = 0.3
# Only the best candidates are checked for repeats, bounding the pairs one recall asks.
# ponytail: a repeat ranked below this survives; raise it if long pools show repeats.
_REPEAT_CANDIDATES = 40
_DATE_PREFIX = re.compile(r"^\[Date: [^\]]*\]")
_WORD = re.compile(r"[a-z0-9][a-z0-9.]*")


def _date(doc: str) -> str | None:
    match = _DATE_PREFIX.match(doc)
    return match.group(0) if match else None


def _words(doc: str) -> set[str]:
    # The sentence only: the date prefix and the trailing "| When: … | Involving: …" fields
    # are shared by unrelated memories about the same day or person.
    body = _DATE_PREFIX.sub("", doc).split(" | ")[0]
    return {word for word in _WORD.findall(body.lower()) if len(word) > 2}


def _may_repeat(a: str, b: str) -> bool:
    # Two memories dated differently are two occasions — "went climbing" on the 3rd and on
    # the 10th — however alike they read; the model calls those repeats.
    if _date(a) and _date(b) and _date(a) != _date(b):
        return False
    words_a, words_b = _words(a), _words(b)
    return len(words_a & words_b) >= _REPEAT_MIN_OVERLAP * max(1, min(len(words_a), len(words_b)))


class DecisionModelReranker(CrossEncoderModel):
    """Ranks a recall pool with a decision model: one Noul per candidate, asking whether
    showing it would help rather than distract.

    Every candidate lands on one absolute scale, so the order holds all the way down to
    where the caller's budget cuts. Asking one comparative Choice over the whole pool
    instead ranks the head better (recall@1 0.94 against 0.87) but not the tail, and where
    the caller keeps dozens of memories for a generator it measured far worse end to end:
    0.59 against 0.79 on a 20-question answer-quality set.

    The scores handed back are positions, not confidences: descending from 1.0 in rank
    order. With ``prune`` on, a candidate it would leave out scores exactly 0.0, which is
    how :attr:`prunes_candidates` tells recall to drop it: one judged no help, or one that
    repeats a better-ranked candidate. The best candidate always survives.
    """

    # Its decision-model requests are traced one by one; a rerank row would count them twice.
    traces_calls = False

    def __init__(self, client: DecisionModel, prune: bool = False):
        self._client = client
        self.prunes_candidates = prune

    @classmethod
    def from_config(cls, config: DecisionModelConfig, prune: bool = False) -> DecisionModelReranker:
        return cls(DecisionModel.from_config(config), prune=prune)

    @property
    def provider_name(self) -> str:
        return PROVIDER_NAME

    async def initialize(self) -> None:
        pass

    def _capped(self, text: str, ceiling: int) -> str:
        ceiling = min(ceiling, self._client.max_request_tokens // 4)
        tokenizer = self._client.tokenizer
        return truncate_to_tokens(text, ceiling, tokenizer).text if count_tokens(text, tokenizer) > ceiling else text

    def _state(self, query: str) -> str:
        """The question's state: what is being asked, plus what every candidate shares."""
        query = self._capped(query, _MAX_QUERY_TOKENS)
        shared = self._capped(rerank_shared_context.get() or "", _MAX_SHARED_CONTEXT_TOKENS)
        return f"{shared}\n\nQuestion: {query}" if shared else f"Question: {query}"

    async def _rerank(self, query: str, docs: list[str], indices: list[int], scores: list[float]) -> None:
        """Scores ``indices`` by position, best first."""
        # Each memory capped like the query, so no single question can outgrow the window:
        # a failed request now raises instead of ranking its candidates last.
        questions = {
            f"m{slot}": DecisionQuestion(
                type="noul",
                instructions={"task": _KEEP_INSTRUCTIONS, "memory": self._capped(docs[index], _MAX_MEMORY_TOKENS)},
            )
            for slot, index in enumerate(indices)
        }
        answers = await self._client.ask_all(self._state(query), questions, scope="decision_recall_rerank")

        def answer(slot: int) -> float:
            found = answers.get(f"m{slot}")
            return found.noul if found is not None and found.noul is not None else 0.0

        # Stable on the caller's input order, so equal answers keep their RRF order.
        order = sorted(range(len(indices)), key=lambda slot: -answer(slot))
        dropped = await self._pruned(docs, indices, order, answer) if self.prunes_candidates else set()
        for position, slot in enumerate(order):
            scores[indices[slot]] = 0.0 if slot in dropped else (len(order) - position) / len(order)

    async def _pruned(
        self, docs: list[str], indices: list[int], order: list[int], answer: Callable[[int], float]
    ) -> set[int]:
        """The slots to leave out: judged no help, or a repeat of one kept above them."""
        floor = max(_PRUNE_BELOW, _PRUNE_BELOW_BEST * answer(order[0]))
        relevant = [slot for position, slot in enumerate(order) if position == 0 or answer(slot) >= floor]
        dropped = set(order) - set(relevant)
        checked = relevant[:_REPEAT_CANDIDATES]
        pairs = [
            (above, slot)
            for position, slot in enumerate(checked)
            for above in checked[:position]
            if _may_repeat(docs[indices[above]], docs[indices[slot]])
        ]
        if not pairs:
            return dropped

        async def ask(above: int, slot: int) -> float:
            a, b = (self._capped(docs[indices[s]], _MAX_MEMORY_TOKENS) for s in (above, slot))
            state = f"Memory A: {a}\nMemory B: {b}"
            question = DecisionQuestion(type="noul", instructions=_REPEAT_INSTRUCTIONS)
            found = await self._client.ask_all(state, {"repeat": question}, scope="decision_recall_prune")
            return found["repeat"].noul or 0.0

        answers = await asyncio.gather(*(ask(above, slot) for above, slot in pairs))
        repeats = {pair for pair, repeat in zip(pairs, answers) if repeat >= _REPEAT_ABOVE}
        # Greedy in rank order: a candidate goes only when it repeats one that stays, so of
        # two repeats the better-ranked one is kept.
        kept: list[int] = []
        for slot in checked:
            if any((above, slot) in repeats for above in kept):
                dropped.add(slot)
            else:
                kept.append(slot)
        return dropped

    async def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        docs = [doc for _, doc in pairs]
        by_query: dict[str, list[int]] = {}
        for index, (query, _) in enumerate(pairs):
            by_query.setdefault(query, []).append(index)
        scores = [0.0] * len(pairs)
        await asyncio.gather(*(self._rerank(query, docs, indices, scores) for query, indices in by_query.items()))
        return scores
