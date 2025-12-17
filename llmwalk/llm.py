from __future__ import annotations

import heapq
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache

import mlx.core as mx
from mlx.nn import Module
from mlx_lm.models.cache import KVCache
from mlx_lm.tokenizer_utils import TokenizerWrapper
from sortedcontainers import SortedList


@dataclass
class SearchConfig:
    """Configuration for the tree search algorithm."""

    n: int = 10
    top_k: int = 50
    top_p: float = 1.0
    temperature: float = 1.0
    min_probability: float = 0.0001


@dataclass
class OutputToken:
    token: int
    prob: float


@dataclass(eq=False)
class Branch:
    parent: Branch | None
    token: OutputToken | None
    probability: float = 1.0
    finish_reason: str | None = None
    cache: list[KVCache] | None = None

    def answer_tokens(self) -> list[OutputToken]:
        toks: list[OutputToken] = []
        cur: Branch | None = self
        while cur is not None and cur.token is not None:
            toks.append(cur.token)
            cur = cur.parent
        toks.reverse()
        return toks


def _clone_kv_cache(c: KVCache) -> KVCache:
    cloned = KVCache()
    cloned.offset = c.offset
    cloned.keys = mx.array(c.keys) if c.keys is not None else None
    cloned.values = mx.array(c.values) if c.values is not None else None
    return cloned


def _clone_prompt_cache(cache: list[KVCache]) -> list[KVCache]:
    return [_clone_kv_cache(c) for c in cache]


def _infer_num_layers(model: Module) -> int | None:
    for obj in (model, getattr(model, "model", None)):
        if obj is None:
            continue

        n = getattr(obj, "num_hidden_layers", None)
        if isinstance(n, int) and n > 0:
            return n

        layers = getattr(obj, "layers", None)
        if layers is None:
            continue
        try:
            n_layers = len(layers)  # type: ignore[arg-type]
        except TypeError:
            n_layers = None
        if isinstance(n_layers, int) and n_layers > 0:
            return n_layers

    return None


def _make_kv_cache(model: Module) -> list[KVCache] | None:
    make_cache = getattr(model, "make_cache", None)
    if callable(make_cache):
        cache = make_cache()
        if cache is None:
            return None
        if isinstance(cache, list):
            return cache
        if isinstance(cache, Iterable):
            return list(cache)
        return None

    n_layers = _infer_num_layers(model)
    if n_layers is None:
        return None
    return [KVCache() for _ in range(n_layers)]


def _top_tokens_from_logprobs(
    logprobs: mx.array, config: SearchConfig
) -> list[OutputToken]:
    vocab = int(logprobs.shape[0])
    k = min(config.top_k, vocab)
    part = mx.argpartition(logprobs, vocab - k)
    top_idx = part[vocab - k :]
    top_lp = mx.take(logprobs, top_idx)
    order = mx.argsort(top_lp)[::-1]
    sorted_indices = mx.take(top_idx, order)

    if config.temperature == 1.0:
        probs = mx.exp(mx.take(logprobs, sorted_indices))
    else:
        lse = mx.logsumexp(logprobs / config.temperature, axis=-1)
        probs = mx.exp(mx.take(logprobs, sorted_indices) / config.temperature - lse)

    mx.eval(sorted_indices, probs)
    token_ids = list(sorted_indices.astype(mx.int64).tolist())  # type: ignore[arg-type]
    token_probs = list(mx.reshape(probs, (-1,)).tolist())  # type: ignore[arg-type]

    output_tokens: list[OutputToken] = []
    cum_prob = 0.0
    for token_id, prob in zip(token_ids, token_probs):  # type: ignore[call-arg]
        if output_tokens and cum_prob >= config.top_p:
            break
        output_tokens.append(OutputToken(token=token_id, prob=float(prob)))
        cum_prob += float(prob)

    return output_tokens


class PromptTreeSearch:
    model: Module
    tokenizer: TokenizerWrapper
    prompt: list[int]
    config: SearchConfig
    _frontier: list[tuple[float, int, Branch]]
    _finished_eos: SortedList  # SortedList[Branch]
    _heap_counter: int = 0
    _stopped: bool = False

    tokens: int = 0
    pruned: int = 0

    _low_watermark: float | None = None
    _start: datetime | None = None
    _end: datetime | None = None

    def __init__(
        self,
        model: Module,
        tokenizer: TokenizerWrapper,
        prompt: list[int],
        config: SearchConfig | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.config = config or SearchConfig()
        self._frontier = []
        self._finished_eos = SortedList(key=lambda b: -b.probability)

        root = Branch(parent=None, token=None)
        self.branches = SortedList(key=lambda b: -b.probability)
        self.branches.add(root)
        self._push_frontier(root)

    @lru_cache(maxsize=65536)
    def decode_token(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id], skip_special_tokens=True)  # type: ignore[call-arg]

    def _run_model(self, cache: list[KVCache] | None, input_ids: list[int]) -> mx.array:
        self.tokens += 1
        inputs = mx.array([input_ids], mx.int32)
        logits = self.model(inputs, cache=cache)[:, -1, :]
        logits = logits.astype(mx.float32)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        return mx.reshape(logprobs, (-1,))

    @property
    def active(self) -> int:
        return len(self._frontier)

    def top_branches(self, n: int) -> list[Branch]:
        return list(self.branches[:n])

    def _push_frontier(self, branch: Branch) -> None:
        self._heap_counter += 1
        heapq.heappush(
            self._frontier, (-branch.probability, self._heap_counter, branch)
        )

    def _update_low_watermark(self) -> None:
        if len(self._finished_eos) < self.config.n:
            self._low_watermark = None
            return
        self._low_watermark = self._finished_eos[self.config.n - 1].probability

    def stop(self) -> None:
        self._stopped = True

    def should_stop(self) -> bool:
        if self._stopped:
            return True
        if not self._frontier:
            return True
        if self._low_watermark is None:
            return False
        best_prob = -self._frontier[0][0]
        return best_prob < self._low_watermark

    def step(self) -> None:
        if self._start is None:
            self._start = datetime.now()

        if self.should_stop():
            return

        _, _, branch = heapq.heappop(self._frontier)

        if self._low_watermark is not None and branch.probability < self._low_watermark:
            self.pruned += 1
            branch.finish_reason = "pruned"
            branch.cache = None
            return

        if branch.token is None:  # root branch
            cache_after = _make_kv_cache(self.model)
            logprobs = self._run_model(cache_after, self.prompt)
        else:
            if branch.cache is None:
                input_ids = self.prompt + [t.token for t in branch.answer_tokens()]
                cache_after = None
                logprobs = self._run_model(cache_after, input_ids)
            else:
                cache_after = _clone_prompt_cache(branch.cache)
                logprobs = self._run_model(cache_after, [branch.token.token])

        self.branches.remove(branch)

        new_branches: list[Branch] = []
        frontier_add: list[Branch] = []
        eos_add: list[Branch] = []
        for tok in _top_tokens_from_logprobs(logprobs, self.config):
            new_prob = branch.probability * tok.prob

            if new_prob < self.config.min_probability:
                self.pruned += 1
                new_branch = Branch(
                    parent=branch,
                    token=tok,
                    probability=new_prob,
                    finish_reason="low_probability",
                )
                new_branches.append(new_branch)
                continue

            if tok.token in self.tokenizer.eos_token_ids:
                new_branch = Branch(
                    parent=branch,
                    token=tok,
                    probability=new_prob,
                    finish_reason="eos_token",
                )
                new_branches.append(new_branch)
                eos_add.append(new_branch)
                continue

            new_branch = Branch(
                parent=branch,
                token=tok,
                probability=new_prob,
                cache=cache_after,
            )
            new_branches.append(new_branch)
            frontier_add.append(new_branch)

        for b in new_branches:
            self.branches.add(b)
        for b in frontier_add:
            self._push_frontier(b)
        for b in eos_add:
            self._finished_eos.add(b)
        if eos_add:
            self._update_low_watermark()

        branch.cache = None
