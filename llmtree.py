# /// script
# dependencies = [
#   "mlx-lm==0.28.4",
#   "rich==14.2.0",
#   "sortedcontainers==2.4.0",
# ]
# ///
from __future__ import annotations

import argparse
import heapq
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from threading import Lock, Thread

import mlx.core as mx
from mlx.nn import Module
from mlx_lm import load
from mlx_lm.models.cache import KVCache
from mlx_lm.tokenizer_utils import TokenizerWrapper
from rich.console import Console, Group
from rich.live import Live
from rich.style import Style
from rich.table import Table
from rich.text import Text
from sortedcontainers import SortedList


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
    next_logprobs: mx.array | None = None

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
    cloned.keys = mx.contiguous(c.keys) if c.keys is not None else None
    cloned.values = mx.contiguous(c.values) if c.values is not None else None
    return cloned


def _clone_prompt_cache(cache: list[KVCache]) -> list[KVCache]:
    return [_clone_kv_cache(c) for c in cache]


def _eval_prompt_cache(cache: list[KVCache]) -> None:
    arrays: list[mx.array] = []
    for c in cache:
        if c.keys is not None:
            arrays.append(c.keys)
        if c.values is not None:
            arrays.append(c.values)
    if arrays:
        mx.eval(arrays)


def _top_tokens_from_logprobs(logprobs: mx.array) -> list[OutputToken]:
    vocab = int(logprobs.shape[0])
    k = min(args.top_k, vocab)
    part = mx.argpartition(logprobs, vocab - k)
    top_idx = part[vocab - k :]
    top_lp = mx.take(logprobs, top_idx)
    order = mx.argsort(top_lp)[::-1]
    sorted_indices = mx.take(top_idx, order)

    if args.temperature == 1.0:
        probs = mx.exp(mx.take(logprobs, sorted_indices))
    else:
        lse = mx.logsumexp(logprobs / args.temperature, axis=-1)
        probs = mx.exp(mx.take(logprobs, sorted_indices) / args.temperature - lse)

    mx.eval(sorted_indices, probs)
    token_ids: list[int] = sorted_indices.astype(mx.int64).tolist()
    token_probs: list[float] = mx.reshape(probs, (-1,)).tolist()

    output_tokens: list[OutputToken] = []
    cum_prob = 0.0
    for token_id, prob in zip(token_ids, token_probs):  # type: ignore[call-arg]
        if output_tokens and cum_prob >= args.top_p:
            break
        output_tokens.append(OutputToken(token=token_id, prob=float(prob)))
        cum_prob += float(prob)

    return output_tokens


class StopSignal:
    _stop = False

    def stop(self):
        self._stop = True

    @property
    def stopped(self) -> bool:
        return self._stop


class PromptTreeSearch:
    model: Module
    tokenizer: TokenizerWrapper
    prompt: list[int]
    signal: StopSignal
    _lock: Lock
    _decoded_token_cache: dict[int, str]
    _frontier: list[tuple[float, int, Branch]]
    _finished_eos: SortedList[Branch]
    _heap_counter: int = 0

    tokens: int = 0
    pruned: int = 0

    _low_watermark: float | None = None
    _start: datetime | None = None
    _end: datetime | None = None

    def __init__(self, model: Module, tokenizer: TokenizerWrapper, prompt: list[int], signal: StopSignal) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.signal = signal
        self._lock = Lock()
        self._frontier = []
        self._finished_eos = SortedList(key=lambda b: -b.probability)

        root = Branch(parent=None, token=None)
        self.branches = SortedList(key=lambda b: -b.probability)
        self.branches.add(root)
        self._push_frontier(root)

    def decode_token(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id], skip_special_tokens=True)  # type: ignore[call-arg]

    @property
    def active(self) -> int:
        with self._lock:
            return len(self._frontier)

    @property
    def n_finished(self) -> int:
        with self._lock:
            return len(self._finished_eos)

    def top_branches(self, n: int) -> list[Branch]:
        with self._lock:
            return list(self.branches[:n])

    def stats_snapshot(self) -> tuple[int, int, int]:
        with self._lock:
            return (len(self._frontier), self.pruned, self.tokens)

    def _discard_branch_unlocked(self, branch: Branch) -> None:
        try:
            self.branches.remove(branch)
        except ValueError:
            pass

    def _push_frontier_unlocked(self, branch: Branch) -> None:
        self._heap_counter += 1
        # Use a max-heap by storing negative probability.
        heapq.heappush(self._frontier, (-branch.probability, self._heap_counter, branch))

    def _push_frontier(self, branch: Branch) -> None:
        with self._lock:
            self._push_frontier_unlocked(branch)

    def _update_low_watermark(self) -> None:
        if len(self._finished_eos) < args.n:
            self._low_watermark = None
            return
        self._low_watermark = self._finished_eos[args.n - 1].probability

    def _should_stop(self) -> bool:
        if self.signal.stopped:
            return True
        with self._lock:
            if not self._frontier:
                return True
            if self._low_watermark is None:
                return False
            best_prob = -self._frontier[0][0]
            return best_prob < self._low_watermark

    def start(self) -> Thread:
        self._start = datetime.now()

        def loop():
            while not self._should_stop():
                with self._lock:
                    _, _, branch = heapq.heappop(self._frontier)
                    low_watermark = self._low_watermark

                if branch.finish_reason is not None:
                    continue
                if low_watermark is not None and branch.probability < low_watermark:
                    with self._lock:
                        self.pruned += 1
                        branch.finish_reason = "pruned"
                    continue

                self._ensure_branch_state(branch)
                if branch.next_logprobs is None:
                    branch.finish_reason = "error"
                    break

                with self._lock:
                    # `branch` is no longer a leaf once expanded, so remove it from the display set.
                    self._discard_branch_unlocked(branch)

                new_branches: list[Branch] = []
                frontier_add: list[Branch] = []
                eos_add: list[Branch] = []
                pruned_children = 0
                for tok in _top_tokens_from_logprobs(branch.next_logprobs):
                    new_prob = branch.probability * tok.prob
                    new_branch = Branch(parent=branch, token=tok, probability=new_prob)

                    if new_prob < args.min_probability:
                        pruned_children += 1
                        new_branch.finish_reason = "low_probability"
                        new_branches.append(new_branch)
                        continue

                    if tok.token in self.tokenizer.eos_token_ids:
                        new_branch.finish_reason = "eos_token"
                        new_branches.append(new_branch)
                        eos_add.append(new_branch)
                        continue

                    new_branches.append(new_branch)
                    frontier_add.append(new_branch)

                with self._lock:
                    self.pruned += pruned_children
                    for b in new_branches:
                        self.branches.add(b)
                    for b in frontier_add:
                        self._push_frontier_unlocked(b)
                    for b in eos_add:
                        self._finished_eos.add(b)
                    if eos_add:
                        self._update_low_watermark()
                    self.tokens += 1

                # The cache is needed for descendants; the logits are not.
                branch.next_logprobs = None

        thread = Thread(target=loop)
        thread.start()
        return thread

    def _ensure_branch_state(self, branch: Branch) -> None:
        if branch.next_logprobs is not None and (branch.cache is not None):
            return

        if branch.parent is None:
            cache = self.model.make_cache() if hasattr(self.model, "make_cache") else []  # type: ignore[assignment]
            inputs = mx.array([self.prompt], mx.int32)
            logits = self.model(inputs, cache=cache)[:, -1, :]
            logits = logits.astype(mx.float32)
            logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            mx.eval(logprobs)
            _eval_prompt_cache(cache)
            branch.cache = cache
            branch.next_logprobs = mx.reshape(logprobs, (-1,))
            return

        parent = branch.parent
        self._ensure_branch_state(parent)
        if parent.cache is None or branch.token is None:
            return

        cache = _clone_prompt_cache(parent.cache)
        token_id = branch.token.token
        inputs = mx.array([[token_id]], mx.int32)
        logits = self.model(inputs, cache=cache)[:, -1, :]
        logits = logits.astype(mx.float32)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        mx.eval(logprobs)
        _eval_prompt_cache(cache)
        branch.cache = cache
        branch.next_logprobs = mx.reshape(logprobs, (-1,))


def style_for_token_probability(prob: float) -> Style:
    # Discrete 10% probability bands for readability.
    if prob != prob:  # NaN
        prob = 0.0
    elif prob < 0.0:
        prob = 0.0
    elif prob > 1.0:
        prob = 1.0

    band = min(int(prob * 10), 9)  # 0..9

    # Low -> high: red -> orange -> yellow -> green.
    # Chosen to look distinct even on terminals without truecolor support.
    band_colors = [
        "#7f7f7f",  # 0-10%: grey
        "#ff3b30",  # 10-20%: red
        "#ff6a00",  # 20-30%: orange
        "#ff8c00",  # 30-40%: dark orange
        "#ffb000",  # 40-50%: amber
        "#ffd000",  # 50-60%: yellow
        "#d7e500",  # 60-70%: yellow-green
        "#a8e600",  # 70-80%: greenish
        "#4cd964",  # 80-90%: green
        "#00c853",  # 90-100%: bright green
    ]

    return Style(color=band_colors[band])


def render_probability_legend() -> Text:
    legend = Text("Legend: ", style="bold", no_wrap=True, overflow="ellipsis")
    for i in range(9, -1, -1):
        style = style_for_token_probability((i + 0.5) / 10)
        if i == 9:
            label = "90%+"
        elif i == 0:
            label = "0–10%"
        else:
            label = f"{i * 10}%+"

        legend.append("■", style=style)
        legend.append(f" {label}")
        if i != 0:
            legend.append("  ")

    return legend


def render_branches(walker: PromptTreeSearch) -> Table:
    table = Table(expand=True)
    table.add_column("Fin", justify="center", no_wrap=True, width=3)
    table.add_column("Prob.", justify="right", no_wrap=True, width=8)
    table.add_column("Answer", ratio=1)

    branches = walker.top_branches(args.n)
    for i in range(args.n):
        if i >= len(branches):
            table.add_row("", "", "")
            continue

        branch = branches[i]
        answer_text = Text()
        for tok in branch.answer_tokens():
            piece = walker.decode_token(tok.token)
            if not piece:
                continue
            answer_text.append(piece, style=style_for_token_probability(tok.prob))
        probability_text = f"{branch.probability * 100:6.2f}%"
        status: Text
        if branch.finish_reason == "eos_token":
            status = Text("✓", style="green")
        elif branch.finish_reason == "low_probability":
            status = Text("✓", style="yellow")
        elif branch.finish_reason == "pruned":
            status = Text("✓", style="dim")
        else:
            status = Text("")

        table.add_row(status, probability_text, answer_text)

    return table


def render_stats_bar(walker: PromptTreeSearch) -> Table:
    elapsed = (datetime.now() - walker._start).total_seconds() if walker._start else 0.0
    active, pruned, tokens = walker.stats_snapshot()
    tps = tokens / elapsed if elapsed > 0 else 0.0
    left = f"active {active}  pruned {pruned} tps {tps:0.1f}"
    grid = Table.grid(expand=True)
    grid.add_column(ratio=1)
    grid.add_column(justify="right", no_wrap=True)
    grid.add_row(
        Text(left, overflow="ellipsis", no_wrap=True),
        Text(
            f"top_k={args.top_k}  top_p={args.top_p}  temp={args.temperature}",
            no_wrap=True,
        ),
    )
    return grid


def render_view(walker: PromptTreeSearch) -> Group:
    return Group(
        render_probability_legend(),
        render_branches(walker),
        render_stats_bar(walker),
    )


def main() -> None:
    load_resp = load(args.model)
    model = load_resp[0]
    tokenizer = load_resp[1]

    prompt = tokenizer.apply_chat_template(  # type: ignore[call-arg]
        [{"role": "user", "content": args.prompt}],
        add_generation_prompt=True,
    )

    console = Console()

    signal = StopSignal()
    walker = PromptTreeSearch(model, tokenizer, prompt, signal)
    walker_thread = walker.start()

    try:
        with Live(console=console, transient=False) as live:
            def render():
                while not signal.stopped:
                    time.sleep(max(0.1, args.stats_interval))
                    live.update(render_view(walker))
            render_thread = Thread(target=render, daemon=True)
            render_thread.start()
            walker_thread.join()
            signal.stop()
            render_thread.join()
            live.update(render_view(walker))
    except KeyboardInterrupt:
        signal.stop()


parser = argparse.ArgumentParser()
parser.add_argument("-p", "--prompt", default="What is 2+2?", help="Prompt to score")
parser.add_argument("-m", "--model", default="mlx-community/Llama-3.2-1B-Instruct-4bit")
parser.add_argument("-n", default=10, type=int, help="Number of answers to show")
parser.add_argument("--min-probability", type=float, default=0.0001)
parser.add_argument("--top-k", dest="top_k", default=50, type=int)
parser.add_argument("--top-p", dest="top_p", default=1.0, type=float, help="Nucleus sampling threshold (0 < p <= 1)")
parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature (> 0)")
parser.add_argument(
    "--stats-interval",
    type=float,
    default=0.1,
    help="Seconds between live stats bar updates (<=0 disables)",
)
argv = [a for a in sys.argv[1:] if a != "--"]
args = parser.parse_args(argv)

if args.temperature <= 0:
    parser.error("--temperature must be > 0")
if not (0 < args.top_p <= 1):
    parser.error("--top-p must be in the range (0, 1]")

main()
