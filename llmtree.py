# /// script
# dependencies = [
#   "mlx-lm==0.28.4",
#   "rich==14.2.0",
#   "sortedcontainers==2.4.0",
# ]
# ///
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from threading import Thread

import mlx.core as mx
from mlx.nn import Module
from mlx_lm import load
from mlx_lm.generate import BatchGenerator, BatchStats
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


@dataclass
class Branch:
    prompt: list[int]
    answer: list[OutputToken]
    probability: float = 1.0
    finish_reason: str | None = None

    def with_new_token(self, token: OutputToken) -> Branch:
        return Branch(
            prompt=self.prompt,
            answer=self.answer + [token],
            finish_reason=self.finish_reason,
            probability=self.probability * token.prob,
        )

    def tokens(self) -> list[int]:
        return self.prompt + [t.token for t in self.answer]


stats: BatchStats | None = None
active: int = 0
queued: int = 0
pruned: int = 0


def response_to_output_tokens(
    response: BatchGenerator.Response,
) -> list[OutputToken]:
    logits = response.logprobs / args.temperature
    probs = mx.softmax(logits, axis=-1)

    vocab = int(probs.shape[0])
    k = vocab if args.top_k <= 0 else min(args.top_k, vocab)
    sorted_indices = mx.argsort(probs)[-k:][::-1]
    sorted_probs = mx.take(probs, sorted_indices)

    token_ids: list[int] = sorted_indices.astype(mx.int64).tolist()
    token_probs: list[float] = mx.reshape(sorted_probs, (-1,)).tolist()

    output_tokens: list[OutputToken] = []
    cum_prob = 0.0
    for token_id, prob in zip(token_ids, token_probs):  # type: ignore[call-arg]
        if output_tokens and cum_prob >= args.top_p:
            break
        output_tokens.append(OutputToken(token=token_id, prob=prob))
        cum_prob += prob

    return output_tokens


class StopSignal:
    _stop = False

    def stop(self):
        self._stop = True

    @property
    def stopped(self) -> bool:
        return self._stop


class PromptTreeSearch:
    _branch_lookup: dict[int, Branch]
    branches: SortedList[Branch]
    gen: BatchGenerator
    model: Module
    tokenizer: TokenizerWrapper
    prompt: list[int]
    signal: StopSignal
    _decoded_token_cache: dict[int, str]

    tokens: int = 0
    pruned: int = 0

    _low_watermark: float = 1.0
    _start: datetime | None = None
    _end: datetime | None = None

    def __init__(self, model: Module, tokenizer: TokenizerWrapper, prompt: list[int], signal: StopSignal) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.gen = BatchGenerator(model)
        self.signal = signal
        self._decoded_token_cache = {}

        uid = self.gen.insert([prompt], max_tokens=1)[0]

        root = Branch(prompt=prompt, answer=[])
        self._branch_lookup = {}
        self._branch_lookup[uid] = root
        self.branches = SortedList(key=lambda b: -b.probability)
        self.branches.add(root)

    def decode_token(self, token_id: int) -> str:
        cached = self._decoded_token_cache.get(token_id)
        if cached is not None:
            return cached
        decoded = self.tokenizer.decode([token_id], skip_special_tokens=True)  # type: ignore[call-arg]
        self._decoded_token_cache[token_id] = decoded
        return decoded

    @property
    def active(self) -> int:
        return len(self._branch_lookup)

    @property
    def queued(self) -> int:
        return len(self.gen.unprocessed_prompts)

    @property
    def n_finished(self) -> int:
        n_finished = 0
        for branch in self.branches:
            if branch.finish_reason == "eos_token":
                n_finished += 1
            else:
                break
        return n_finished

    def prune_branches(self):
        uids_to_remove: list[int] = []
        if self.n_finished >= args.n:
            for uid, branch in self._branch_lookup.items():
                if branch.probability < self._low_watermark:
                    self.pruned += 1
                    uids_to_remove.append(uid)
            self.gen.remove(uids_to_remove)
        for uid in uids_to_remove:
            branch = self._branch_lookup[uid]
            self.branches.remove(branch)
            del self._branch_lookup[uid]
        return uids_to_remove

    def start(self) -> Thread:
        self._start = datetime.now()

        def loop():
            responses: list[BatchGenerator.Response]
            while responses := self.gen.next():
                if self.signal.stopped:
                    break

                to_insert = []

                for r in responses:
                    self.tokens += 1
                    self.prune_branches()

                    if r.uid not in self._branch_lookup:
                        continue

                    branch = self._branch_lookup[r.uid]
                    del self._branch_lookup[r.uid]
                    self.branches.remove(branch)

                    for token in response_to_output_tokens(r):
                        new_branch = branch.with_new_token(token)

                        if new_branch.probability < args.min_probability:
                            self.pruned += 1
                            new_branch.finish_reason = "low_probability"
                            self.branches.add(new_branch)
                            continue

                        if token.token in self.tokenizer.eos_token_ids:
                            new_branch.finish_reason = "eos_token"
                            self._low_watermark = min(self._low_watermark, new_branch.probability)
                            self.branches.add(new_branch)
                            continue

                        to_insert.append((new_branch, r.prompt_cache))

                inputs = [b.tokens() for b, _ in to_insert]
                uids = self.gen.insert(inputs, max_tokens=1)
                for uid, (new_branch, _) in zip(uids, to_insert):
                    self._branch_lookup[uid] = new_branch
                    self.branches.add(new_branch)

        thread = Thread(target=loop)
        thread.start()
        return thread


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

    for i in range(args.n):
        if i >= len(walker.branches):
            table.add_row("", "", "")
            continue

        branch = walker.branches[i]
        answer_text = Text()
        for tok in branch.answer:
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
        else:
            status = Text("")

        table.add_row(status, probability_text, answer_text)

    return table


def render_stats_bar(walker: PromptTreeSearch) -> Table:
    elapsed = (datetime.now() - walker._start).total_seconds() if walker._start else 0.0
    tps = walker.tokens / elapsed if elapsed > 0 else 0.0
    left = f"active {walker.active}  queued {walker.queued}  pruned {walker.pruned} tps {tps:0.1f}"
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


def _strip_profile_args(argv: list[str]) -> list[str]:
    cleaned: list[str] = []
    it = iter(argv)
    for arg in it:
        if arg == "--profile":
            continue
        if arg == "--profile-native":
            continue
        if arg == "--profile-output":
            next(it, None)
            continue
        if arg.startswith("--profile-output="):
            continue
        if arg == "--profile-rate":
            next(it, None)
            continue
        if arg.startswith("--profile-rate="):
            continue
        cleaned.append(arg)
    return cleaned


def _maybe_run_py_spy_and_exit(parser: argparse.ArgumentParser) -> None:
    if not args.profile:
        return
    if os.environ.get("LLMTREE_PROFILE_CHILD") == "1":
        return

    py_spy = shutil.which("py-spy")
    if py_spy is None:
        parser.error(
            "--profile requires `py-spy` on PATH (try `brew install py-spy` or `cargo install py-spy`)"
        )

    child_env = os.environ.copy()
    child_env["LLMTREE_PROFILE_CHILD"] = "1"

    cmd = [
        py_spy,
        "record",
        "-o",
        args.profile_output,
        "--rate",
        str(args.profile_rate),
    ]
    if args.profile_native:
        cmd.append("--native")

    cmd.append("--")
    cmd.extend([sys.executable, __file__, *_strip_profile_args(sys.argv[1:])])

    raise SystemExit(subprocess.call(cmd, env=child_env))


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
parser.add_argument(
    "--profile",
    action="store_true",
    help="Run under py-spy and write a flamegraph SVG, then exit.",
)
parser.add_argument(
    "--profile-output",
    default="profile.svg",
    help="Where to write the flamegraph SVG.",
)
parser.add_argument(
    "--profile-rate",
    type=int,
    default=100,
    help="Sampling rate for py-spy (samples/sec).",
)
parser.add_argument(
    "--profile-native",
    action="store_true",
    help="Include native frames (py-spy --native).",
)
args = parser.parse_args()

if args.temperature <= 0:
    parser.error("--temperature must be > 0")
if not (0 < args.top_p <= 1):
    parser.error("--top-p must be in the range (0, 1]")

_maybe_run_py_spy_and_exit(parser)
main()
