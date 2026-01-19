from __future__ import annotations

# Check for --offline flag before any HuggingFace imports
import os
import sys

if "--offline" in sys.argv:
    os.environ["HF_HUB_OFFLINE"] = "1"

# Metal GPU trace capture requires this to be set at process start.
# See https://ml-explore.github.io/mlx/build/html/dev/metal_debugger.html
if any(
    arg == "--metal-capture" or arg.startswith("--metal-capture=") for arg in sys.argv
):
    os.environ.setdefault("MTL_CAPTURE_ENABLED", "1")

import argparse
import cProfile
import csv
import io
import json
import pstats
import time
from datetime import datetime
from importlib.metadata import version as pkg_version

import mlx.core as mx
from mlx_lm import load
from rich.console import Console, Group
from rich.live import Live
from rich.style import Style
from rich.table import Table
from rich.text import Text

from .llm import PromptTreeSearch, SearchConfig

args: argparse.Namespace

_BAND_COLORS = [
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
_BAND_STYLES = [Style(color=c) for c in _BAND_COLORS]


def style_for_token_probability(prob: float) -> Style:
    if prob != prob:  # NaN
        prob = 0.0
    elif prob < 0.0:
        prob = 0.0
    elif prob > 1.0:
        prob = 1.0

    band = min(int(prob * 10), 9)  # 0..9
    return _BAND_STYLES[band]


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


_PROBABILITY_LEGEND = render_probability_legend()


def render_branches(walker: PromptTreeSearch) -> Table:
    table = Table(expand=True, show_header=False, show_edge=False)
    table.add_column("Prob.", justify="right", no_wrap=True, width=8)
    table.add_column("Answer", ratio=1)

    branches = walker.top_branches(args.n)
    for i in range(args.n):
        if i >= len(branches):
            table.add_row("", "", "")
            continue

        branch = branches[i]
        answer_text = Text()

        if walker.is_sentencepiece_model():
            answer_text = Text(
                walker.tokenizer.decode([tok.token for tok in branch.answer_tokens()])  # type: ignore[call-arg]
            )
        else:
            for tok in branch.answer_tokens():
                piece = walker.tokenizer.decode(  # type: ignore[call-arg]
                    [tok.token],
                    skip_special_tokens=True,
                )
                if not piece:
                    continue
                piece = piece.replace("\n", "\\n")
                answer_text.append(piece, style=style_for_token_probability(tok.prob))

        status: Text
        if branch.finish_reason == "eos_token":
            status = Text("✓", style="green")
        elif branch.finish_reason == "low_probability":
            status = Text("?", style="yellow")
        elif branch.finish_reason == "pruned":
            status = Text("-", style="dim")
        else:
            status = Text(" ")

        probability_text = Text.assemble(status, f"{branch.probability * 100:6.2f}%")

        table.add_row(probability_text, answer_text)

    return table


def render_stats_bar(walker: PromptTreeSearch) -> Table:
    elapsed = (datetime.now() - walker._start).total_seconds() if walker._start else 0.0
    tps = walker.tokens / elapsed if elapsed > 0 else 0.0
    left = f"frontier {walker.active} pruned {walker.pruned} tps {tps:0.1f}"
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
    if args.minimal:
        return Group(render_branches(walker))
    return Group(
        _PROBABILITY_LEGEND,
        render_branches(walker),
        render_stats_bar(walker),
    )


def format_results_json(walker: PromptTreeSearch) -> str:
    branches = walker.top_branches(args.n)
    results = []
    for branch in branches:
        tokens = []
        for tok in branch.answer_tokens():
            tokens.append(
                {
                    "token": walker.tokenizer.decode([tok.token]),  # type: ignore[call-arg]
                    "probability": tok.prob,
                }
            )
        answer_text = walker.tokenizer.decode(  # type: ignore[attr-defined]
            [tok.token for tok in branch.answer_tokens()]
        )
        results.append(
            {
                "answer": answer_text,
                "probability": branch.probability,
                "finish_reason": branch.finish_reason,
                "tokens": tokens,
            }
        )
    return json.dumps(results, indent=2)


def format_results_csv(walker: PromptTreeSearch) -> str:
    branches = walker.top_branches(args.n)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["answer", "probability", "finish_reason"])
    for branch in branches:
        answer_text = walker.tokenizer.decode(  # type: ignore[attr-defined]
            [tok.token for tok in branch.answer_tokens()]
        )
        writer.writerow([answer_text, branch.probability, branch.finish_reason or ""])
    return output.getvalue()


def run() -> None:
    load_resp = load(args.model, revision=args.revision)
    model = load_resp[0]
    tokenizer = load_resp[1]

    model.eval()

    prompt = tokenizer.apply_chat_template(  # type: ignore[call-arg]
        [{"role": "user", "content": args.prompt}],
        add_generation_prompt=True,
    )

    config = SearchConfig(
        n=args.n,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        min_probability=args.min_probability,
    )

    walker = PromptTreeSearch(model, tokenizer, prompt, config)

    profiler: cProfile.Profile | None = None
    if args.cprofile is not None:
        profiler = cProfile.Profile()

    capture_enabled = False
    if args.metal_capture is not None and hasattr(mx, "metal"):
        start_capture = getattr(mx.metal, "start_capture", None)
        if callable(start_capture):
            try:
                start_capture(args.metal_capture)
                capture_enabled = True
            except Exception as exc:
                print(
                    f"Warning: failed to start Metal capture: {exc}",
                    file=sys.stderr,
                )

    steps = 0

    def do_step_loop() -> None:
        nonlocal steps
        try:
            while not walker.should_stop():
                walker.step()
                steps += 1
                if args.max_steps > 0 and steps >= args.max_steps:
                    walker.stop()
                    break
        except KeyboardInterrupt:
            walker.stop()

    try:
        if args.format:
            # Machine-readable output: no interactive display
            try:
                if profiler is not None:
                    profiler.enable()
                do_step_loop()
            finally:
                if profiler is not None:
                    profiler.disable()

            if args.format == "json":
                print(format_results_json(walker))
            elif args.format == "csv":
                print(format_results_csv(walker), end="")
        else:
            # Interactive display
            console = Console()
            try:
                with Live(console=console, transient=False) as live:
                    interval = max(0.1, args.stats_interval)
                    next_render = time.monotonic()
                    live.update(render_view(walker))
                    try:
                        if profiler is not None:
                            profiler.enable()
                        while not walker.should_stop():
                            walker.step()
                            steps += 1
                            if args.max_steps > 0 and steps >= args.max_steps:
                                walker.stop()
                                break
                            if (
                                args.stats_interval > 0
                                and time.monotonic() >= next_render
                            ):
                                live.update(render_view(walker))
                                next_render = time.monotonic() + interval
                    finally:
                        if profiler is not None:
                            profiler.disable()
                    live.update(render_view(walker))
            except KeyboardInterrupt:
                walker.stop()
    finally:
        if capture_enabled and hasattr(mx, "metal"):
            stop_capture = getattr(mx.metal, "stop_capture", None)
            if callable(stop_capture):
                try:
                    stop_capture()
                except Exception as exc:
                    print(
                        f"Warning: failed to stop Metal capture: {exc}",
                        file=sys.stderr,
                    )

    if profiler is not None:
        if args.cprofile == "-":
            stream = io.StringIO()
            stats = pstats.Stats(profiler, stream=stream)
            stats.strip_dirs().sort_stats("cumtime").print_stats(50)
            print(stream.getvalue(), file=sys.stderr, end="")
        else:
            profiler.dump_stats(args.cprofile)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--prompt",
        default="What is 2+2?",
        help="The prompt to walk. Can be a file path, in which case the file contents will be used.",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="mlx-community/Llama-3.2-1B-Instruct-4bit",
        help="Which model to use. Must be an mlx-community/ model from HuggingFace.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional HuggingFace revision (branch, tag, or commit hash) for the model.",
    )
    parser.add_argument(
        "-n",
        default=10,
        type=int,
        help="The top N answers to track. Search will stop after the top N answers have been found, so increasing this can increase runtime.",
    )
    parser.add_argument(
        "--min-probability",
        type=float,
        default=0.001,
        help="A minimum probability threshold for branches. If a branch becomes less likely than this, we stop walking it. Lowering this can increase runtime.",
    )
    parser.add_argument(
        "--top-k",
        dest="top_k",
        default=50,
        type=int,
        help="How many tokens to branch on at each step. Increasing this will increase runtime.",
    )
    parser.add_argument(
        "--top-p",
        dest="top_p",
        default=1.0,
        type=float,
        help="Like --top-k, this will limit the tokens branched on at each step, but by cumulative probability instead of a static number. Decreasing this can reduce runtime.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help='Sampling temperature, decreasing this will "sharpen" the token probabilities and make high-probability tokens more likely. Increasing it will make the distribution more uniform, making less likely tokens more likely.',
    )
    parser.add_argument(
        "--stats-interval",
        type=float,
        default=0.1,
        help="In interactive mode, i.e. no --format, this will control how often the table is updated.",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "json"],
        default=None,
        help="Output format for machine-readable output (disables interactive display)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Stop after this many search steps (0 = no limit). Useful for profiling and quick runs.",
    )
    parser.add_argument(
        "--cprofile",
        default=None,
        help="Write a Python cProfile to this path (use '-' to print top stats to stderr).",
    )
    parser.add_argument(
        "--metal-capture",
        default=None,
        help="Write a Metal capture trace to this path (macOS only).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run in offline mode (skip HuggingFace Hub network requests)",
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Hide the legend and stats bar for a cleaner display",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {pkg_version('llmwalk')}",
    )

    raw = list(sys.argv[1:] if argv is None else argv)
    filtered = [a for a in raw if a != "--"]
    parsed = parser.parse_args(filtered)

    if parsed.temperature <= 0:
        parser.error("--temperature must be > 0")
    if not (0 < parsed.top_p <= 1):
        parser.error("--top-p must be in the range (0, 1]")
    if parsed.max_steps < 0:
        parser.error("--max-steps must be >= 0")

    # If prompt is a file path, read the file contents
    from pathlib import Path

    prompt_path = Path(parsed.prompt)
    if prompt_path.is_file():
        parsed.prompt = prompt_path.read_text()

    return parsed


def main(argv: list[str] | None = None) -> None:
    global args
    args = parse_args(argv)
    run()
