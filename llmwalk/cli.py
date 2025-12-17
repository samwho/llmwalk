from __future__ import annotations

# Check for --offline flag before any HuggingFace imports
import os
import sys

if "--offline" in sys.argv:
    os.environ["HF_HUB_OFFLINE"] = "1"

import argparse
import csv
import io
import json
import time
from datetime import datetime
from importlib.metadata import version as pkg_version

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
        for tok in branch.answer_tokens():
            piece = walker.decode_token(tok.token)
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
                    "token": walker.decode_token(tok.token),
                    "probability": tok.prob,
                }
            )
        answer_text = "".join(t["token"] for t in tokens)
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
        answer_text = "".join(
            walker.decode_token(tok.token) for tok in branch.answer_tokens()
        )
        writer.writerow([answer_text, branch.probability, branch.finish_reason or ""])
    return output.getvalue()


def run() -> None:
    load_resp = load(args.model)
    model = load_resp[0]
    tokenizer = load_resp[1]

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

    if args.format:
        # Machine-readable output: no interactive display
        try:
            while not walker.should_stop():
                walker.step()
        except KeyboardInterrupt:
            walker.stop()

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
                while not walker.should_stop():
                    walker.step()
                    if args.stats_interval > 0 and time.monotonic() >= next_render:
                        live.update(render_view(walker))
                        next_render = time.monotonic() + interval
                live.update(render_view(walker))
        except KeyboardInterrupt:
            walker.stop()


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
        "-n",
        default=10,
        type=int,
        help="The top N answers to track. Search will stop after the top N answers have been found, so increasing this can increase runtime.",
    )
    parser.add_argument(
        "--min-probability",
        type=float,
        default=0.0001,
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
