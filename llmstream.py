# /// script
# dependencies = [
#   "mlx-lm==0.28.4",
# ]
# ///
from __future__ import annotations

import argparse
import sys
from typing import Any

from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler


def _build_prompt(tokenizer: Any, prompt: str, use_chat_template: bool) -> str | list[int]:
    if prompt == "-":
        prompt = sys.stdin.read()

    if not use_chat_template:
        return prompt

    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if apply_chat_template is None:
        return prompt

    try:
        return apply_chat_template(  # type: ignore[misc]
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
        )
    except Exception:
        return prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prompt", default="Write me a very long, funny poem", help="Prompt text (or '-' for stdin)")
    parser.add_argument("-m", "--model", default="mlx-community/Llama-3.2-1B-Instruct-4bit")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum generation tokens")
    parser.add_argument("--temperature", type=float, default=0.8, help="0 for greedy decoding")
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling threshold (0 < p <= 1)")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling (0 disables)")
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Do not wrap the prompt in the tokenizer chat template.",
    )
    args = parser.parse_args()

    if args.max_tokens <= 0:
        parser.error("--max-tokens must be > 0")
    if args.temperature < 0:
        parser.error("--temperature must be >= 0")
    if not (0 < args.top_p <= 1):
        parser.error("--top-p must be in the range (0, 1]")
    if args.top_k < 0:
        parser.error("--top-k must be >= 0")

    model, tokenizer = load(args.model)

    prompt = _build_prompt(tokenizer, args.prompt, use_chat_template=not args.no_chat_template)
    sampler = make_sampler(temp=args.temperature, top_p=args.top_p, top_k=args.top_k)

    last = None
    try:
        for resp in stream_generate(model, tokenizer, prompt, max_tokens=args.max_tokens, sampler=sampler):
            last = resp
            if resp.text:
                print(resp.text, end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        print(flush=True)
        if last is not None:
            print(f"{last.generation_tps:.2f} tokens/s", file=sys.stderr)


if __name__ == "__main__":
    main()
