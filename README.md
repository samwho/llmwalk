# llmwalk

Explore the answer-space for any prompt and any MLX-supported model. See
<https://huggingface.co/mlx-community/models> for supported models.

![Usage example gif](example1.gif)

Instead of sampling from the possible tokens each step, llmwalk branches out
and completes all of the branches the sampler would consider based on
`--top-k`, `--top-p` and `--temperature`, ranking the results by probability
as it goes.

The tree is walked prioritising the most likely branches, until it finds `-n`
branches and then it stops. It doesn't enumerate all possibilities, just enough
to know for sure it has found the `-n` most likely branches.

## Usage

- `uvx llmwalk -p "In what year was Barack Obama born?"`
- `uvx llmwalk -p "Write a haiku about compilers" -n 5`
- `uvx llmwalk -p "Give me one word: " --top-k 200 --temperature 0.7`

## Options

- `-p, --prompt TEXT`: Prompt to score (wrapped with the model’s chat template).
- `-m, --model MODEL`: MLX-LM model identifier or path (default: `mlx-community/Llama-3.2-1B-Instruct-4bit`), supported models can be found at <https://huggingface.co/mlx-community/models>
- `-n N`: Number of answers to show. The search stops once it has `N` finished answers and no unfinished branch can beat the worst of those `N`.
- `--min-probability FLOAT`: Any branch whose cumulative probability falls below this is marked finished (`low_probability`) and not expanded further.
- `--top-k INT`: At each step, expand at most `k` next tokens (highest probability).
- `--top-p FLOAT`: Nucleus cutoff applied *within the top-k tokens* at each step (keep adding tokens until cumulative probability ≥ `p`).
- `--temperature FLOAT`: Softmax temperature applied when computing per-step probabilities (`1.0` is the model distribution; must be `> 0`).
- `--stats-interval SECONDS`: How often to refresh the live view (`<= 0` disables periodic refresh; still renders at start/end).

Test
