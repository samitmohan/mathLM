from __future__ import annotations

"""GSM8K evaluation harness with 2-shot CoT prompting.

GPT-2 baseline is ~2% accuracy.

    python -m mathlm.eval.gsm8k                  # full test set (1319 problems)
    python -m mathlm.eval.gsm8k --n 100          # quick run
    python -m mathlm.eval.gsm8k --verbose
"""

import re
import argparse
import torch
from rich.console import Console
from rich.table import Table
from rich.progress import track
from mathlm.infer.inference import load_model, generate_text

console = Console()


def extract_number(text: str) -> str | None:
    """Extract a numerical answer: prefers '#### N' format, else last number."""
    m = re.search(r'####\s*(-?[\d,]+)', text)
    if m:
        return m.group(1).replace(',', '')
    nums = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', text)
    if nums:
        return nums[-1].replace(',', '')
    return None


# 2-shot CoT prompt: ~130 tokens. Leaves ~280 tokens for the model's CoT in a 512-tok ctx.
FEW_SHOT = """\
Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
A: Natalia sold 48/2 = 24 clips in May. Natalia sold 48+24 = 72 clips altogether in April and May. #### 72

Q: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
A: Betty has only 100/2 = $50. Betty's grandparents gave her 15*2 = $30. Betty needs 100-50-15-30 = $5 more. #### 5

"""


def run_eval(
    checkpoint: str | None = None,
    n_problems: int | None = None,
    temperature: float = 0.1,
    max_new_tokens: int = 200,
    verbose: bool = False,
):
    from datasets import load_dataset
    model, enc, config, device = load_model(ckpt_file=checkpoint)
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if n_problems:
        ds = ds.select(range(min(n_problems, len(ds))))

    correct = 0
    total = 0
    results = []

    for example in track(ds, description="Evaluating", console=console):
        question   = example["question"]
        gold_text  = example["answer"]
        gold_ans   = extract_number(gold_text)

        prompt   = FEW_SHOT + f"Q: {question}\nA:"
        response = generate_text(model, enc, device, prompt,
                                  max_new_tokens=max_new_tokens,
                                  temperature=temperature,
                                  top_k=1,           # near-greedy for eval
                                  top_p=1.0,
                                  stop_at_newline=False)
        pred_ans = extract_number(response)

        is_correct = (pred_ans is not None) and (pred_ans == gold_ans)
        correct += int(is_correct)
        total   += 1

        results.append({
            "question": question[:80],
            "gold": gold_ans,
            "pred": pred_ans,
            "response": response[:120],
            "correct": is_correct,
        })

        if verbose:
            status = "[green]✓[/green]" if is_correct else "[red]✗[/red]"
            console.print(f"{status} gold={gold_ans}  pred={pred_ans}  |  {response[:80]}")

    acc = correct / total * 100 if total > 0 else 0.0

    table = Table(title=f"GSM8K Results — {total} problems", show_header=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Correct",  str(correct))
    table.add_row("Total",    str(total))
    color = 'green' if acc > 2 else 'yellow'
    table.add_row("Accuracy", f"[bold {color}]{acc:.1f}%[/bold {color}]")
    table.add_row("GPT-2 baseline", "~2.0%")
    console.print(table)

    return acc, results


def main():
    parser = argparse.ArgumentParser(description="GSM8K evaluation harness")
    parser.add_argument("--checkpoint",   default=None)
    parser.add_argument("--n",            type=int,   default=None, help="Limit to first N problems")
    parser.add_argument("--temperature",  type=float, default=0.1)
    parser.add_argument("--max-tokens",   type=int,   default=200)
    parser.add_argument("--verbose",      action="store_true")
    args = parser.parse_args()

    run_eval(
        checkpoint=args.checkpoint,
        n_problems=args.n,
        temperature=args.temperature,
        max_new_tokens=args.max_tokens,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
