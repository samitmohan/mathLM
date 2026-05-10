"""Multi-domain evaluation: GSM8K + calculus + ML/DL math.

    python -m mathlm.eval.math --checkpoint checkpoint_mathlm.pt --n 200
    python -m mathlm.eval.math --compare ckpt_a.pt ckpt_b.pt --n 200
    python -m mathlm.eval.math --domain gsm8k --n 100 --verbose
"""

import os
import re
import argparse
import random

from rich.console import Console
from rich.table import Table
from rich.progress import track

from mathlm.infer.inference import load_model, generate_text

console = Console()



def extract_number(text: str) -> str | None:
    m = re.search(r'####\s*(-?[\d,]+)', text)
    if m:
        return m.group(1).replace(',', '')
    nums = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', text)
    if nums:
        return nums[-1].replace(',', '')
    return None


def normalize_expr(text: str) -> str:
    """Normalize a math expression for comparison."""
    text = text.strip().lower()
    text = text.replace('**', '^').replace('*', '').replace(' ', '')
    return text.split('\n')[0]  # first line only



FEW_SHOT_GSM8K = """\
Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
A: Natalia sold 48/2 = 24 clips in May. Natalia sold 48+24 = 72 clips altogether in April and May. #### 72

Q: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
A: Betty has only 100/2 = $50. Betty's grandparents gave her 15*2 = $30. Betty needs 100-50-15-30 = $5 more. #### 5

"""



def eval_gsm8k(model, enc, device, n=None, verbose=False) -> dict:
    """
    Evaluate on the GSM8K test set (1,319 problems, never seen during training).
    Returns accuracy dict with 'correct', 'total', 'accuracy'.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        console.print("[red]pip install datasets[/red]")
        return {}

    ds = load_dataset("openai/gsm8k", "main", split="test")
    if n:
        ds = ds.select(range(min(n, len(ds))))

    correct = 0
    results = []

    for ex in track(ds, description="[cyan]GSM8K[/cyan]", console=console):
        prompt   = FEW_SHOT_GSM8K + f"Q: {ex['question']}\nA:"
        response = generate_text(model, enc, device, prompt,
                                  max_new_tokens=200, temperature=0.1, top_k=1,
                                  stop_at_newline=False)
        gold_ans = extract_number(ex["answer"])
        pred_ans = extract_number(response)
        ok = (pred_ans is not None) and (pred_ans == gold_ans)
        correct += int(ok)
        results.append({"gold": gold_ans, "pred": pred_ans, "correct": ok,
                         "response": response[:100]})
        if verbose and not ok:
            console.print(f"  [red]✗[/red] gold={gold_ans}  pred={pred_ans}  |  {response[:60]}")

    acc = correct / len(results) * 100 if results else 0.0
    return {"correct": correct, "total": len(results), "accuracy": acc}



def eval_calculus(model, enc, device, n=200, verbose=False) -> dict:
    """
    Evaluate on a held-out slice of the synthetic calculus Q/A pairs.
    Generates from scratch each run (deterministic, seeded).

    Answer matching: normalize both strings and check exact match,
    then check if gold expression appears as substring of generated text.
    """
    from mathlm.data.generate_calculus import build_pairs as calc_build

    all_pairs = calc_build(seed=99)  # different seed from training seed (42)
    # Take from the END — training uses first 90%, we use last 10% as held-out
    cut = int(0.9 * len(all_pairs))
    held_out_raw = all_pairs[cut:]

    # Parse back to (question, answer) tuples
    pairs = []
    for text in held_out_raw:
        if not text.startswith("Q: "):
            continue
        body = text[3:]
        parts = body.split("\nA: ", 1)
        if len(parts) != 2:
            continue
        q = parts[0].strip()
        a = parts[1].strip().split('\n')[0]  # key expression is the first line
        if len(a) < 50:  # only short, unambiguous answers
            pairs.append((q, a))

    if n:
        random.seed(42)
        pairs = random.sample(pairs, min(n, len(pairs)))

    correct = 0
    for question, gold in track(pairs, description="[cyan]Calculus[/cyan]", console=console):
        prompt   = f"Q: {question}\nA:"
        response = generate_text(model, enc, device, prompt,
                                  max_new_tokens=60, temperature=0.1, top_k=1,
                                  stop_at_newline=False)
        gold_n = normalize_expr(gold)
        resp_n = normalize_expr(response)
        ok = (gold_n == resp_n) or (gold_n in resp_n)
        correct += int(ok)
        if verbose and not ok:
            console.print(f"  [red]✗[/red] gold={gold!r}  got={response[:50]!r}")

    acc = correct / len(pairs) * 100 if pairs else 0.0
    return {"correct": correct, "total": len(pairs), "accuracy": acc}



def eval_ml_math(model, enc, device, n=100, verbose=False) -> dict:
    """
    Evaluate on a held-out slice of the synthetic ML/DL math Q/A pairs.

    Answer matching is softer than calculus: we check if key terms from the
    gold answer appear in the generated text. ML math answers are longer
    (gradient expressions with multiple terms) so exact match is too strict.
    """
    from mathlm.data.generate_ml_math import build_pairs as ml_build

    all_pairs = ml_build(seed=99)  # different seed from training
    # Parse back to (question, answer) tuples
    pairs = []
    for text in all_pairs:
        if not text.startswith("Q: "):
            continue
        body = text[3:]
        parts = body.split("\nA: ", 1)
        if len(parts) != 2:
            continue
        q = parts[0].strip()
        a = parts[1].strip()
        pairs.append((q, a))

    cut = int(0.9 * len(pairs))
    held_out = pairs[cut:]

    if n:
        random.seed(42)
        held_out = random.sample(held_out, min(n, len(held_out)))

    correct = 0
    for question, gold in track(held_out, description="[cyan]ML math[/cyan]", console=console):
        prompt   = f"Q: {question}\nA:"
        response = generate_text(model, enc, device, prompt,
                                  max_new_tokens=150, temperature=0.1, top_k=1,
                                  stop_at_newline=False)

        # Extract key terms from gold: math symbols, operator names, key numbers
        # A "correct" answer contains at least the primary result expression
        gold_first_line = gold.split('.')[0].strip()  # main result is in the first sentence
        gold_terms = [t for t in re.split(r'[\s,;]+', gold_first_line) if len(t) > 2]

        # Check if most key terms appear in the response
        n_terms = len(gold_terms)
        if n_terms == 0:
            continue
        n_matched = sum(1 for t in gold_terms if t.lower() in response.lower())
        ok = (n_matched / n_terms) >= 0.5  # ≥50% of key terms present

        correct += int(ok)
        if verbose and not ok:
            console.print(f"  [red]✗[/red] ({n_matched}/{n_terms} terms)  got={response[:60]!r}")

    acc = correct / len(held_out) * 100 if held_out else 0.0
    return {"correct": correct, "total": len(held_out), "accuracy": acc}



def print_report(checkpoint: str, results: dict[str, dict], compare_results: dict | None = None):
    title = f"MathLM Evaluation — {os.path.basename(checkpoint)}"
    table = Table(title=title, show_header=True)
    table.add_column("Domain",    style="bold")
    table.add_column("Correct")
    table.add_column("Total")
    table.add_column("Accuracy")
    if compare_results:
        table.add_column("Compare")
        table.add_column("Δ")

    for domain, r in results.items():
        if not r:
            continue
        acc = r["accuracy"]
        color = "green" if acc > 5 else "yellow" if acc > 2 else "red"
        row = [domain, str(r["correct"]), str(r["total"]), f"[{color}]{acc:.1f}%[/{color}]"]
        if compare_results and domain in compare_results:
            cr = compare_results[domain]
            cacc = cr["accuracy"]
            delta = cacc - acc
            delta_str = f"[green]+{delta:.1f}%[/green]" if delta > 0 else f"[red]{delta:.1f}%[/red]"
            row += [f"{cacc:.1f}%", delta_str]
        table.add_row(*row)

    console.print(table)



def run_all(checkpoint: str, domain: str, n: int, verbose: bool) -> dict[str, dict]:
    model, enc, config, device = load_model(ckpt_file=checkpoint)
    results = {}

    if domain in ("all", "gsm8k"):
        results["gsm8k"]    = eval_gsm8k(model, enc, device, n=n, verbose=verbose)
    if domain in ("all", "calculus"):
        results["calculus"] = eval_calculus(model, enc, device, n=n, verbose=verbose)
    if domain in ("all", "ml_math"):
        results["ml_math"]  = eval_ml_math(model, enc, device, n=n, verbose=verbose)

    return results


def main():
    parser = argparse.ArgumentParser(description="MathLM multi-domain evaluation")
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint to evaluate (auto-detects if omitted)")
    parser.add_argument("--compare",    nargs=2, metavar=("CKPT_A", "CKPT_B"), default=None,
                        help="Compare two checkpoints side by side")
    parser.add_argument("--domain",     default="all",
                        choices=["all", "gsm8k", "calculus", "ml_math"])
    parser.add_argument("--n",          type=int, default=None,
                        help="Max problems per domain (default: full test set)")
    parser.add_argument("--verbose",    action="store_true",
                        help="Print wrong answers")
    args = parser.parse_args()

    if args.compare:
        ckpt_a, ckpt_b = args.compare
        console.print(f"\n[bold]Checkpoint A:[/bold] {ckpt_a}")
        results_a = run_all(ckpt_a, args.domain, args.n, args.verbose)
        console.print(f"\n[bold]Checkpoint B:[/bold] {ckpt_b}")
        results_b = run_all(ckpt_b, args.domain, args.n, args.verbose)
        print_report(ckpt_a, results_a)
        print_report(ckpt_b, results_b, compare_results=results_a)
        console.print("\n[dim]Δ = B accuracy − A accuracy (positive = B improved)[/dim]")
    else:
        ckpt = args.checkpoint
        results = run_all(ckpt or "auto", args.domain, args.n, args.verbose)
        print_report(ckpt or "auto", results)


if __name__ == "__main__":
    main()
