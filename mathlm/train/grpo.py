"""GRPO — Group Relative Policy Optimization, applied after SFT.

For each question, sample N candidate answers, compute a binary reward
(answer matches gold?), and update the policy by group-normalising
rewards as advantages. A KL penalty against a frozen reference model
keeps the policy from drifting too far from SFT. No critic, no reward
model — math has verifiable rewards.

Input:  checkpoint_mathlm.pt
Output: checkpoint_mathlm_grpo.pt

    python -m mathlm.train.grpo
    python -m mathlm.train.grpo --steps 1000 --lr 1e-5 --n-candidates 4
"""

import re
import sys
import math
import os
import time
import random
import argparse

import torch
import torch.nn.functional as F
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from mathlm.model.gpt import GPT
from mathlm.model.tokenizer import MathTokenizer
import tiktoken

console = Console()

try:
    import wandb
    WANDB = True
except ImportError:
    WANDB = False



def make_tokenizer():
    if MathTokenizer.is_available():
        tok = MathTokenizer(); tok.load()
        return tok
    return tiktoken.get_encoding("gpt2")


def _encode(enc, text: str) -> list[int]:
    try:
        return enc.encode(text)
    except TypeError:
        return enc.encode(text, disallowed_special=())


def _decode(enc, ids: list[int]) -> str:
    try:
        return enc.decode(ids)
    except Exception:
        return ""



def extract_number(text: str) -> str | None:
    """
    Extract the final numerical answer from generated text.
    Looks for '#### N' format first (GSM8K convention), then falls back to
    the last standalone number in the string.
    Reusing the same logic as eval_gsm8k.py for consistency.
    """
    m = re.search(r'####\s*(-?[\d,]+)', text)
    if m:
        return m.group(1).replace(',', '')
    nums = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', text)
    if nums:
        return nums[-1].replace(',', '')
    return None


def normalize_math_answer(text: str) -> str:
    """Normalize a math expression for comparison: strip spaces, unify notation."""
    text = text.strip().lower()
    text = text.replace('**', '^').replace(' ', '')
    # Remove trailing \n or extra whitespace
    return text.split('\n')[0].strip()


def compute_reward(generated: str, gold: str, problem_type: str = "gsm8k") -> float:
    """
    Binary reward: 1.0 if the generated answer matches gold, 0.0 otherwise.

    For GSM8K/word problems: extract '#### N' and compare numerically.
    For calculus: normalize both strings and compare (e.g. '5x^4' == '5X^4').
    """
    if problem_type == "gsm8k":
        pred = extract_number(generated)
        gold_num = extract_number(gold)
        return 1.0 if (pred is not None and gold_num is not None and pred == gold_num) else 0.0
    elif problem_type == "calculus":
        pred = normalize_math_answer(generated)
        gold_norm = normalize_math_answer(gold)
        # exact match after normalization
        if pred == gold_norm:
            return 1.0
        # check if gold expression appears anywhere in the generated text
        if gold_norm in pred:
            return 1.0
        return 0.0
    return 0.0



def load_training_data() -> list[dict]:
    """
    Load GRPO training questions from two verifiable sources:
      1. GSM8K train split — grade-school word problems with '#### N' gold answers
      2. Calculus Q/A — derivatives/integrals with exact symbolic gold answers

    Only problems with VERIFIABLE answers are used for GRPO.
    The MATH dataset (LaTeX answers) is skipped for v1 — LaTeX normalization is complex.
    """
    data = []

    # GSM8K — most reliable reward signal (#### N format is unambiguous)
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="train")
        for ex in ds:
            data.append({
                "question": ex["question"],
                "gold":     ex["answer"],
                "type":     "gsm8k",
            })
        console.print(f"  [dim]GSM8K: {len(data):,} questions loaded[/dim]")
    except Exception as e:
        console.print(f"  [yellow]GSM8K load failed: {e}[/yellow]")

    # Calculus Q/A (generated, clean symbolic answers)
    try:
        from mathlm.data.generate_calculus import build_pairs as calc_build
        calc_pairs = calc_build()
        for text in calc_pairs:
            if not text.startswith("Q: "):
                continue
            body = text[3:]
            parts = body.split("\nA: ", 1)
            if len(parts) != 2:
                continue
            q = parts[0].strip()
            a = parts[1].strip().split('\n')[0]  # first line of answer = the key expression
            # Only include pairs where the gold answer is short and unambiguous (< 40 chars)
            # Longer answers (multi-sentence explanations) are harder to exact-match
            if len(a) < 40:
                data.append({"question": q, "gold": a, "type": "calculus"})
        console.print(f"  [dim]Total after calculus: {len(data):,} questions[/dim]")
    except Exception as e:
        console.print(f"  [yellow]Calculus data load failed: {e}[/yellow]")

    random.shuffle(data)
    return data



def compute_sequence_logprob(
    model,
    input_ids: torch.Tensor,
    answer_start: int,
    autocast_dtype,
    device: str,
) -> torch.Tensor:
    """
    Compute the sum of log-probabilities for answer tokens using teacher-forcing.

    input_ids: (1, seq_len) — full Q/A sequence (prompt + answer)
    answer_start: index in input_ids where the answer begins

    The model sees input_ids[:-1] and must predict input_ids[1:].
    Answer tokens are at positions [answer_start-1 : -1] in the input.

    Returns a scalar tensor (requires_grad=True for the policy model).
    """
    with torch.autocast(device_type=device, dtype=autocast_dtype):
        logits, _ = model(input_ids)  # (1, seq_len, vocab)

    log_probs = F.log_softmax(logits[0], dim=-1)  # (seq_len, vocab)

    # Positions [answer_start-1 : seq_len-1] in log_probs predict tokens [answer_start : seq_len]
    ans_start_lp = max(0, answer_start - 1)
    ans_logprobs = log_probs[ans_start_lp:-1]          # (n_answer, vocab)
    ans_tokens   = input_ids[0, answer_start:]          # (n_answer,)

    if ans_tokens.numel() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Gather log-prob for each actual answer token and sum
    token_logprobs = ans_logprobs.gather(1, ans_tokens.unsqueeze(1)).squeeze(1)
    return token_logprobs.sum()



@torch.no_grad()
def generate_candidate(model, prompt_ids: list[int], device: str, max_new_tokens: int,
                        temperature: float, top_k: int, enc) -> str:
    """
    Generate one candidate answer using the policy model.
    Uses @torch.no_grad() — generation is separate from the backward pass.
    Log-probs are recomputed with teacher-forcing in the backward phase.
    """
    tok = torch.tensor([prompt_ids], device=device)
    generated_ids = []
    for tid in model.stream_generate(tok, max_new_tokens=max_new_tokens,
                                      temperature=temperature, top_k=top_k):
        generated_ids.append(tid)
        if len(generated_ids) >= max_new_tokens:
            break
    return _decode(enc, generated_ids)



def main():
    parser = argparse.ArgumentParser(description="GRPO reasoning training for MathLM")
    parser.add_argument("--checkpoint",   default="checkpoint_mathlm.pt",
                        help="SFT checkpoint to start from")
    parser.add_argument("--output",       default="checkpoint_mathlm_grpo.pt")
    parser.add_argument("--steps",        type=int,   default=500)
    parser.add_argument("--lr",           type=float, default=1e-5,
                        help="Policy LR — lower than SFT; RL updates are noisy")
    parser.add_argument("--beta",         type=float, default=0.04,
                        help="KL penalty weight (DeepSeek-R1 default=0.04). "
                             "Larger = stay closer to SFT checkpoint.")
    parser.add_argument("--n-candidates", type=int,   default=8,
                        help="Candidates generated per question. More = better advantage estimates, slower.")
    parser.add_argument("--n-questions",  type=int,   default=8,
                        help="Questions per GRPO step.")
    parser.add_argument("--max-tokens",   type=int,   default=200)
    parser.add_argument("--temperature",  type=float, default=0.8,
                        help="Generation temperature. Must be > 0 for diverse candidates.")
    parser.add_argument("--top-k",        type=int,   default=50)
    parser.add_argument("--eval-interval",type=int,   default=50)
    args = parser.parse_args()

    torch.manual_seed(42)
    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    enc = make_tokenizer()

    if not os.path.exists(args.checkpoint):
        console.print(f"[red]Error: {args.checkpoint} not found — run mathlm.train.sft first.[/red]")
        return

    ckpt   = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]

    # Policy model — trained during GRPO
    policy = GPT(config).to(device)
    policy.load_state_dict(ckpt["model"], strict=False)
    policy.train()

    # Reference model — FROZEN copy of the SFT checkpoint.
    # The KL penalty keeps the policy from drifting too far from the SFT checkpoint,
    # which prevents mode collapse and catastrophic forgetting of the format learned in SFT.
    ref = GPT(config).to(device)
    ref.load_state_dict(ckpt["model"], strict=False)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    params = sum(p.numel() for p in policy.parameters()) / 1e6
    console.print(
        f"[dim]GRPO  params={params:.1f}M  device={device}  steps={args.steps}  "
        f"N={args.n_candidates}  beta={args.beta}  lr={args.lr:.0e}[/dim]"
    )

    train_data = load_training_data()
    if not train_data:
        console.print("[red]No training data found.[/red]")
        return
    console.print(f"[dim]{len(train_data):,} questions available[/dim]")

    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0,
    )

    seq_len = config.sequence_length

    if WANDB:
        wandb.init(
            project="nanochat",
            name=f"grpo-{params:.0f}M-{args.steps}steps",
            config={
                "steps": args.steps, "lr": args.lr, "beta": args.beta,
                "n_candidates": args.n_candidates, "n_questions": args.n_questions,
                "params_M": params, "max_tokens": args.max_tokens,
            },
            resume="allow",
        )

    step = 0
    data_idx = 0
    t_start = time.time()
    reward_history = []

    with Progress(
        TextColumn("[bold cyan]GRPO[/bold cyan]"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%  step {task.completed}/{task.total}"),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=2,
    ) as progress:
        task = progress.add_task("grpo", total=args.steps)

        while step < args.steps:
            batch = []
            for _ in range(args.n_questions):
                batch.append(train_data[data_idx % len(train_data)])
                data_idx += 1

            # Generation is separated from the backward pass.
            # The policy model is in eval mode during generation to disable dropout.
            # We'll recompute log-probs with teacher-forcing in Phase 2.
            policy.eval()
            all_candidates  = []  # list of (question, gold, type, candidate_text, prompt_ids, answer_start)
            all_rewards     = []  # shape: [n_questions * N]

            for item in batch:
                prompt = f"Q: {item['question']}\nA:"
                prompt_ids = _encode(enc, prompt)
                prefix_len = len(prompt_ids)

                candidates_for_q = []
                rewards_for_q    = []

                for _ in range(args.n_candidates):
                    candidate_text = generate_candidate(
                        policy, prompt_ids, device,
                        max_new_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        enc=enc,
                    )
                    reward = compute_reward(candidate_text, item["gold"], item["type"])
                    candidates_for_q.append((candidate_text, prompt_ids, prefix_len))
                    rewards_for_q.append(reward)

                all_candidates.extend([(item["question"], item["gold"], item["type"],
                                        c, pid, plen) for c, pid, plen in candidates_for_q])
                all_rewards.extend(rewards_for_q)

            # Reshape to (n_questions, N) to normalize within each question's group.
            # This is the GRPO key idea: advantages are RELATIVE within a group,
            # not absolute. A question where all N answers are wrong → all advantages ≈ 0
            # (no useful signal, no destructive update). A mixed group → clear signal.
            rewards_t = torch.tensor(all_rewards, dtype=torch.float32)  # (n_q * N,)
            rewards_m = rewards_t.view(args.n_questions, args.n_candidates)
            group_mean = rewards_m.mean(dim=1, keepdim=True)            # (n_q, 1)
            group_std  = rewards_m.std(dim=1, keepdim=True) + 1e-8      # (n_q, 1)
            advantages = ((rewards_m - group_mean) / group_std).view(-1)  # (n_q * N,)

            mean_reward = rewards_t.mean().item()
            reward_history.append(mean_reward)

            # Re-run the policy with teacher-forcing on each (question, candidate) pair.
            # This recomputes log-probs with gradients so we can do the backward pass.
            policy.train()
            optimizer.zero_grad()
            total_loss = 0.0
            n_valid    = 0

            for idx, (question, gold, prob_type, candidate, prompt_ids, prefix_len) in enumerate(all_candidates):
                advantage = advantages[idx].item()

                # Skip zero-advantage candidates — no learning signal.
                # This happens when all N candidates are correct (or all wrong).
                if abs(advantage) < 1e-6:
                    continue

                # Build full Q/A sequence for teacher-forcing
                full_text = f"Q: {question}\nA: {candidate}\n\n"
                full_ids  = _encode(enc, full_text)

                if len(full_ids) > seq_len:
                    full_ids = full_ids[:seq_len]  # truncate to context window

                input_t = torch.tensor([full_ids], dtype=torch.long, device=device)

                # answer_start: position in full_ids where answer tokens begin.
                # prefix_len is the length of "Q: {question}\nA:" in tokens.
                answer_start = min(prefix_len, len(full_ids) - 1)

                # Policy log-prob (with gradient)
                policy_logp = compute_sequence_logprob(
                    policy, input_t, answer_start, autocast_dtype, device
                )

                # Reference log-prob (no gradient)
                with torch.no_grad():
                    ref_logp = compute_sequence_logprob(
                        ref, input_t, answer_start, autocast_dtype, device
                    )

                # KL divergence approximation: log(π/π_ref) = log_π - log_π_ref.
                # This is an unbiased per-sequence KL estimate.
                # Positive KL → policy moved away from ref; KL penalty pulls it back.
                kl = policy_logp - ref_logp

                # GRPO loss for this candidate:
                #   -advantage * log_π   → policy gradient (maximize reward)
                #   + beta * KL          → stay close to SFT reference
                # Dividing by total candidates normalizes the gradient scale.
                loss = (-advantage * policy_logp + args.beta * kl) / (args.n_questions * args.n_candidates)
                loss.backward()
                total_loss += loss.item()
                n_valid    += 1

            # Gradient clipping — RL objectives can produce large gradient norms
            # if advantage estimates have high variance. Clipping to 1.0 is standard.
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            progress.update(task, advance=1)

            if step % 10 == 0:
                reward_color = "green" if mean_reward > 0.1 else "yellow" if mean_reward > 0.01 else "red"
                progress.console.print(
                    f"  [dim]step[/dim] [bold]{step:4d}[/bold]  │"
                    f"  [{reward_color}]reward {mean_reward:.3f}[/{reward_color}]  │"
                    f"  [dim]loss[/dim] {total_loss:.4f}  │"
                    f"  [dim]gnorm[/dim] {grad_norm:.3f}  │"
                    f"  [dim]valid_candidates[/dim] {n_valid}"
                )
                if WANDB:
                    wandb.log({
                        "grpo/mean_reward": mean_reward,
                        "grpo/loss": total_loss,
                        "grpo/grad_norm": float(grad_norm),
                        "grpo/n_valid_candidates": n_valid,
                    }, step=step)

            if step > 0 and step % args.eval_interval == 0:
                # Show a sample generation to see qualitative progress
                policy.eval()
                sample_q  = "Q: James has 3 apples. He buys 2 bags of 5 apples each. He gives 4 away. How many does he have?\nA:"
                sample_ids = _encode(enc, sample_q)
                tok_t      = torch.tensor([sample_ids], device=device)
                sample_out = ""
                with torch.no_grad():
                    for tid in policy.stream_generate(tok_t, max_new_tokens=100,
                                                       temperature=0.3, top_k=10):
                        try:
                            sample_out += enc.decode([tid])
                        except Exception:
                            pass

                tag = "[green]####[/green]" if "####" in sample_out else "[red]no ####[/red]"
                recent_avg = sum(reward_history[-20:]) / max(1, len(reward_history[-20:]))
                progress.console.print(
                    f"[dim]sample @ step {step}:[/dim] {sample_out.strip() or '(empty)'} "
                    f"[{tag} | recent_avg_reward={recent_avg:.3f}]"
                )
                policy.train()

                # Save checkpoint
                raw = policy._orig_mod if hasattr(policy, "_orig_mod") else policy
                torch.save({
                    "model": raw.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config,
                    "step": step,
                    "meta": {
                        "grpo_step": step,
                        "mean_reward": mean_reward,
                        "recent_avg_reward": recent_avg,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    },
                }, args.output)
                progress.console.print(f"  [dim]saved → {args.output}[/dim]")

            step += 1

    if WANDB:
        wandb.finish()

    final_avg = sum(reward_history[-50:]) / max(1, len(reward_history[-50:]))
    console.print(
        f"\nGRPO complete. {args.output} saved. "
        f"Final avg reward (last 50 steps): {final_avg:.3f}"
    )


if __name__ == "__main__":
    main()
