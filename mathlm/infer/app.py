"""HuggingFace Space entry point. Downloads the checkpoint from HF Hub if absent.

    python -m mathlm.infer.app                  # uses local checkpoint if available
    python -m mathlm.infer.app --share          # public Gradio link
"""

import os
import argparse

MODEL_REPO = "samitmohan/mathlm"
CKPT_NAME = "checkpoint_mathlm_grpo.pt"
CKPT_FALLBACK = "checkpoint_mathlm.pt"

def get_checkpoint() -> str:
    for candidate in (CKPT_NAME, CKPT_FALLBACK, "checkpoint.pt"):
        if os.path.exists(candidate):
            return candidate
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=MODEL_REPO, filename=CKPT_NAME)
    except Exception as e:
        raise FileNotFoundError(
            f"No checkpoint found locally or on HF Hub ({MODEL_REPO}/{CKPT_NAME}): {e}. "
            "Run scripts/upload_hf.py to publish a checkpoint."
        )


import gradio as gr
from mathlm.infer.inference import load_model, generate_text, chat_response

CHECKPOINT = get_checkpoint()
model, enc, config, device = load_model(ckpt_file=CHECKPOINT)

MATH_SYSTEM = (
    "You are MathLM, a math reasoning assistant built from scratch "
    "on a 40M parameter GPT trained on calculus, ML/DL math, and word problems."
)

DESCRIPTION = """
# MathLM
A 40M parameter math reasoning LLM **built entirely from scratch** —
custom BPE tokenizer → pretraining → unified SFT → GRPO reasoning.

**Covers:** Calculus (derivatives, integrals) · ML/DL math (backprop, attention, gradients) · Word problems

**Use Q:/A: format for best results** — that's what the model was trained on.
"""

EXAMPLES = [
    ["Q: What is the derivative of 3x^5 - 2x^3 + 7?\nA:"],
    ["Q: Find d/dx of sin(x^2) using the chain rule.\nA:"],
    ["Q: Compute the indefinite integral of 4x^3.\nA:"],
    ["Q: What is the gradient of the MSE loss L=(1/n)||Xw-y||^2 with respect to w?\nA:"],
    ["Q: In backpropagation, what is the gradient of cross-entropy loss with respect to softmax input logits?\nA:"],
    ["Q: What is scaled dot-product attention? Write the formula.\nA:"],
    ["Q: Tom has 15 apples. He buys 3 bags of 6 apples each. He gives 7 to his friend. How many does he have?\nA:"],
    ["Q: Sarah earns $12/hour and works 8 hours per day, 5 days per week. How much does she earn in 4 weeks?\nA:"],
    ["Q: What is the gradient of f(x) = x^T A x with respect to x, for symmetric A?\nA:"],
]


def qa_fn(prompt: str, max_tokens: int, temperature: float, top_k: int) -> str:
    if not prompt.strip():
        return ""
    # Auto-wrap in Q:/A: if the user forgot
    if not prompt.strip().startswith("Q:"):
        prompt = f"Q: {prompt.strip()}\nA:"
    return generate_text(
        model, enc, device, prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        stop_at_newline=False,
    )


with gr.Blocks(title="mathLM", theme=gr.themes.Soft()) as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Tab("Math Q/A"):
        with gr.Row():
            with gr.Column(scale=3):
                qa_input  = gr.Textbox(
                    label="Prompt (Q: ... \\nA: format)",
                    placeholder="Q: What is the derivative of x^5?\nA:",
                    lines=4,
                )
                qa_output = gr.Textbox(label="MathLM response", lines=6, interactive=False)
                qa_btn    = gr.Button("Generate", variant="primary")
            with gr.Column(scale=1):
                qa_temp   = gr.Slider(0.1, 1.5, value=0.3,  label="Temperature",
                                      info="Lower = more deterministic")
                qa_topk   = gr.Slider(1,   100, value=10,   label="Top-k", step=1)
                qa_tokens = gr.Slider(10,  400, value=150,  label="Max tokens", step=10)

        qa_btn.click(qa_fn, inputs=[qa_input, qa_tokens, qa_temp, qa_topk], outputs=qa_output)
        qa_input.submit(qa_fn, inputs=[qa_input, qa_tokens, qa_temp, qa_topk], outputs=qa_output)

        gr.Examples(examples=EXAMPLES, inputs=qa_input, label="Example prompts")

    with gr.Tab("About"):
        gr.Markdown("""
## Architecture
| Component | Choice |
|---|---|
| Parameters | **40.4M** |
| Layers / dim / heads | 8L / 512d / 8Q 4KV (GQA) |
| Tokenizer | Custom BPE (32k vocab, math-aware) |
| Attention | Grouped Query Attention + RoPE + QK-norm |
| MLP | SwiGLU activation |
| Normalization | RMSNorm (pre-norm) |
| Inference | KV cache + streaming generation |

## Training Pipeline
1. **Custom BPE tokenizer** — trained on OpenWebMath, treats `∇`, `∫`, `x^2` as single tokens
2. **Pretraining** — 100k steps on 0.62B tokens (OpenWebMath, NuminaMath, OpenR1, MATH, GSM8K, synthetic)
3. **Unified SFT** — fine-tuned on all math domains simultaneously at 10× lower LR to prevent catastrophic forgetting
4. **GRPO** — RL with binary correctness reward (no critic, no value function) to improve reasoning

## Code
[github.com/samitmohan/mathLM](https://github.com/samitmohan/mathLM)
        """)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port",  type=int, default=7860)
    args = parser.parse_args()
    demo.launch(share=args.share, server_port=args.port)
