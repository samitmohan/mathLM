"""Gradio web demo.

    pip install gradio
    python -m mathlm.infer.demo               # localhost:7860
    python -m mathlm.infer.demo --share       # public link
"""

import argparse
from mathlm.infer.inference import load_model, generate_text, chat_response

try:
    import gradio as gr
except ImportError:
    raise ImportError("pip install gradio")


def build_demo(model, enc, config, device):

    def qa_fn(prompt: str, max_tokens: int, temperature: float, top_k: int) -> str:
        """Single-shot Q/A mode."""
        if not prompt.strip():
            return ""
        # Auto-wrap in Q: A: format if not already
        if not prompt.strip().startswith("Q:"):
            prompt = f"Q: {prompt.strip()}\nA:"
        return generate_text(model, enc, device, prompt,
                              max_new_tokens=max_tokens,
                              temperature=temperature,
                              top_k=top_k,
                              stop_at_newline=False)

    def chat_fn(message: str, history: list, max_tokens: int, temperature: float) -> str:
        """Multi-turn chat mode (uses chat checkpoint special tokens if available)."""
        if not message.strip():
            return ""
        # Gradio history is list of [user, assistant] pairs
        pairs = [(h[0], h[1]) for h in history if h[1] is not None]
        return chat_response(model, enc, device, config,
                              message, history=pairs,
                              temperature=temperature,
                              max_new_tokens=max_tokens)

    with gr.Blocks(title="mathLM", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# mathLM\n"
            "GPT trained from scratch on math data. ~40M params, 8 layers, "
            "GQA, RoPE, SwiGLU, KV cache."
        )

        with gr.Tab("Q/A"):
            with gr.Row():
                with gr.Column(scale=3):
                    qa_input  = gr.Textbox(label="Prompt",
                                           placeholder="What is the derivative of x^5?",
                                           lines=3)
                    qa_output = gr.Textbox(label="Response", lines=4, interactive=False)
                    qa_btn    = gr.Button("Generate", variant="primary")
                with gr.Column(scale=1):
                    qa_temp   = gr.Slider(0.1, 2.0, value=0.8,  label="Temperature")
                    qa_topk   = gr.Slider(1,   100, value=50,    label="Top-k", step=1)
                    qa_tokens = gr.Slider(10,  500, value=100,   label="Max tokens", step=10)
            qa_btn.click(qa_fn,
                         inputs=[qa_input, qa_tokens, qa_temp, qa_topk],
                         outputs=qa_output)
            qa_input.submit(qa_fn,
                            inputs=[qa_input, qa_tokens, qa_temp, qa_topk],
                            outputs=qa_output)

            gr.Examples(
                examples=[
                    ["What is the derivative of x^5?"],
                    ["What is the derivative of sin(x)?"],
                    ["Find d/dx of (3x+1)^4."],
                    ["Integrate x^3 with respect to x."],
                    ["What is 127 + 389?"],
                ],
                inputs=qa_input,
            )

        with gr.Tab("Chat"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot   = gr.Chatbot(height=400)
                    chat_in   = gr.Textbox(placeholder="Ask a math question...", show_label=False)
                    with gr.Row():
                        chat_btn   = gr.Button("Send", variant="primary")
                        clear_btn  = gr.Button("Clear")
                with gr.Column(scale=1):
                    chat_temp   = gr.Slider(0.1, 2.0, value=0.8,  label="Temperature")
                    chat_tokens = gr.Slider(10,  500, value=200,   label="Max tokens", step=10)

            def respond(message, history, max_tokens, temperature):
                reply = chat_fn(message, history, max_tokens, temperature)
                history.append((message, reply))
                return history, ""

            chat_btn.click(respond,
                           inputs=[chat_in, chatbot, chat_tokens, chat_temp],
                           outputs=[chatbot, chat_in])
            chat_in.submit(respond,
                           inputs=[chat_in, chatbot, chat_tokens, chat_temp],
                           outputs=[chatbot, chat_in])
            clear_btn.click(lambda: ([], ""), outputs=[chatbot, chat_in])

    return demo


def main():
    parser = argparse.ArgumentParser(description="mathLM Gradio demo")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--share",      action="store_true", help="Create public Gradio link")
    parser.add_argument("--port",       type=int, default=7860)
    args = parser.parse_args()

    model, enc, config, device = load_model(ckpt_file=args.checkpoint)
    demo = build_demo(model, enc, config, device)
    demo.launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
