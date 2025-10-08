import os
from gpt4all import GPT4All
import gradio as gr

# --- CONFIG ---
MODEL_NAME = "Phi-3-mini-4k-instruct-q4.gguf"   # file name of your GGUF
MODEL_DIR  = os.path.expanduser(r"C:\Users\balum\OneDrive\Documents\AI\LLMs\models")  # change to wherever your model is
USE_GPU    = True  # set False if your build/device can't use GPU

# Generation params (tweak for your laptop)
GEN_CFG = dict(
    max_tokens=512,
    temp=0.7,
    top_p=0.9,
    repeat_penalty=1.1,
)

# Runtime params (safe defaults for corp laptops)
RUNTIME_CFG = dict(
    device="gpu" if USE_GPU else "cpu",
    n_ctx=2048,                     # reduce to 1024 if RAM/VRAM is tight
    n_threads=os.cpu_count() or 4,  # CPU threads
    allow_download=False,           # avoid auto-download on corp devices
)

SYSTEM_PROMPT = (
    "You are a helpful, concise assistant. "
    "If you are unsure, say you are unsure. Keep answers short and clear."
)

def format_prompt(system_msg: str, history: list[list[str]], user_msg: str) -> str:
    """
    Simple instruction-style formatting that works well with Phi-3.
    """
    parts = [f"### System:\n{system_msg}\n"]
    for u, a in history:
        if u:
            parts.append(f"### User:\n{u}\n")
        if a:
            parts.append(f"### Assistant:\n{a}\n")
    parts.append(f"### User:\n{user_msg}\n### Assistant:\n")
    return "\n".join(parts)

# Load model (first run may take a few seconds)
model = GPT4All(MODEL_NAME, model_path=MODEL_DIR, **RUNTIME_CFG)

def chat_fn(message, history):
    """
    Gradio expects either a string or a generator for streaming.
    We stream tokens from GPT4All.
    """
    prompt = format_prompt(SYSTEM_PROMPT, history or [], message)
    out = ""
    for token in model.generate(prompt, streaming=True, **GEN_CFG):
        out += token
        yield out

ui = gr.ChatInterface(
    fn=chat_fn,
    title="Phi-3 Mini (GGUF) — GPT4All + Gradio",
    description="Local, offline chat using Microsoft Phi-3 Mini 4K Instruct (GGUF) via GPT4All.",
    submit_btn="Send",
    stop_btn="Stop",
)

if __name__ == "__main__":
    ui.launch(server_name="0.0.0.0", server_port=7860, share=False)
