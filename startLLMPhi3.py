import inspect
import os
from gpt4all import GPT4All
import gradio as gr

# --- CONFIG ---
MODEL_NAME = "Phi-3-mini-4k-instruct-q4.gguf"
MODEL_DIR  = os.path.expanduser(r"C:\Users\balum\OneDrive\Documents\AI\LLMs\models")
USE_GPU    = True

# Generation config (NO 'stop' kwarg here)
GEN_CFG = dict(
    max_tokens=512,
    temp=0.7,
    top_p=0.9,
    repeat_penalty=1.1,
)

# We’ll stop manually when model tries to start a new header
STOP_MARKERS = ["### User:", "### System:", "<|end|>"]

# Runtime options (keep simple; older GPT4All wheels ignore unknown kwargs)
RUNTIME_CFG = dict(
    model_path=MODEL_DIR,
    allow_download=False,
)


def _supported_kwargs():
    """Return the set of supported GPT4All.__init__ keyword parameters."""
    try:
        params = inspect.signature(GPT4All.__init__).parameters
    except (TypeError, ValueError):  # pragma: no cover - C level signature
        return None
    allowed = set(params)
    allowed.discard("self")
    return allowed


_SUPPORTED_KWARGS = _supported_kwargs()


def _filter_kwargs(kwargs):
    if not _SUPPORTED_KWARGS:
        return kwargs
    return {k: v for k, v in kwargs.items() if k in _SUPPORTED_KWARGS}


def _init_model():
    base = dict(RUNTIME_CFG)
    if USE_GPU:
        gpu_cfg = dict(device="gpu", n_gpu_layers=-1, n_batch=512)
        try:
            cfg = _filter_kwargs({**base, **gpu_cfg})
            return GPT4All(MODEL_NAME, **cfg)
        except Exception as exc:
            print("[GPU init failed -> falling back to CPU]", exc)

    cpu_cfg = dict(device="cpu", n_gpu_layers=0, n_batch=256,
                   n_threads=max(2, (os.cpu_count() or 4) // 2))
    cfg = _filter_kwargs({**base, **cpu_cfg})
    return GPT4All(MODEL_NAME, **cfg)

SYSTEM_PROMPT = (
    "You are a helpful, concise assistant. "
    "If unsure, say you are unsure. Keep answers short and clear."
)

def messages_to_prompt(messages):
    parts = []
    sys = next((m["content"] for m in messages if m["role"] == "system"), SYSTEM_PROMPT)
    parts.append(f"### System:\n{sys}\n")
    for m in messages:
        role, content = m["role"], m["content"]
        if role == "user":
            parts.append(f"### User:\n{content}\n")
        elif role == "assistant":
            parts.append(f"### Assistant:\n{content}\n")
    parts.append("### Assistant:\n")
    return "\n".join(parts)

# Load model once (prefer GPU but fall back gracefully)
model = _init_model()

def _should_stop(text: str) -> bool:
    return any(text.endswith(m) or m in text[-32:] for m in STOP_MARKERS)

def _strip_trailing_markers(text: str) -> str:
    for m in STOP_MARKERS:
        if text.endswith(m):
            return text[: -len(m)].rstrip()
    return text

def chat_fn(message, history, system_prompt=SYSTEM_PROMPT):
    msgs = [{"role": "system", "content": system_prompt}]
    msgs.extend(history or [])
    msgs.append({"role": "user", "content": message})

    prompt = messages_to_prompt(msgs)

    out = ""
    for token in model.generate(prompt, streaming=True, **GEN_CFG):
        out += token
        # manual stop when a new header starts
        if _should_stop(out):
            out = _strip_trailing_markers(out)
            yield out
            return
        yield out

ui = gr.ChatInterface(
    fn=chat_fn,
    title="Phi-3 Mini (GGUF) — GPT4All + Gradio",
    description="Local, offline chat using Microsoft Phi-3 Mini 4K Instruct (GGUF) via GPT4All.",
    submit_btn="Send",
    stop_btn="Stop",
    type="messages",  # role-based history prevents repeat answers
)

if __name__ == "__main__":
    ui.launch(server_name="0.0.0.0", server_port=7860, share=False)
