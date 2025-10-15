# startLLMPhi4.py — GPT4All + Gradio (CPU maxed, big RAM, fast batch)

import os, inspect
import gradio as gr
from gpt4all import GPT4All

# ------------------ HARDWARE / OS TUNING ------------------
# Force OpenMP / BLAS to use (almost) all cores.
_CPU = os.cpu_count() or 16
os.environ.setdefault("OMP_NUM_THREADS", str(_CPU))         # OpenMP
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_CPU))    # OpenBLAS
os.environ.setdefault("MKL_NUM_THREADS", str(_CPU))         # MKL (if linked)
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(_CPU))
os.environ.setdefault("OMP_DYNAMIC", "FALSE")

# ------------------ MODEL CONFIG --------------------------
MODEL_NAME = "phi-4-Q4_K_S.gguf"   # or phi-4-Q4_K_M.gguf
MODEL_DIR  = r"C:\Users\balum\OneDrive\Documents\AI\LLMs\models"

# IMPORTANT: keep GPU OFF (your build’s CUDA DLLs misbehave)
USE_GPU = False

# Larger context + keep weights locked in RAM
RUNTIME_CFG = dict(
    model_path=MODEL_DIR,
    allow_download=False,
    n_ctx=8192,      # 6144 works well on 32GB; try 8192 if your wheel supports it
    mlock=True,      # pin in RAM; fewer page faults
    mmap=False,      # load fully in RAM instead of memory-mapped file
)

# Max CPU pressure: big batch + all threads
CPU_INIT = dict(
    device="cpu",
    n_gpu_layers=0,
    n_batch=1536,             # 1024→1536 is a sweet spot; drop to 1024 if stutters
    n_threads=max(1, _CPU-1), # use almost all cores
)

# If you later fix CUDA and WANT to try GPU, flip USE_GPU=True
GPU_INIT = dict(device="gpu", n_gpu_layers=-1, n_batch=1536)

# Generation tuning (deterministic, concise)
GEN_CFG = dict(max_tokens=768, temp=0.18, top_p=0.9, repeat_penalty=1.07)

STOP_MARKERS = ["### User:", "### System:", "<|end|>"]

SYSTEM_PROMPT = (
    "You are a precise, concise assistant. If unsure, say you are unsure. "
    "When converting counts to percentages, always show the math."
)

# ------------------ HELPER: filter kwargs the wheel supports ------------------
def _supported_kwargs():
    try:
        params = inspect.signature(GPT4All.__init__).parameters
    except Exception:
        return None
    s = set(params); s.discard("self"); return s

_SUPPORTED = _supported_kwargs()
def _f(kwargs): return {k:v for k,v in kwargs.items()} if not _SUPPORTED else {k:v for k,v in kwargs.items() if k in _SUPPORTED}

# ------------------ INIT MODEL (CPU first, no GPU overhead) -------------------
def _init_model():
    base = dict(RUNTIME_CFG)
    if USE_GPU:
        try:
            cfg = _f({**base, **GPU_INIT})
            print("[INFO] Trying GPU:", cfg)
            return GPT4All(MODEL_NAME, **cfg)
        except Exception as e:
            print("[WARN] GPU init failed; falling back to CPU:", e)
    cfg = _f({**base, **CPU_INIT})
    print("[INFO] CPU init:", cfg)
    return GPT4All(MODEL_NAME, **cfg)

model = _init_model()

# ------------------ PROMPT FORMAT -------------------------
def messages_to_prompt(messages):
    parts = []
    sys = next((m["content"] for m in messages if m["role"]=="system"), SYSTEM_PROMPT)
    parts.append(f"### System:\n{sys}\n")
    for m in messages:
        if m["role"]=="user":
            parts.append(f"### User:\n{m['content']}\n")
        elif m["role"]=="assistant":
            parts.append(f"### Assistant:\n{m['content']}\n")
    parts.append("### Assistant:\n")
    return "\n".join(parts)

# ------------------ STREAMING CHAT FN ---------------------
def _should_stop(t: str) -> bool:
    return any(t.endswith(m) or m in t[-64:] for m in STOP_MARKERS)

def _strip_markers(t: str) -> str:
    for m in STOP_MARKERS:
        if t.endswith(m): return t[:-len(m)].rstrip()
    return t

def chat_fn(message, history, system_prompt=SYSTEM_PROMPT):
    msgs = [{"role":"system","content":system_prompt}]
    msgs.extend(history or [])
    msgs.append({"role":"user","content":message})
    prompt = messages_to_prompt(msgs)

    out = ""
    for tok in model.generate(prompt, streaming=True, **GEN_CFG):
        out += tok
        if _should_stop(out):
            out = _strip_markers(out)
            yield out
            return
        yield out

ui = gr.ChatInterface(
    fn=chat_fn,
    type="messages",
    title="Phi-4 — GPT4All (CPU-max, big RAM)",
    description="Max threads, big batch, mlock, and large context.",
    submit_btn="Send",
    stop_btn="Stop",
)

if __name__ == "__main__":
    ui.launch(server_name="0.0.0.0", server_port=7860, share=False)
