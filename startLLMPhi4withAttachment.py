# startLLMPhi4withAttachment.py — GPT4All + Gradio (CSV-only attachment)
# - Adds a single File input restricted to CSVs
# - Reads & summarizes the CSV (shape, headers, sample) and appends it to the user prompt
# - Safely ignores oversized or malformed files with a user-friendly note

import os, inspect, io, csv
import gradio as gr
import pandas as pd
from gpt4all import GPT4All

# ------------------ HARDWARE / OS TUNING ------------------
_CPU = os.cpu_count()-1 or 16
os.environ.setdefault("OMP_NUM_THREADS", str(_CPU))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_CPU))
os.environ.setdefault("MKL_NUM_THREADS", str(_CPU))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(_CPU))
os.environ.setdefault("OMP_DYNAMIC", "FALSE")

# ------------------ MODEL CONFIG --------------------------
MODEL_NAME = "phi-4-Q4_K_S.gguf"   # or phi-4-Q4_K_M.gguf
MODEL_DIR  = r"C:\\Users\\balum\\OneDrive\\Documents\\AI\\LLMs\\models"
USE_GPU = False

RUNTIME_CFG = dict(
    model_path=MODEL_DIR,
    allow_download=False,
    n_ctx=8192,
    mlock=True,
    mmap=False,
)

CPU_INIT = dict(device="cpu", n_gpu_layers=0, n_batch=1536, n_threads=max(1, _CPU-1))
GPU_INIT = dict(device="gpu", n_gpu_layers=-1, n_batch=1536)

GEN_CFG = dict(max_tokens=768, temp=0.18, top_p=0.9, repeat_penalty=1.07)
STOP_MARKERS = ["### User:", "### System:", "<|end|>"]
SYSTEM_PROMPT = (
    "You are a precise, concise assistant. If unsure, say you are unsure. "
    "When converting counts to percentages, always show the math."
)

# ------------------ HELPERS -------------------------------

def _supported_kwargs():
    try:
        params = inspect.signature(GPT4All.__init__).parameters
    except Exception:
        return None
    s = set(params); s.discard("self"); return s

_SUPPORTED = _supported_kwargs()

def _f(kwargs):
    if not _SUPPORTED:
        return {k:v for k,v in kwargs.items()}
    return {k:v for k,v in kwargs.items() if k in _SUPPORTED}

# Basic CSV checks
_MAX_BYTES = 300 * 1024 * 1024  # 8 MB cap; adjust if needed
_MAX_PREVIEW_ROWS = 12        # rows to include in preview


def _is_csv_path(path: str) -> bool:
    # Quick extension check
    if not path.lower().endswith('.csv'):
        return False
    # Light sniffing: try reading a few bytes via csv
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            sniffer = csv.Sniffer()
            sample = f.read(4096)
            f.seek(0)
            # If it can detect a dialect OR we can parse the first line as CSV, treat as CSV
            try:
                sniffer.sniff(sample)
                return True
            except Exception:
                reader = csv.reader(io.StringIO(sample))
                first = next(reader, None)
                return first is not None and len(first) > 0
    except Exception:
        return False


def _summarize_csv(path: str) -> str:
    try:
        size = os.path.getsize(path)
    except OSError:
        size = -1

    if size > _MAX_BYTES > 0:
        return f"[CSV attached but skipped: file is {size/1024/1024:.1f} MB (> {_MAX_BYTES/1024/1024:.0f} MB). Please upload a smaller file or increase _MAX_BYTES in the script.]"

    if not _is_csv_path(path):
        return "[Attachment rejected: only .csv files are allowed.]"

    try:
        # Use pandas for a robust read; fall back to csv if necessary
        df = pd.read_csv(path)
        rows, cols = df.shape
        headers = list(df.columns)
        # Build a compact preview
        head_rows = min(_MAX_PREVIEW_ROWS, len(df))
        preview_csv = df.head(head_rows).to_csv(index=False)
        return (
            "[CSV summary]\n"
            f"Path: {os.path.basename(path)}\n"
            f"Size: {size if size>=0 else 'unknown'} bytes\n"
            f"Shape: {rows} rows × {cols} columns\n"
            f"Headers: {headers}\n"
            f"Preview (first {head_rows} rows):\n{preview_csv}"
        )
    except Exception as e:
        # Fallback: minimal line-count read with csv module
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.reader(f)
                preview = []
                for i, row in enumerate(reader):
                    preview.append(row)
                    if i+1 >= _MAX_PREVIEW_ROWS:
                        break
                hdr = preview[0] if preview else []
                pr_txt = "\n".join([",".join(r) for r in preview])
                return (
                    "[CSV summary — fallback]\n"
                    f"Path: {os.path.basename(path)}\n"
                    f"Size: {size if size>=0 else 'unknown'} bytes\n"
                    f"Headers: {hdr}\n"
                    f"Preview (first {len(preview)} rows):\n{pr_txt}"
                )
        except Exception as ee:
            return f"[Attachment error: failed to read CSV — {ee}]"


# ------------------ MODEL INIT ---------------------------

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


def _should_stop(t: str) -> bool:
    return any(t.endswith(m) or m in t[-64:] for m in STOP_MARKERS)


def _strip_markers(t: str) -> str:
    for m in STOP_MARKERS:
        if t.endswith(m):
            return t[:-len(m)].rstrip()
    return t

# ------------------ STREAMING CHAT FN ---------------------
# ChatInterface passes: (message, history, *additional_inputs)
# Our single additional input is the CSV filepath (or None)

def chat_fn(message, history, csv_path, system_prompt=SYSTEM_PROMPT):
    # Build messages with optional CSV annotation
    attachment_note = ""
    if csv_path:
        attachment_note = "\n\n" + _summarize_csv(csv_path)

    msgs = [{"role":"system", "content":system_prompt}]
    msgs.extend(history or [])
    msgs.append({"role":"user", "content":message + attachment_note})

    prompt = messages_to_prompt(msgs)

    out = ""
    for tok in model.generate(prompt, streaming=True, **GEN_CFG):
        out += tok
        if _should_stop(out):
            out = _strip_markers(out)
            yield out
            return
        yield out


# ------------------ UI ------------------------------------
with gr.Blocks(title="Phi-4 — GPT4All (CSV Attach)") as ui:
    gr.Markdown("# Phi-4 — GPT4All (CSV-only attachment)\nAttach a **.csv** file to enrich your prompt. The file is summarized and included in the prompt.")

    csv_input = gr.File(
        label="Attach CSV",
        file_types=[".csv"],
        file_count="single",
        type="filepath",  # function receives a path string
        height=80,
    )

    chat = gr.ChatInterface(
        fn=chat_fn,
        type="messages",
        title="Phi-4 — GPT4All (CPU-max, big RAM)",
        description="Max threads, big batch, mlock, large context. CSVs only.",
        submit_btn="Send",
        stop_btn="Stop",
        additional_inputs=[csv_input],
    )

if __name__ == "__main__":
    ui.launch(server_name="0.0.0.0", server_port=7860, share=False)
