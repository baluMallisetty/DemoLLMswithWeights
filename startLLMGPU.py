import os, io, csv, re, time
import gradio as gr

# ---------- CONFIG ----------
MODEL_PATH = r"C:\Users\balum\OneDrive\Documents\AI\LLMs\models\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"  # change to your .gguf
CTX_TOKENS = 2048
USE_GPU = True        # set False to force CPU
DET_SEED = 42         # set to None for non-deterministic

# ---------- LLAMA INIT ----------
from llama_cpp import Llama

def _init_llm():
    kwargs = dict(
        model_path=MODEL_PATH,
        n_ctx=CTX_TOKENS,
        verbose=False,
        chat_format="llama-2",         # TinyLlama behaves best with llama-like chat
    )
    if DET_SEED is not None:
        kwargs["seed"] = DET_SEED

    if USE_GPU:
        try:
            # Try full offload; if OOM, user can reduce this number.
            kwargs.update(dict(n_gpu_layers=-1, n_batch=512, flash_attn=True))
            return Llama(**kwargs)
        except Exception as e:
            print("[GPU init failed -> falling back to CPU]", e)

    # CPU fallback (still fine for TinyLlama)
    kwargs.update(dict(n_gpu_layers=0, n_batch=256))
    return Llama(**kwargs)

llm = _init_llm()

# ---------- HELPERS ----------
def _extract_text(file_dict):
    """Gradio File with type='binary' -> dict{name,data}; return a short text preview."""
    if not file_dict:
        return ""
    name = os.path.basename(file_dict.get("name", "attachment"))
    data = file_dict.get("data", b"")
    ext = os.path.splitext(name)[1].lower()
    try:
        if ext in (".txt", ".md", ".log", ".json"):
            return data.decode("utf-8", "ignore")[:40000]
        if ext == ".csv":
            rows = []
            rdr = csv.reader(io.StringIO(data.decode("utf-8", "ignore")))
            for i, row in enumerate(rdr):
                rows.append(" • " + ", ".join(row))
                if i >= 50: break
            return "CSV preview:\n" + "\n".join(rows)
        return f"[Unsupported file type: {ext or 'unknown'}]"
    except Exception as e:
        return f"[Error reading {name}: {e}]"

def _maybe_math(msg):
    if not isinstance(msg, str): return None
    if not re.fullmatch(r"\s*[\d\.\s\+\-\*\/\(\)]+?\s*", msg or ""): return None
    try:
        return str(eval(msg, {"__builtins__": {}}, {}))
    except Exception:
        return None

def _chat(messages, max_tokens=200, temperature=0.1, stop=None):
    """Stable wrapper around llama-cpp create_chat_completion."""
    stop = stop or ["<|user|>", "<|system|>", "<|assistant|>", "</s>"]
    resp = llm.create_chat_completion(
        messages=messages,
        temperature=temperature,
        top_p=0.9,
        repeat_penalty=1.08,
        max_tokens=max_tokens,
        stop=stop,
    )
    return resp["choices"][0]["message"]["content"].strip()

def _summarize(text):
    text = (text or "").strip()
    if not text: return ""
    sys = ("You are a careful summarizer. Write 3–6 bullets in your own words. "
           "No prefaces or conclusions. No quotes.")
    usr = f"Summarize the following:\n---\n{text[:60000]}\n---\n"
    s = _chat(
        [
            {"role":"system","content":sys},
            {"role":"user","content":usr}
        ],
        max_tokens=160,
        temperature=0.15,
    )
    # Normalize to bullets
    lines = re.split(r"\s*[-•]\s*", s)
    bullets = ["- " + ln.strip() for ln in lines if ln.strip()]
    return "\n".join(bullets[:6])[:900]

def _cleanup(s):
    # Strip common boilerplate
    s = re.sub(r"^(sure[,!]?|here'?s .*?:|thank you.*?\. )", "", s, flags=re.I|re.S).strip()
    # Cut on leaked role markers
    for t in ("<|user|>", "<|system|>", "<|assistant|>", "</s>"):
        i = s.find(t)
        if i != -1:
            s = s[:i].rstrip()
            break
    return s

# ---------- CHAT LOGIC ----------
def reply(user_msg, history, upload):
    history = history or []

    # quick math
    m = _maybe_math(user_msg)
    if m is not None:
        return history + [(user_msg, m)]

    # with/without attachment path
    attach_text = _extract_text(upload) if upload else ""
    messages = []

    if attach_text:
        summary = _summarize(attach_text)
        sys = ("You are a concise assistant. Use the ATTACHMENT SUMMARY as factual context. "
               "Do not copy it. If the summary lacks the answer, say so. Start directly. "
               "≤ 60 words unless asked otherwise.")
        messages.append({"role":"system","content":sys})
        # add a little recent history for context (last 3 turns)
        for u,a in history[-3:]:
            messages.append({"role":"user","content":u})
            messages.append({"role":"assistant","content":a})
        messages.append({"role":"system","content":"ATTACHMENT SUMMARY:\n" + summary})
        messages.append({"role":"user","content":user_msg})
    else:
        sys = ("You are a concise assistant. Start directly with the answer. "
               "No prefaces, no disclaimers. ≤ 60 words unless asked otherwise.")
        messages.append({"role":"system","content":sys})
        for u,a in history[-3:]:
            messages.append({"role":"user","content":u})
            messages.append({"role":"assistant","content":a})
        messages.append({"role":"user","content":user_msg})

    out = _chat(messages, max_tokens=140, temperature=0.1)
    out = _cleanup(out)
    return history + [(user_msg, out)]

# ---------- SELF TEST ----------
def self_test():
    t0 = time.time()
    out = _chat(
        [
            {"role":"system","content":"You are a helpful assistant. Answer in ≤ 20 words."},
            {"role":"user","content":"In one short sentence, what is Kubernetes?"}
        ],
        max_tokens=40,
        temperature=0.1
    )
    return f"✅ Model responded in {time.time()-t0:.1f}s:\n{out}"

# ---------- UI ----------
with gr.Blocks(title="Local LLM Chat + Attachment (stable baseline)") as demo:
    gr.Markdown("### Local LLM Chat + Attachment — stable baseline (llama-cpp-python)")
    test_btn = gr.Button("Run self-test")
    test_out = gr.Textbox(label="Self-test result")

    chat = gr.Chatbot(height=420)
    msg  = gr.Textbox(placeholder="Ask a question…", label="Message")
    file = gr.File(label="Attach text/CSV (optional)", file_types=[".txt",".md",".log",".json",".csv"],
                   type="binary")
    send = gr.Button("Send", variant="primary")
    clear= gr.Button("Clear")

    test_btn.click(lambda: self_test(), None, test_out)
    send.click(reply, [msg, chat, file], chat).then(lambda: "", None, msg)
    msg.submit(reply, [msg, chat, file], chat).then(lambda: "", None, msg)
    clear.click(lambda: [], None, chat, queue=False)

if __name__ == "__main__":
    demo.launch()  # add share=True if you want a public link
