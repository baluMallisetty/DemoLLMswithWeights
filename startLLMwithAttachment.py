
# startLLMwithAttachment.py — prevents mid‑answer cutoffs & fixes code blocks
import os, io, re, csv
from pathlib import Path
import gradio as gr

os.environ.setdefault("GPT4ALL_LOG_LEVEL", "error")

try:
    from gpt4all import GPT4All
except Exception as e:
    raise RuntimeError("gpt4all is required: pip install gpt4all") from e

MODEL_NAME = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MODEL_PATH = Path(__file__).resolve().parent / "models" / MODEL_NAME
if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}.")

llm = GPT4All(str(MODEL_PATH), allow_download=False, n_threads=max(2, (os.cpu_count() or 4)//2))

# ----------------- Relevance guard -----------------
GREETINGS = re.compile(r"^\s*(hi|hello|hey|hola|yo|sup|good\s*(morning|evening|afternoon))\s*[!.?]*\s*$", re.I)
SMALLTALK = re.compile(r"^\s*(how are you|what's up|wyd)\s*[?.!]*\s*$", re.I)

def router_shortcuts(msg: str):
    if not msg or not isinstance(msg, str) or len(msg.strip()) == 0:
        return "hey—what would you like to do?"
    if GREETINGS.match(msg): return "hi! what can I help you with?"
    if SMALLTALK.match(msg): return "doing great. what would you like me to help with?"
    if len(msg.strip()) < 5 and "?" not in msg: return "could you share a bit more detail?"
    return None

# ---------- helpers ----------
def _extract_text(file):
    if not file: return ""
    if isinstance(file, dict): data, name = file.get("data", b""), os.path.basename(file.get("name","attachment"))
    else:
        name = os.path.basename(getattr(file,"name","attachment"))
        try: data = file.read()
        except Exception: data = b""
    ext = os.path.splitext(name)[1].lower()
    try:
        if ext in (".txt",".md",".log",".json"): return data.decode("utf-8","ignore")[:40000]
        if ext == ".csv":
            text = data.decode("utf-8","ignore")
            out, rdr = [], csv.reader(io.StringIO(text))
            for i,row in enumerate(rdr):
                out.append(" • "+", ".join(row))
                if i>=40: break
            return "CSV preview (first rows):\n"+"\n".join(out)
    except Exception as e: return f"[Error reading {name}: {e}]"
    return "[Unsupported attachment type]"

def _maybe_calc_math(msg):
    if not isinstance(msg, str): return None
    if not re.fullmatch(r"\s*[\d\.\s\+\-\*\/\(\)]+?\s*", msg or ""): return None
    try: return str(eval(msg, {"__builtins__": {}}, {}))
    except Exception: return None

def _to_chatml(messages):
    parts = []
    for m in messages:
        role = m.get("role","user")
        content = (m.get("content") or "").strip()
        role = "system" if role=="system" else "user" if role=="user" else "assistant"
        parts.append(f"<|{role}|>\n{content}\n")
    parts.append("<|assistant|>\n")
    return "".join(parts)

# trimmed stop tokens to avoid accidental early stops in code
STOP_TOKENS = ["<|user|>", "<|system|>", "<|assistant|>", "</s>"]

def _generate_fallback(prompt, max_tokens, temperature):
    for kwargs in (
        dict(max_tokens=max_tokens, temp=temperature, top_p=0.95, repeat_penalty=1.08, stop=STOP_TOKENS, streaming=False),
        dict(n_predict=max_tokens, temp=temperature, top_p=0.95, repeat_penalty=1.08, stop=STOP_TOKENS, streaming=False),
        dict(n_predict=max_tokens),
        {},
    ):
        try:
            return llm.generate(prompt, **kwargs)
        except TypeError:
            continue
    return llm.generate(prompt)

def _chat(messages, *, max_tokens=256, temperature=0.0):
    if hasattr(llm, "chat_completion"):
        response = llm.chat_completion(messages=messages, max_tokens=max_tokens, temperature=temperature, top_p=0.95, stop=STOP_TOKENS)
        return response["choices"][0]["message"]["content"].strip()
    prompt = _to_chatml(messages)
    text = _generate_fallback(prompt, max_tokens, temperature)
    return (text or "").strip()

def _needs_more(text: str) -> bool:
    if not text: return False
    # Unfinished code fence or sentence
    open_fences = text.count("```")
    if open_fences % 2 == 1: return True
    # Unbalanced braces/brackets common in code
    if text.count("{") > text.count("}") or text.count("(") > text.count(")") or text.count("[") > text.count("]"):
        return True
    # Ends abruptly
    if not re.search(r"[\.\?\!]\s*$|```$", text): return True
    return False

def _continue(messages, partial, extra_tokens=512):
    # Ask the model to continue exactly
    msgs = list(messages) + [
        {"role":"assistant","content": partial},
        {"role":"user","content":"Continue exactly from where you stopped. Do not repeat previous lines."}
    ]
    more = _chat(msgs, max_tokens=extra_tokens, temperature=0.0)
    return more

def _merge_continuation(base: str, addition: str) -> str:
    """Return ``base`` with ``addition`` stitched on, avoiding duplicated spans."""
    if not addition:
        return base
    addition = addition.lstrip()
    if not addition:
        return base
    # If the model repeated the entire answer, drop it.
    if addition.strip().startswith(base.strip()):
        return base
    # Find the largest overlap between the end of ``base`` and the beginning of ``addition``.
    max_overlap = min(len(base), len(addition), 600)
    for size in range(max_overlap, 0, -1):
        if base[-size:] == addition[:size]:
            return base + addition[size:]
    # If the continuation is already contained, avoid re-appending.
    if addition in base:
        return base
    return base + ("\n" if base and not base.endswith("\n") else "") + addition

def _ensure_closed_fences(text: str) -> str:
    if text.count("```") % 2 == 1:
        text += "\n```"
    return text

def _summarize_attachment(raw: str) -> str:
    if not raw.strip(): return ""
    messages = [
        {"role":"system","content":"Summarize crisply in 3–6 bullets. No prefaces or conclusions. Only facts from text."},
        {"role":"user","content":f"Summarize:\n---\n{raw[:60000]}\n---\n"},
    ]
    summary = _chat(messages, max_tokens=180, temperature=0.0)
    bullets = re.split(r"\n\s*[-•]\s*", "- "+summary)
    bullets = ["- "+b.strip() for b in bullets if b.strip()]
    return "\n".join(bullets[:6])[:800]

SYS_BASE = (
    "You are a precise assistant. Respond in one short paragraph (≤100 words) unless code is requested. "
    "Stay strictly on the user's topic. If ambiguous, ask one clarifying question. If you lack facts, say 'I don't know'."
)

def _format_prompt_no_attach(history, user_msg):
    messages = [{"role":"system","content":SYS_BASE}]
    for u,a in (history or [])[-4:]:
        messages += [{"role":"user","content":u},{"role":"assistant","content":a}]
    messages.append({"role":"user","content":user_msg})
    return messages

def _format_prompt_with_attach(history, user_msg, summary):
    sys = SYS_BASE + " Use the ATTACHMENT SUMMARY as context; do not invent details."
    messages = [{"role":"system","content":sys}]
    for u,a in (history or [])[-3:]:
        messages += [{"role":"user","content":u},{"role":"assistant","content":a}]
    messages += [
        {"role":"system","content":"ATTACHMENT SUMMARY:\n"+summary},
        {"role":"user","content":user_msg},
    ]
    return messages

def _cleanup(text: str) -> str:
    for t in ("<|user|>","<|system|>","<|assistant|>","</s>"):
        i = text.find(t)
        if i != -1: text = text[:i].rstrip(); break
    return _ensure_closed_fences(text.strip())

def _max_tokens_for(message: str) -> int:
    # If likely code or long answer, allow more tokens.
    if re.search(r"```|code|function|class|example|java|python|sql|regex|json|xml", message, re.I):
        return 768
    return 256

def chat_fn(message, history, file):
    fast = router_shortcuts(message)
    if fast is not None: return (history or []) + [(message, fast)]
    calc = _maybe_calc_math(message)
    if calc is not None: return (history or []) + [(message, calc)]
    raw = _extract_text(file) if file else ""
    messages = _format_prompt_with_attach(history, message, _summarize_attachment(raw)) if raw else _format_prompt_no_attach(history, message)
    out = _chat(messages, max_tokens=_max_tokens_for(message), temperature=0.0)
    # If truncated, ask the model to continue once (or twice) and stitch.
    tries = 0
    while _needs_more(out) and tries < 2:
        more = _continue(messages, out, extra_tokens=512)
        if not more: break
        out = _merge_continuation(out, more)
        tries += 1
    reply = _cleanup(out)
    return (history or []) + [(message, reply)]

# -------- UI --------
with gr.Blocks(title="Local Chat + Attachment (TinyLlama)") as demo:
    gr.Markdown("### Local Chat + Attachment (TinyLlama) — concise answers, no parroting")
    chat = gr.Chatbot(height=560, type="tuples")  # taller to view long code
    msg  = gr.Textbox(placeholder="Ask a question…", label="Textbox")
    file = gr.File(label="Attach a text/CSV file (optional)", file_types=[".txt",".md",".log",".json",".csv"], type="binary")
    send = gr.Button("Send", variant="primary")
    clear= gr.Button("Clear")

    send.click(chat_fn, [msg, chat, file], chat).then(lambda: "", None, msg)
    msg.submit(chat_fn, [msg, chat, file], chat).then(lambda: "", None, msg)
    clear.click(lambda: [], None, chat, queue=False)

if __name__ == "__main__":
    demo.launch()
