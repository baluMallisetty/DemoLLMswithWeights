# startLLMwithAttachment_fixed.py
import os, io, re, csv
from pathlib import Path
import gradio as gr
from gpt4all import GPT4All

MODEL_NAME = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MODEL_PATH = Path(__file__).resolve().parent / "models" / MODEL_NAME

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. "
        "Place the model in the 'models' directory next to this script."
    )

llm = GPT4All(
    str(MODEL_PATH),
    allow_download=False,
    n_threads=max(2, (os.cpu_count() or 4)//2)
)

def _extract_text(file):
    if not file:
        return ""
    # gr.File(type="binary") gives dict{name,data}; fallback to file-like
    if isinstance(file, dict):
        data = file.get("data", b"")
        name = os.path.basename(file.get("name", "attachment"))
    else:
        name = os.path.basename(getattr(file, "name", "attachment"))
        try:
            data = file.read()
        except Exception:
            data = b""
    ext = os.path.splitext(name)[1].lower()
    try:
        if ext in (".txt", ".md", ".log", ".json"):
            return data.decode("utf-8", "ignore")[:40000]
        if ext == ".csv":
            text = data.decode("utf-8", "ignore")
            out, rdr = [], csv.reader(io.StringIO(text))
            for i, row in enumerate(rdr):
                out.append(" • " + ", ".join(row))
                if i >= 40: break
            return "CSV preview (first rows):\n" + "\n".join(out)
    except Exception as e:
        return f"[Error reading {name}: {e}]"
    return "[Unsupported attachment type]"

def _maybe_calc_math(msg):
    if not isinstance(msg, str): return None
    if not re.fullmatch(r"\s*[\d\.\s\+\-\*\/\(\)]+?\s*", msg or ""): return None
    try:
        return str(eval(msg, {"__builtins__": {}}, {}))
    except Exception:
        return None

def _chat(messages, *, max_tokens=200, temperature=0.1):
    response = llm.chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9,
        repeat_penalty=1.08,
    )
    return response["choices"][0]["message"]["content"].strip()

def _summarize_attachment(raw: str) -> str:
    if not raw.strip():
        return ""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful summariser. Write 3–6 short bullets in your "
                "own words. No prefaces or conclusions."
            ),
        },
        {
            "role": "user",
            "content": f"Summarise the following text.\n---\n{raw[:60000]}\n---\n",
        },
    ]
    summary = _chat(messages, max_tokens=160, temperature=0.15)
    bullets = re.split(r"\n\s*[-•]\s*", "- " + summary)
    bullets = ["- " + b.strip() for b in bullets if b.strip()]
    return "\n".join(bullets[:6])[:900]

def _format_prompt_no_attach(history, user_msg):
    sys = (
        "You are a concise assistant. Start directly with the answer. "
        "No prefaces, no disclaimers. Keep answers ≤ 60 words unless asked otherwise."
    )
    messages = [{"role": "system", "content": sys}]
    for u, a in (history or [])[-6:]:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": user_msg})
    return messages

def _format_prompt_with_attach(history, user_msg, summary):
    sys = (
        "You are a concise assistant. Use the ATTACHMENT SUMMARY as factual context. "
        "Do not copy it. If the summary lacks the answer, say so. Start directly. "
        "≤ 60 words unless asked otherwise."
    )
    messages = [{"role": "system", "content": sys}]
    for u, a in (history or [])[-4:]:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "system", "content": "ATTACHMENT SUMMARY:\n" + summary})
    messages.append({"role": "user", "content": user_msg})
    return messages

def _cleanup(text: str) -> str:
    # strip common boilerplate
    text = re.sub(r"^(thank you .*?|sure[,!]?.*?|here'?s .*?:)\s*", "", text, flags=re.I|re.S)
    # cut if model leaks role markers
    for t in ("<|user|>", "<|system|>", "<|assistant|>", "</s>"):
        i = text.find(t)
        if i != -1:
            text = text[:i].rstrip()
            break
    # enforce one-paragraph shortness
    return text.strip()

def chat_fn(message, history, file):
    calc = _maybe_calc_math(message)
    if calc is not None:
        return (history or []) + [(message, calc)]

    raw = _extract_text(file) if file else ""
    if raw:
        summary = _summarize_attachment(raw)
        messages = _format_prompt_with_attach(history, message, summary)
    else:
        messages = _format_prompt_no_attach(history, message)

    out = _chat(messages, max_tokens=140, temperature=0.1)
    reply = _cleanup(out)
    return (history or []) + [(message, reply)]

# -------- UI --------
with gr.Blocks(title="Local Chat + Attachment (TinyLlama)") as demo:
    gr.Markdown("### Local Chat + Attachment (TinyLlama) — concise answers, no parroting")
    chat = gr.Chatbot(height=420)
    msg  = gr.Textbox(placeholder="Ask a question…", label="Textbox")
    file = gr.File(label="Attach a text/CSV file (optional)",
                   file_types=[".txt", ".md", ".log", ".json", ".csv"],
                   type="binary")
    send = gr.Button("Send", variant="primary")
    clear= gr.Button("Clear")

    send.click(chat_fn, [msg, chat, file], chat).then(lambda: "", None, msg)
    msg.submit(chat_fn, [msg, chat, file], chat).then(lambda: "", None, msg)
    clear.click(lambda: [], None, chat, queue=False)

if __name__ == "__main__":
    demo.launch()  # add share=True if you want a public link
