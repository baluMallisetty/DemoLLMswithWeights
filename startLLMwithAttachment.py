# startLLMwithAttachment_fixed.py
import os, io, re, csv
import gradio as gr
from gpt4all import GPT4All

MODEL = r"C:\Users\balum\OneDrive\Documents\AI\LLMs\models\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
llm = GPT4All(MODEL, allow_download=False, n_threads=max(2, (os.cpu_count() or 4)//2))

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

def _summarize_attachment(raw: str) -> str:
    if not raw.strip(): return ""
    prompt = (
        "<|system|>\nSummarize the text into 3–6 short bullet points. "
        "Use your own words only. No prefaces, no conclusions.\n"
        "<|user|>\n---\n" + raw[:60000] + "\n---\n"
        "<|assistant|>\n- "
    )
    with llm.chat_session():
        s = llm.generate(
            prompt, max_tokens=160, temp=0.15, top_p=0.9, top_k=40,
            repeat_penalty=1.12, repeat_last_n=256, n_batch=256
        ).strip()
    # Normalize to bullets, keep it tight
    bullets = re.split(r"\n\s*[-•]\s*", "- " + s)
    bullets = [("- " + b.strip()) for b in bullets if b.strip()]
    return "\n".join(bullets[:6])[:900]

def _format_prompt_no_attach(history, user_msg):
    sys = ("You are a concise assistant. Start directly with the answer. "
           "No prefaces, no disclaimers. Keep answers ≤ 60 words unless asked otherwise.")
    msgs = [f"<|system|>\n{sys}\n"]
    for u,a in (history or [])[-8:]:
        msgs.append(f"<|user|>\n{u}\n<|assistant|>\n{a}")
    msgs.append(f"<|user|>\n{user_msg}\n<|assistant|>\n")
    return "\n".join(msgs)

def _format_prompt_with_attach(history, user_msg, summary):
    sys = ("You are a concise assistant. Use the ATTACHMENT SUMMARY as factual context. "
           "Do not copy it. If the summary lacks the answer, say so. "
           "Start directly. ≤ 60 words unless asked otherwise.")
    msgs = [f"<|system|>\n{sys}\n"]
    for u,a in (history or [])[-6:]:
        msgs.append(f"<|user|>\n{u}\n<|assistant|>\n{a}")
    msgs.append(f"<|system|>\nATTACHMENT SUMMARY:\n{summary}\n")
    msgs.append(f"<|user|>\n{user_msg}\n<|assistant|>\n")
    return "\n".join(msgs)

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
        prompt = _format_prompt_with_attach(history, message, summary)
    else:
        prompt = _format_prompt_no_attach(history, message)

    with llm.chat_session():
        out = llm.generate(
            prompt, max_tokens=140, temp=0.1, top_p=0.9, top_k=40,
            repeat_penalty=1.08, repeat_last_n=128, n_batch=256
        )
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
