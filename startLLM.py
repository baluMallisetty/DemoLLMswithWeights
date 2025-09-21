# from gpt4all import GPT4All

# for row in GPT4All.list_models():
#     print(row["filename"])
# MODEL = r"C:\Users\balum\OneDrive\Documents\AI\LLMs\models\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"   # or any .gguf / .bin you copied
# m = GPT4All(MODEL)
# print(m.generate("what is 66+66?"))


from gpt4all import GPT4All
import time, os

MODEL = r"C:\Users\balum\OneDrive\Documents\AI\LLMs\models\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
m = GPT4All(MODEL, allow_download=False, n_threads= max(2, os.cpu_count()//2),
            n_ctx=2048)

t0 = time.time()

with m.chat_session():
    with open('products-100.csv', 'r') as file:
        data = file.read()
    out = m.generate(
        "you will anlyze and find me products whose price is greater than 60usd from csv file"+data,
        max_tokens=10000,
        temp=0.1,
        top_p=0.95
    )
print(out)
print(f"took {time.time()-t0:.1f}s")

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