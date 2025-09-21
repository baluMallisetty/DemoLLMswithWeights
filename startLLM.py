from gpt4all import GPT4All

for row in GPT4All.list_models():
    print(row["filename"])
MODEL = r"C:\Users\balum\OneDrive\Documents\AI\LLMs\models\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"   # or any .gguf / .bin you copied
m = GPT4All(MODEL)
print(m.generate("Hello, my name is"))