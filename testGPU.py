from llama_cpp import Llama


# path to your quantized gguf file
model_path = r"C:\Users\balum\OneDrive\Documents\AI\LLMs\models\phi-4-Q4_K_S.gguf"

# create the model with GPU offload (offload_layers auto-tunes)
llm = Llama(
    model_path=model_path,
    n_gpu_layers=-1,        # offload everything possible to GPU
    n_ctx=16384,
    verbose=True
)

try:
    print(llm("Q: What is RAG in AI?\nA:", max_tokens=64)["choices"][0]["text"])
finally:
    llm = None  # avoid __del__ crash