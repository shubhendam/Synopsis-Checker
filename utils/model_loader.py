from pathlib import Path
from langchain_community.llms import LlamaCpp

def local_llm():
    model_path = Path("models") / "Llama-3.2-3B-Instruct-Q4_K_M.gguf"

    llm = LlamaCpp(
        model_path=str(model_path),
        temperature=0.3,
        max_tokens=512,
        top_p=0.95,
        n_ctx=2048,
        n_batch=16,
        n_threads=4,  #adjust based on CPU
        stop=["###"],
        verbose=False
    )

    return llm