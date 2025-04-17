from langchain_community.embeddings import LlamaCppEmbeddings
from pathlib import Path

def local_embedding_model():
    model_path = Path("models") / "nomic-embed-text-v1.5.f16.gguf"

    return LlamaCppEmbeddings(
        model_path=str(model_path),
        n_ctx=2048,
        n_threads=4,
        verbose=False
    )
