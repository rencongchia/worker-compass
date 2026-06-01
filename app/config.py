import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # Primary LLM (Llama 3.3-70B via Groq — free, 30 RPM)
    llm_model: str = "llama-3.3-70b-versatile"
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_api_key: str = ""

    # Fallback LLM (Qwen3 via OpenRouter) — activated on 429 from primary
    fallback_model: str = "qwen/qwen3-next-80b-a3b-instruct:free"
    fallback_base_url: str = "https://openrouter.ai/api/v1"
    fallback_api_key: str = ""

    # Retrieval
    similarity_threshold: float = 0.27   # refusal gate — cosine similarity (BGE-M3 range: 0.1–0.6)
    top_k: int = 8                        # chunks passed to LLM
    bm25_candidates: int = 20             # BM25 candidates before RRF
    dense_candidates: int = 20            # dense candidates before RRF

    # Paths
    lancedb_path: str = "corpus/lancedb"
    bm25_path: str = "corpus/bm25_index.pkl"


def load_config() -> Config:
    return Config(
        llm_model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
        llm_base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
        llm_api_key=os.getenv("GROQ_API_KEY", ""),
        fallback_model=os.getenv("FALLBACK_MODEL", "qwen/qwen3-next-80b-a3b-instruct:free"),
        fallback_base_url=os.getenv("FALLBACK_BASE_URL", "https://openrouter.ai/api/v1"),
        fallback_api_key=os.getenv("OPENROUTER_API_KEY", ""),
        similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.25")),
        lancedb_path=os.getenv("LANCEDB_PATH", "corpus/lancedb"),
        bm25_path=os.getenv("BM25_PATH", "corpus/bm25_index.pkl"),
    )
