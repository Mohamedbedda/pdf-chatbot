from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    GROQ_API_KEY: str

    class Config:
        env_file = ".env"
    
    GROQ_MODELS: dict[str, str] = {
        "llama-3.1-8b - Fast":         "llama-3.1-8b-instant",
        "llama-4-scout - Long docs":    "meta-llama/llama-4-scout-17b-16e-instruct",
        "llama-3.3-70b - Best quality": "llama-3.3-70b-versatile",
        "kimi-k2 - High load":    "moonshotai/kimi-k2-instruct",
        "qwen3-32b - Reasoning":    "qwen/qwen3-32b",
    }
    RERANK_THRESHOLD: float = 0.5
    TOP_K: int = 10
    RERANK_TOP_K: int = 5
    FINAL_CONTEXT_K: int = 3

settings = Settings()