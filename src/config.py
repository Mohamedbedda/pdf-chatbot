from pydantic_settings import BaseSettings
import os
from dotenv import load_dotenv


class Settings(BaseSettings):
    load_dotenv()
    GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
    ACCESS_PASSWORD: str = os.environ.get("ACCESS_PASSWORD","")

    GROQ_MODELS: dict[str, str] = {
        "llama-3.1-8b - Fast": "llama-3.1-8b-instant",
        "gpt-oss-20b - Reasoning": "openai/gpt-oss-20b",
        "kimi-k2 - High throughput": "moonshotai/kimi-k2-instruct",
        "qwen3-32b - Fast reasoning": "qwen/qwen3-32b",
    }

    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    TOP_K: int = 10
    RERANK_TOP_K: int = 7
    RERANK_THRESHOLD: float = 0.5
    FINAL_CONTEXT_K: int = 3
    TEMPERATURE: float = 0.1
    MAX_OUTPUT_TOKENS: int = 500

settings = Settings()