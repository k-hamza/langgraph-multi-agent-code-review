import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Config:
    # Modèles
    coder_model:    str = "qwen2.5-coder:7b"   # agents spécialisés
    reasoning_model: str = "llama3.1:8b"        # critique + synthèse
    
    # Ollama
    base_url: str = "http://localhost:11434"
    temperature: float = 0.2                    # faible = réponses stables
    
    # Contrôle du débat
    max_debate_rounds: int = 2
    consensus_threshold: float = 0.7

config = Config()


def get_langfuse_handler():
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key  = os.getenv("LANGFUSE_SECRET_KEY")
    host        = os.getenv("LANGFUSE_HOST", "http://localhost:3000")

    if not public_key or not secret_key:
        return None

    try:
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler   # ← corrigé

        Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
        return CallbackHandler()

    except ImportError:
        print("[Langfuse] Package non installé — traces désactivées.")
        return None
