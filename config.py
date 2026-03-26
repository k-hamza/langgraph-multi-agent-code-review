from dataclasses import dataclass

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