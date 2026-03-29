from pathlib import Path
import yaml


_cache: dict[str, dict] = {}


def load_prompt(agent_name: str) -> dict:
    """
    Charge le prompt YAML d'un agent depuis prompts/{agent_name}.yaml.
    Met en cache au premier appel — le fichier n'est lu qu'une fois
    par exécution.
    """
    if agent_name in _cache:
        return _cache[agent_name]

    path = Path(__file__).parent / "prompts" / f"{agent_name}.yaml"

    if not path.exists():
        raise FileNotFoundError(
            f"Prompt introuvable : {path}\n"
            f"Agents disponibles : {[p.stem for p in path.parent.glob('*.yaml')]}"
        )

    prompt = yaml.safe_load(path.read_text(encoding="utf-8"))

    # Validation minimale
    for key in ("system", "user"):
        if key not in prompt:
            raise ValueError(f"Clé '{key}' manquante dans {path.name}")

    _cache[agent_name] = prompt
    return prompt