# agents/critic.py
import json
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from state import AgentState, CriticVerdict, ReviewState
from config import config
from typing_extensions import TypedDict
from typing import Optional


# ─────────────────────────────────────────
# ÉTAT INTERNE DU CRITIQUE
# ─────────────────────────────────────────

class CriticState(TypedDict):
    # Reçu du ReviewState
    code: str
    filename: str
    debate_round: int
    security_output:    Optional[object]
    performance_output: Optional[object]
    style_output:       Optional[object]
    # Privé
    raw_analysis: str        # réponse brute du LLM
    verdict: Optional[CriticVerdict]


class CriticOutput(TypedDict):
    critic_verdict: Optional[CriticVerdict]
    debate_round:   int


# ─────────────────────────────────────────
# PROMPT SYSTÈME
# ─────────────────────────────────────────

SYSTEM_PROMPT = """Tu es un expert en revue de code chargé d'évaluer la qualité des analyses produites par trois agents spécialisés.

Tu dois évaluer :
1. Chaque agent est-il resté dans son domaine ? (sécurité / performance / style)
2. Les analyses sont-elles cohérentes ou contradictoires ?
3. Certaines analyses sont-elles trop superficielles ?

Tu dois répondre UNIQUEMENT avec un objet JSON valide, sans texte avant ni après, avec cette structure exacte :
{
  "needs_revision": ["security", "performance", "style"],  // liste vide si tout va bien
  "reasons": {
    "security": "raison si révision demandée",
    "performance": "raison si révision demandée",
    "style": "raison si révision demandée"
  },
  "consensus_score": 0.8,
  "global_assessment": "Évaluation globale en 2-3 phrases."
}

needs_revision ne contient que les agents qui doivent retravailler.
consensus_score est entre 0.0 (analyses contradictoires/insuffisantes) et 1.0 (analyses solides et cohérentes)."""


# ─────────────────────────────────────────
# NŒUDS
# ─────────────────────────────────────────

def init_critic(state: ReviewState) -> CriticState:
    return {
        "code":               state["code"],
        "filename":           state["filename"],
        "debate_round":       state.get("debate_round", 0),
        "security_output":    state.get("security_output"),
        "performance_output": state.get("performance_output"),
        "style_output":       state.get("style_output"),
        "raw_analysis":       "",
        "verdict":            None,
    }


def _format_agent_output(output) -> str:
    """Formate un AgentOutput en texte lisible pour le LLM."""
    if output is None:
        return "Non disponible."
    findings_text = "\n".join(f"  - {f}" for f in output.findings[:8])
    return (
        f"Sévérité globale : {output.severity}/10\n"
        f"Résumé : {output.summary}\n"
        f"Findings (extrait) :\n{findings_text}"
    )


def analyze(state: CriticState) -> CriticState:
    llm = ChatOllama(
        model=config.reasoning_model,
        base_url=config.base_url,
        temperature=config.temperature,
    )

    user_message = f"""Fichier analysé : {state["filename"]}
Tour de débat : {state["debate_round"]}

═══ ANALYSE SÉCURITÉ ═══
{_format_agent_output(state["security_output"])}

═══ ANALYSE PERFORMANCE ═══
{_format_agent_output(state["performance_output"])}

═══ ANALYSE STYLE ═══
{_format_agent_output(state["style_output"])}

Évalue ces trois analyses et retourne ton verdict JSON."""

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]

    response = llm.invoke(messages)
    return {**state, "raw_analysis": response.content.strip()}


def parse_verdict(state: CriticState) -> dict:
    """
    Parse le JSON retourné par le LLM.
    Stratégie défensive : si le JSON est malformé, on produit
    un verdict neutre qui laisse passer vers la synthèse.
    """
    raw = state["raw_analysis"]

    try:
        # Le LLM glisse parfois du texte avant/après le JSON
        # On cherche le premier '{' et le dernier '}'
        start = raw.index("{")
        end   = raw.rindex("}") + 1
        data  = json.loads(raw[start:end])

        verdict = CriticVerdict(
            needs_revision   = data.get("needs_revision", []),
            reasons          = data.get("reasons", {}),
            consensus_score  = float(data.get("consensus_score", 0.5)),
            global_assessment= data.get("global_assessment", ""),
        )

    except (ValueError, KeyError, json.JSONDecodeError) as e:
        # Verdict de repli : on laisse passer, la synthèse prend le relais
        print(f"[CRITIC] Avertissement : parsing JSON échoué ({e}), verdict neutre appliqué.")
        verdict = CriticVerdict(
            needs_revision=[],
            reasons={},
            consensus_score=0.75,
            global_assessment="Analyse du critique indisponible — synthèse directe.",
        )

    # On incrémente debate_round ici : c'est le Critique qui valide
    # qu'un tour complet s'est écoulé
    return {
        "critic_verdict": verdict,
        "debate_round":   state["debate_round"] + 1,
    }


# ─────────────────────────────────────────
# ASSEMBLAGE
# ─────────────────────────────────────────

def build_critic_agent():
    graph = StateGraph(
        CriticState,
        input=ReviewState,
        output=CriticOutput,
    )

    graph.add_node("init_critic",   init_critic)
    graph.add_node("analyze",       analyze)
    graph.add_node("parse_verdict", parse_verdict)

    graph.add_edge(START,          "init_critic")
    graph.add_edge("init_critic",  "analyze")
    graph.add_edge("analyze",      "parse_verdict")
    graph.add_edge("parse_verdict", END)

    return graph.compile()


critic_agent = build_critic_agent()
