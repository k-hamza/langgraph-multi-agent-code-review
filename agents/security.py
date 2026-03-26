# agents/security.py
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from state import AgentState, AgentOutput, ReviewState, SecurityOutput
from config import config

# ─────────────────────────────────────────
# PROMPT SYSTÈME
# ─────────────────────────────────────────

SYSTEM_PROMPT = """Tu es un expert en sécurité applicative Python.
Analyse le code fourni et identifie les vulnérabilités de sécurité.

Pour chaque problème trouvé, indique :
- La nature de la vulnérabilité (injection, exposition de données, etc.)
- La ligne approximative si identifiable
- Le niveau de criticité (1-10)

Réponds en français. Sois précis et concis."""


# ─────────────────────────────────────────
# NŒUDS DU SUBGRAPH
# ─────────────────────────────────────────

def init_agent(state: ReviewState) -> AgentState:
    """
    Construit l'AgentState initial depuis le ReviewState.
    Injecte les revision_notes si l'agent a été renvoyé par le Critique.
    """
    revision_notes = None

    # 2ème tour : le Critique a peut-être des notes pour cet agent
    if state.get("critic_verdict") and state["debate_round"] > 0:
        verdict = state["critic_verdict"]
        if "security" in verdict.needs_revision:
            revision_notes = verdict.reasons.get("security")

    return {
        "code": state["code"],
        "filename": state["filename"],
        "agent_name": "security",
        "messages": [],
        "draft_findings": [],
        "revision_notes": revision_notes,
        "output": None,
    }


def analyze(state: AgentState) -> AgentState:
    """
    Appel LLM principal. Construit le prompt, appelle Ollama,
    stocke la réponse dans messages et draft_findings.
    """
    llm = ChatOllama(
        model=config.coder_model,
        base_url=config.base_url,
        temperature=config.temperature,
    )

    # Si l'agent est en révision, on enrichit le prompt avec le feedback
    revision_context = ""
    if state["revision_notes"]:
        revision_context = f"""
        
FEEDBACK DU CRITIQUE (révision demandée) :
{state["revision_notes"]}

Tiens compte de ce feedback et approfondis ton analyse en conséquence."""

    user_message = f"""Fichier : {state["filename"]}
```python
{state["code"]}
```
{revision_context}

Identifie toutes les vulnérabilités de sécurité présentes dans ce code."""

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]

    response = llm.invoke(messages)

    # On extrait les findings : chaque ligne non vide = un finding
    raw = response.content.strip()
    findings = [line.strip() for line in raw.split("\n") if line.strip()]

    return {
        **state,
        "messages": messages + [response],
        "draft_findings": findings,
    }


def format_output(state: AgentState) -> dict:
    """
    Construit l'AgentOutput depuis draft_findings.
    Calcule la sévérité globale : max des scores trouvés, ou 5 par défaut.
    Retourne UNIQUEMENT la clé ReviewState à mettre à jour.
    """
    findings = state["draft_findings"]

    # Heuristique simple : cherche des chiffres dans les findings
    # pour estimer la sévérité globale
    severity = 5  # valeur par défaut
    for finding in findings:
        for word in finding.split():
            if word.isdigit():
                val = int(word)
                if 1 <= val <= 10:
                    severity = max(severity, val)

    output = AgentOutput(
        agent_name="security",
        findings=findings,
        severity=severity,
        summary=findings[0] if findings else "Aucune vulnérabilité détectée.",
        revised=state["revision_notes"] is not None,
        revision_notes=state["revision_notes"],
    )

    # On retourne UNIQUEMENT la clé du ReviewState à mettre à jour
    return {"security_output": output}


# ─────────────────────────────────────────
# ASSEMBLAGE DU SUBGRAPH
# ─────────────────────────────────────────

def build_security_agent() -> object:
    graph = StateGraph(
        AgentState,
        input=ReviewState,
        output=SecurityOutput
    )

    graph.add_node("init_agent", init_agent)
    graph.add_node("analyze", analyze)
    graph.add_node("format_output", format_output)

    graph.add_edge(START, "init_agent")
    graph.add_edge("init_agent", "analyze")
    graph.add_edge("analyze", "format_output")
    graph.add_edge("format_output", END)

    return graph.compile()


security_agent = build_security_agent()
