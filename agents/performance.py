# agents/performance.py
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from state import AgentState, AgentOutput, ReviewState, PerformanceOutput
from config import config

SYSTEM_PROMPT = """Tu es un expert en optimisation et performance Python.
Analyse le code fourni et identifie les problèmes de performance.

Pour chaque problème trouvé, indique :
- La nature du problème (complexité algorithmique, I/O bloquant, etc.)
- La ligne approximative si identifiable
- L'impact estimé (1-10)

Réponds en français. Sois précis et concis."""


def init_agent(state: ReviewState) -> AgentState:
    revision_notes = None
    if state.get("critic_verdict") and state["debate_round"] > 0:
        verdict = state["critic_verdict"]
        if "performance" in verdict.needs_revision:
            revision_notes = verdict.reasons.get("performance")

    return {
        "code": state["code"],
        "filename": state["filename"],
        "agent_name": "performance",
        "messages": [],
        "draft_findings": [],
        "revision_notes": revision_notes,
        "output": None,
    }


def analyze(state: AgentState) -> AgentState:
    llm = ChatOllama(
        model=config.coder_model,
        base_url=config.base_url,
        temperature=config.temperature,
    )

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

Identifie tous les problèmes de performance présents dans ce code."""

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()
    findings = [line.strip() for line in raw.split("\n") if line.strip()]

    return {
        **state,
        "messages": messages + [response],
        "draft_findings": findings,
    }


def format_output(state: AgentState) -> dict:
    findings = state["draft_findings"]

    severity = 5
    for finding in findings:
        for word in finding.split():
            if word.isdigit():
                val = int(word)
                if 1 <= val <= 10:
                    severity = max(severity, val)

    output = AgentOutput(
        agent_name="performance",
        findings=findings,
        severity=severity,
        summary=findings[0] if findings else "Aucun problème de performance détecté.",
        revised=state["revision_notes"] is not None,
        revision_notes=state["revision_notes"],
    )

    return {"performance_output": output}


def build_performance_agent():
    graph = StateGraph(
        AgentState,
        input=ReviewState,
        output=PerformanceOutput,
    )

    graph.add_node("init_agent", init_agent)
    graph.add_node("analyze", analyze)
    graph.add_node("format_output", format_output)

    graph.add_edge(START, "init_agent")
    graph.add_edge("init_agent", "analyze")
    graph.add_edge("analyze", "format_output")
    graph.add_edge("format_output", END)

    return graph.compile()


performance_agent = build_performance_agent()
