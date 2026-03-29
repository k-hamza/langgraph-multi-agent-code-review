from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from state import ReviewState
from config import config
from typing_extensions import TypedDict
from typing import Optional
from prompt_loader import load_prompt


class SynthesisState(TypedDict):
    code:               str
    filename:           str
    debate_round:       int
    security_output:    Optional[object]
    performance_output: Optional[object]
    style_output:       Optional[object]
    critic_verdict:     Optional[object]
    raw_report:         str


class SynthesisOutput(TypedDict):
    final_report: str


def init_synthesis(state: ReviewState) -> SynthesisState:
    return {
        "code":               state["code"],
        "filename":           state["filename"],
        "debate_round":       state.get("debate_round", 0),
        "security_output":    state.get("security_output"),
        "performance_output": state.get("performance_output"),
        "style_output":       state.get("style_output"),
        "critic_verdict":     state.get("critic_verdict"),
        "raw_report":         "",
    }


def _format_output(output) -> str:
    if output is None:
        return "Non disponible."
    revised = " (révisé)" if output.revised else ""
    findings = "\n".join(f"  - {f}" for f in output.findings[:6])
    return (
        f"Sévérité : {output.severity}/10{revised}\n"
        f"Résumé : {output.summary}\n"
        f"Findings :\n{findings}"
    )


def synthesize(state: SynthesisState) -> SynthesisState:
    llm = ChatOllama(
        model=config.reasoning_model,    # llama3.1:8b
        base_url=config.base_url,
        temperature=config.temperature,
    )

    prompt = load_prompt("synthesis")

    verdict_text = "Non disponible."
    if state["critic_verdict"]:
        v = state["critic_verdict"]
        verdict_text = (
            f"Score de consensus : {v.consensus_score}\n"
            f"Évaluation : {v.global_assessment}\n"
            f"Tours de débat effectués : {state['debate_round']}"
        )

    user_message = prompt["user"].format(
        filename=state["filename"],
        security_section=_format_output(state["security_output"]),
        performance_section=_format_output(state["performance_output"]),
        style_section=_format_output(state["style_output"]),
        verdict_section=verdict_text
    )

    messages = [
        SystemMessage(content=prompt["system"]),
        HumanMessage(content=user_message),
    ]

    response = llm.invoke(messages)
    return {**state, "raw_report": response.content.strip()}


def format_report(state: SynthesisState) -> dict:
    return {"final_report": state["raw_report"]}



def build_synthesis_agent():
    graph = StateGraph(
        SynthesisState,
        input=ReviewState,
        output=SynthesisOutput,
    )

    graph.add_node("init_synthesis", init_synthesis)
    graph.add_node("synthesize",     synthesize)
    graph.add_node("format_report",  format_report)

    graph.add_edge(START,            "init_synthesis")
    graph.add_edge("init_synthesis", "synthesize")
    graph.add_edge("synthesize",     "format_report")
    graph.add_edge("format_report",  END)

    return graph.compile()


synthesis_agent = build_synthesis_agent()
