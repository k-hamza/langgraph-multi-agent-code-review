from __future__ import annotations
from typing import Annotated, Optional
from dataclasses import dataclass, field
from typing_extensions import TypedDict, NotRequired
from langgraph.graph.message import add_messages


@dataclass
class AgentOutput:
    agent_name: str
    findings: list[str]
    severity: int
    summary: str
    revised: bool = False
    revision_notes: Optional[str] = None


@dataclass
class CriticVerdict:
    needs_revision: list[str]
    reasons: dict[str, str]
    consensus_score: float
    global_assessment: str


class ReviewState(TypedDict):
    code: str
    filename: str
    debate_round:       int
    security_output:    NotRequired[Optional[AgentOutput]]
    performance_output: NotRequired[Optional[AgentOutput]]
    style_output:       NotRequired[Optional[AgentOutput]]
    critic_verdict:     NotRequired[Optional[CriticVerdict]]
    final_report:       NotRequired[Optional[str]]


class AgentState(TypedDict):
    code: str
    filename: str
    agent_name: str
    messages: Annotated[list, add_messages]
    draft_findings: list[str]
    revision_notes: Optional[str]
    output: Optional[AgentOutput]


class SecurityOutput(TypedDict):
    security_output: Optional[AgentOutput]


class PerformanceOutput(TypedDict):
    performance_output: Optional[AgentOutput]


class StyleOutput(TypedDict):
    style_output: Optional[AgentOutput]