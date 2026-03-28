from langgraph.graph import StateGraph, START, END

from state import ReviewState
from agents.security    import security_agent
from agents.performance import performance_agent
from agents.style       import style_agent
from agents.critic      import critic_agent
from agents.synthesis   import synthesis_agent
from config import config, get_langfuse_handler


# ─────────────────────────────────────────
# NŒUDS SIMPLES
# ─────────────────────────────────────────

def run_security(state: ReviewState) -> dict:
    return security_agent.invoke(state)

def run_performance(state: ReviewState) -> dict:
    return performance_agent.invoke(state)

def run_style(state: ReviewState) -> dict:
    return style_agent.invoke(state)

def run_critic(state: ReviewState) -> dict:
    return critic_agent.invoke(state)

def run_synthesis(state: ReviewState) -> dict:
    return synthesis_agent.invoke(state)


# ─────────────────────────────────────────
# ROUTAGE CONDITIONNEL
# ─────────────────────────────────────────

def should_debate(state: ReviewState) -> str:
    """
    Décide si on relance un tour de débat ou si on passe à la synthèse.

    Conditions pour débattre (les DEUX doivent être vraies) :
    - Il reste des tours disponibles
    - Le consensus est insuffisant OU des révisions sont demandées
    """
    verdict = state.get("critic_verdict")
    round_  = state.get("debate_round", 0)

    if verdict is None:
        return "synthesize"

    rounds_left   = round_ < config.max_debate_rounds
    needs_work    = (
        verdict.consensus_score < config.consensus_threshold
        or len(verdict.needs_revision) > 0
    )

    if rounds_left and needs_work:
        return "debate"
    return "synthesize"


def route_debate(state: ReviewState) -> list[str]:
    """
    Retourne la liste des agents à relancer.
    Appelé uniquement quand should_debate == "debate".
    """
    verdict = state["critic_verdict"]
    mapping = {
        "security":    "run_security",
        "performance": "run_performance",
        "style":       "run_style",
    }
    targets = [mapping[a] for a in verdict.needs_revision if a in mapping]
    # Sécurité : si la liste est vide malgré tout, on va à la synthèse
    return targets if targets else ["run_synthesis"]


# ─────────────────────────────────────────
# ASSEMBLAGE
# ─────────────────────────────────────────

def build_graph():
    graph = StateGraph(ReviewState)

    # Nœuds
    graph.add_node("run_security",    run_security)
    graph.add_node("run_performance", run_performance)
    graph.add_node("run_style",       run_style)
    graph.add_node("run_critic",      run_critic)
    graph.add_node("run_synthesis",   run_synthesis)

    # Séquence initiale : START → 3 agents → Critique
    graph.add_edge(START,           "run_security")
    graph.add_edge(START,           "run_performance")
    graph.add_edge(START,           "run_style")
    graph.add_edge("run_security",  "run_critic")
    graph.add_edge("run_performance","run_critic")
    graph.add_edge("run_style",     "run_critic")

    # Routage conditionnel après le Critique
    graph.add_conditional_edges(
        "run_critic",
        should_debate,
        {
            "debate":    "route_debate_node",
            "synthesize":"run_synthesis",
        }
    )

    # Nœud de routage du débat → fan-out sélectif
    graph.add_node("route_debate_node", lambda state: state)
    graph.add_conditional_edges(
        "route_debate_node",
        route_debate,
        {
            "run_security":    "run_security",
            "run_performance": "run_performance",
            "run_style":       "run_style",
            "run_synthesis":   "run_synthesis",
        }
    )

    # Les agents révisés rejoignent le Critique
    graph.add_edge("run_security",   "run_critic")
    graph.add_edge("run_performance","run_critic")
    graph.add_edge("run_style",      "run_critic")

    # Sortie
    graph.add_edge("run_synthesis", END)

    return graph.compile()


review_graph = build_graph()


def invoke_with_tracing(state: dict) -> dict:
    """Wrapper autour de invoke avec tracing Langfuse optionnel."""
    handler = get_langfuse_handler()
    config_dict = {"callbacks": [handler]} if handler else {}
    return review_graph.invoke(state, config=config_dict)


def stream_with_tracing(state: dict):
    """Wrapper autour de stream avec tracing Langfuse optionnel."""
    handler = get_langfuse_handler()
    config_dict = {"callbacks": [handler]} if handler else {}
    return review_graph.stream(state, config=config_dict, stream_mode="updates")

