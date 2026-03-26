from state import AgentOutput, CriticVerdict, ReviewState, AgentState
from config import config

# Test instanciation AgentOutput
out = AgentOutput(
    agent_name="security",
    findings=["SQL injection line 42", "hardcoded password line 7"],
    severity=8,
    summary="Two critical vulnerabilities found."
)
print(f"AgentOutput OK : {out.agent_name}, severity={out.severity}")

# Test instanciation CriticVerdict
verdict = CriticVerdict(
    needs_revision=["performance"],
    reasons={"performance": "No complexity analysis provided."},
    consensus_score=0.5,
    global_assessment="Security analysis solid, performance too shallow."
)
print(f"CriticVerdict OK : consensus={verdict.consensus_score}")

# Test ReviewState
state: ReviewState = {
    "code": "def foo(): pass",
    "filename": "example.py",
    "debate_round": 0,
}
print(f"ReviewState OK : keys={list(state.keys())}")

# Test config frozen
print(f"Config OK : coder={config.coder_model}, rounds={config.max_debate_rounds}")
try:
    config.max_debate_rounds = 99  # doit lever une erreur
    print("ERREUR : config aurait dû être frozen !")
except Exception as e:
    print(f"Frozen OK : {type(e).__name__}")
    