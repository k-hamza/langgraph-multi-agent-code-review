from agents.security    import security_agent
from agents.performance import performance_agent
from agents.style       import style_agent
from agents.critic      import critic_agent
from agents.synthesis   import synthesis_agent
from config import config

test_code = """
import sqlite3

def get_user(username, password):
    conn = sqlite3.connect("users.db")
    query = f"SELECT * FROM users WHERE name='{username}' AND pwd='{password}'"
    return conn.execute(query).fetchone()

SECRET_KEY = "hardcoded_secret_123"
"""

state = {
    "code": test_code,
    "filename": "auth.py",
    "debate_round": 0,
}

print("=== Tour 1 : agents spécialisés ===")
state.update(security_agent.invoke(state))
state.update(performance_agent.invoke(state))
state.update(style_agent.invoke(state))
print(f"security    : sévérité={state['security_output'].severity}")
print(f"performance : sévérité={state['performance_output'].severity}")
print(f"style       : sévérité={state['style_output'].severity}")

print("\n=== Critique ===")
state.update(critic_agent.invoke(state))
verdict = state["critic_verdict"]
print(f"consensus={verdict.consensus_score} | révisions={verdict.needs_revision}")
print(f"debate_round après critique : {state['debate_round']}")

# Tour 2 si nécessaire
if verdict.needs_revision and state["debate_round"] <= config.max_debate_rounds:
    print(f"\n=== Tour 2 : révision de {verdict.needs_revision} ===")
    for agent, key in [
        (security_agent,    "security_output"),
        (performance_agent, "performance_output"),
        (style_agent,       "style_output"),
    ]:
        name = key.replace("_output", "")
        if name in verdict.needs_revision:
            state.update(agent.invoke(state))
            print(f"{name} révisé : revised={state[key].revised}")

print("\n=== Synthèse ===")
state.update(synthesis_agent.invoke(state))
print(state["final_report"])
