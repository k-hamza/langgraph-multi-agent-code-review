from agents.security    import security_agent
from agents.performance import performance_agent
from agents.style       import style_agent
from agents.critic      import critic_agent

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

# Tour des agents spécialisés
state.update(security_agent.invoke(state))
state.update(performance_agent.invoke(state))
state.update(style_agent.invoke(state))

# Tour du Critique
result = critic_agent.invoke(state)
verdict = result["critic_verdict"]

print(f"Consensus score : {verdict.consensus_score}")
print(f"Agents à retravailler : {verdict.needs_revision}")
print(f"Évaluation : {verdict.global_assessment}")
if verdict.reasons:
    print("Raisons :")
    for agent, reason in verdict.reasons.items():
        print(f"  [{agent}] {reason}")
