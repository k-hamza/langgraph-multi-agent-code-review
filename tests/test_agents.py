from agents.security import security_agent
from agents.performance import performance_agent
from agents.style import style_agent

test_code = """
import sqlite3

def get_user(username, password):
    conn = sqlite3.connect("users.db")
    query = f"SELECT * FROM users WHERE name='{username}' AND pwd='{password}'"
    return conn.execute(query).fetchone()

SECRET_KEY = "hardcoded_secret_123"
"""

initial_state = {
    "code": test_code,
    "filename": "auth.py",
    "debate_round": 0,
}

for agent, key in [
    (security_agent,    "security_output"),
    (performance_agent, "performance_output"),
    (style_agent,       "style_output"),
]:
    result = agent.invoke(initial_state)
    out = result[key]
    print(f"[{out.agent_name.upper()}] sévérité={out.severity} | {len(out.findings)} findings")
    print(f"  → {out.summary[:80]}")
    print()