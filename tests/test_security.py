import asyncio
from agents.security import security_agent

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

result = security_agent.invoke(initial_state)
print(f"Agent : {result['security_output'].agent_name}")
print(f"Sévérité : {result['security_output'].severity}")
print(f"Findings ({len(result['security_output'].findings)}) :")
for f in result['security_output'].findings[:3]:
    print(f"  - {f}")
print(f"Résumé : {result['security_output'].summary}")