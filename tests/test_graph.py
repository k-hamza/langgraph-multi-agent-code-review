from graph import review_graph

test_code = """
import sqlite3

def get_user(username, password):
    conn = sqlite3.connect("users.db")
    query = f"SELECT * FROM users WHERE name='{username}' AND pwd='{password}'"
    return conn.execute(query).fetchone()

SECRET_KEY = "hardcoded_secret_123"
"""

result = review_graph.invoke({
    "code": test_code,
    "filename": "auth.py",
    "debate_round": 0,
})

print(f"Tours de débat : {result['debate_round']}")
print(f"Rapport généré : {'oui' if result.get('final_report') else 'non'}")
print("\n--- RAPPORT ---\n")
print(result["final_report"])
