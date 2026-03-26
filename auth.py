# auth.py
import sqlite3

def get_user(username, password):
    conn = sqlite3.connect("users.db")
    query = f"SELECT * FROM users WHERE name='{username}' AND pwd='{password}'"
    return conn.execute(query).fetchone()

SECRET_KEY = "hardcoded_secret_123"