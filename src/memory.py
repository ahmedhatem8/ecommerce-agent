import json
import os

MEMORY_FILE = "data/customer_memory.json"

def load_memory(customer_id: str) -> dict:
    if not os.path.exists(MEMORY_FILE):
        return {}
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        all_memory = json.load(f)
    return all_memory.get(customer_id, {})

def save_memory(customer_id: str, data: dict):
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            all_memory = json.load(f)
    else:
        all_memory = {}
    if customer_id not in all_memory:
        all_memory[customer_id] = {}
    all_memory[customer_id].update(data)
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(all_memory, f, indent=2)

def update_memory_after_session(state: dict):
    customer_id = state.get("customer_id", "")
    if not customer_id:
        return
    messages = state.get("messages", [])
    last_intent = state.get("intent", "")
    escalated = state.get("escalate", False)
    session_summary = {
        "last_intent": last_intent,
        "last_message": messages[-1]["content"] if messages else "",
        "escalated": escalated,
        "total_sessions": load_memory(customer_id).get("total_sessions", 0) + 1
    }
    save_memory(customer_id, session_summary)