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
        all_memory[customer_id] = {
            "total_sessions": 0,
            "intents_history": [],
            "escalations": 0,
            "messages_history": [],
            "unresolved_issues": [],
        }

    profile = all_memory[customer_id]
    profile["total_sessions"] += 1
    profile["intents_history"].append(data.get("last_intent", ""))
    profile["messages_history"].append(data.get("last_message", ""))

    if data.get("escalated"):
        profile["escalations"] += 1
        profile["unresolved_issues"].append(data.get("last_message", ""))

    # keep only last 10 messages to avoid growing forever
    profile["intents_history"]  = profile["intents_history"][-10:]
    profile["messages_history"] = profile["messages_history"][-10:]
    profile["unresolved_issues"] = profile["unresolved_issues"][-5:]

    all_memory[customer_id] = profile

    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(all_memory, f, indent=2)

def update_memory_after_session(state: dict):
    customer_id = state.get("customer_id", "")
    if not customer_id:
        return

    messages = state.get("messages", [])
    
    # get only customer messages not agent responses
    customer_messages = [m["content"] for m in messages if m["role"] == "user"]
    
    # summarize the session using Groq
    from langchain_groq import ChatGroq
    from dotenv import load_dotenv
    load_dotenv()
    
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    
    if customer_messages:
        summary_prompt = f"""Summarize this customer support session in one short sentence.
Focus on what the customer needed and whether it was resolved.
Customer messages: {customer_messages}
Agent intent handled: {state.get('intent', '')}
Escalated: {state.get('escalate', False)}

Reply with ONE sentence only."""
        summary = llm.invoke(summary_prompt).content.strip()
    else:
        summary = "No messages in session."

    save_memory(customer_id, {
        "last_intent": state.get("intent", ""),
        "last_message": summary,
        "escalated": state.get("escalate", False),
    })

def get_customer_summary(customer_id: str) -> str:
    profile = load_memory(customer_id)
    if not profile:
        return "New customer, no history."
    return f"""Returning customer — {profile['total_sessions']} sessions.
Past intents: {', '.join(profile['intents_history'])}
Escalations: {profile['escalations']}
Recent messages: {profile['messages_history'][-3:]}
Unresolved issues: {profile['unresolved_issues']}"""