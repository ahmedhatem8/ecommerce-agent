import json
import os
from datetime import datetime

MEMORY_FILE = "data/customer_memory.json"


def load_memory(customer_id: str) -> dict:
    """Load long-term memory for a customer."""
    if not os.path.exists(MEMORY_FILE):
        return {}
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        all_memory = json.load(f)
    return all_memory.get(customer_id, {})


def save_memory(customer_id: str, extracted: dict):
    """Persist extracted facts to long-term memory."""
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            all_memory = json.load(f)
    else:
        all_memory = {}

    if customer_id not in all_memory:
        all_memory[customer_id] = {
            "total_sessions": 0,
            "past_orders": [],
            "stated_preferences": [],
            "unresolved_complaints": [],
            "session_summaries": [],
            "escalations": 0,
            "last_seen": "",
        }

    profile = all_memory[customer_id]
    profile["total_sessions"] += 1
    profile["last_seen"] = datetime.now().strftime("%Y-%m-%d")

    for order in extracted.get("past_orders", []):
        if order not in profile["past_orders"]:
            profile["past_orders"].append(order)

    for pref in extracted.get("stated_preferences", []):
        if pref not in profile["stated_preferences"]:
            profile["stated_preferences"].append(pref)

    for complaint in extracted.get("unresolved_complaints", []):
        profile["unresolved_complaints"].append(complaint)

    if extracted.get("session_summary"):
        profile["session_summaries"].append(extracted["session_summary"])

    if extracted.get("escalated"):
        profile["escalations"] += 1

    # keep lists bounded
    profile["past_orders"] = profile["past_orders"][-10:]
    profile["stated_preferences"] = profile["stated_preferences"][-10:]
    profile["unresolved_complaints"] = profile["unresolved_complaints"][-5:]
    profile["session_summaries"] = profile["session_summaries"][-5:]

    all_memory[customer_id] = profile

    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(all_memory, f, indent=2)


def update_memory_after_session(state: dict):
    """Extract and persist key facts from the completed session using LLM."""
    customer_id = state.get("customer_id", "")
    if not customer_id:
        return

    messages = state.get("messages", [])
    if not messages:
        return

    from langchain_groq import ChatGroq
    from dotenv import load_dotenv
    load_dotenv()

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    conversation = "\n".join(
        f"{'Customer' if m['role'] == 'user' else 'Agent'}: {m['content']}"
        for m in messages
    )

    extract_prompt = f"""Analyze this customer support conversation and extract key facts.
Return a JSON object with exactly these fields:
- "session_summary": one sentence describing what happened and whether it was resolved
- "past_orders": list of order IDs the customer mentioned (e.g. ["ORD-1001"]), empty list if none
- "stated_preferences": list of preferences the customer expressed (e.g. ["wants express shipping", "prefers email contact"]), empty list if none
- "unresolved_complaints": list of unresolved issues as short strings, empty list if everything was resolved

Conversation:
{conversation}

Return ONLY valid JSON, no markdown, no explanation."""

    try:
        result = llm.invoke(extract_prompt).content.strip()
        if result.startswith("```"):
            result = result.split("```")[1]
            if result.startswith("json"):
                result = result[4:]
        extracted = json.loads(result.strip())
    except Exception:
        extracted = {
            "session_summary": "Session completed.",
            "past_orders": [],
            "stated_preferences": [],
            "unresolved_complaints": [],
        }

    extracted["escalated"] = state.get("escalate", False)
    save_memory(customer_id, extracted)


def long_term_context(state) -> str:
    """Format long-term memory as a readable block for agent prompts."""
    mem = state.get("long_term_memory", {})
    if not mem:
        return ""

    lines = []
    if mem.get("total_sessions", 0) > 0:
        lines.append(f"This customer has contacted us {mem['total_sessions']} time(s) before.")
    if mem.get("past_orders"):
        lines.append(f"Past orders discussed: {', '.join(mem['past_orders'])}")
    if mem.get("stated_preferences"):
        lines.append(f"Known preferences: {'; '.join(mem['stated_preferences'])}")
    if mem.get("unresolved_complaints"):
        lines.append(f"Unresolved complaints from past sessions: {'; '.join(mem['unresolved_complaints'])}")
    if mem.get("session_summaries"):
        lines.append(f"Last session: {mem['session_summaries'][-1]}")

    if not lines:
        return ""
    return "Long-term customer history:\n" + "\n".join(lines)


def short_term_history(messages, exclude_last=True) -> str:
    """Format the full session conversation as a readable string."""
    history = messages[:-1] if exclude_last else messages
    if not history:
        return ""
    return "\n".join(
        f"{'Customer' if m['role'] == 'user' else 'Agent'}: {m['content']}"
        for m in history
    )
