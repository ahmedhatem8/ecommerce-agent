from typing import TypedDict, Annotated, List, Optional
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import operator
from agents import order_lookup_agent, policy_agent, escalation_agent, chitchat_agent
from memory import load_memory, update_memory_after_session
from guardrails import check_input, check_policy


load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[List[dict], operator.add]  # short-term: full session history
    customer_id: str
    customer_profile: dict        # name, email, order data loaded at session start
    long_term_memory: dict        # persisted facts loaded from storage
    intent: str
    order_id: str
    response: str
    escalate: bool
    guardrail_triggered: bool

def supervisor_node(state: AgentState) -> AgentState:
    last_msg = state["messages"][-1]["content"]

    guard = check_input(last_msg)
    if guard.get("injection") or guard.get("toxic"):
        print(f"[Guardrail] Blocked: {guard['reason']}")
        if guard.get("toxic"):
            blocked_response = (
                "I'm sorry you're having such a difficult experience. "
                "I'm escalating this to a human agent who will reach out to help resolve things for you."
            )
        else:
            blocked_response = "I'm sorry, I can't process that request."
        return {
            "intent": "escalation",
            "guardrail_triggered": True,
            "escalate": True,
            "response": blocked_response,
        }

    customer_id = state.get("customer_id", "")

    # Load long-term memory on first turn (when it hasn't been set yet)
    long_term = state.get("long_term_memory") or {}
    if not long_term and customer_id:
        long_term = load_memory(customer_id)
        if long_term:
            print(f"[Memory] Returning customer {customer_id} — {long_term.get('total_sessions', 0)} past session(s)")
            if long_term.get("unresolved_complaints"):
                print(f"[Memory] Unresolved complaints: {long_term['unresolved_complaints']}")

    # Build conversation context for smarter intent classification
    prior_messages = state["messages"][:-1]
    history_context = ""
    if prior_messages:
        recent = prior_messages[-4:]  # last 4 messages for context
        history_context = "Recent conversation:\n" + "\n".join(
            f"{'Customer' if m['role'] == 'user' else 'Agent'}: {m['content']}"
            for m in recent
        ) + "\n\n"

    from langchain_groq import ChatGroq
    classifier = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    prompt = f"""You are an intent classifier for a customer support system.
Classify the customer's LATEST message into exactly one of these intents:

- order_lookup: asking about a SPECIFIC order's status, tracking, delivery progress, or estimated arrival
- policy: asking about returns, refunds, cancellations, damaged/wrong items, product prices, product features, product availability, warranties, catalog questions, shipping costs or fees (general — not tied to a specific order)
- escalation: complaints, anger, frustration, urgent unresolved problems
- chitchat: greetings, small talk, general casual questions unrelated to products or orders

Examples:
"What is the status of ORD-1001?" → order_lookup
"Where is my package?" → order_lookup
"When will my order arrive?" → order_lookup
"How much does the Mechanical Keyboard cost?" → policy
"Is the 27-inch Monitor in stock?" → policy
"What features does the Wireless Headphones have?" → policy
"What is your return policy?" → policy
"How much does shipping cost?" → policy
"What's the warranty on this product?" → policy
"I want to return my item" → policy
"I've been waiting 3 weeks and nobody helps me!" → escalation
"Hi there!" → chitchat
"Thank you, goodbye!" → chitchat

{history_context}Latest message: "{last_msg}"

Reply with ONLY one word: order_lookup, policy, escalation, or chitchat."""

    result = classifier.invoke(prompt)
    intent = result.content.strip().lower()
    if intent not in ["order_lookup", "policy", "escalation", "chitchat"]:
        intent = "escalation"

    print(f"[Supervisor] Intent: {intent}")
    return {"intent": intent, "long_term_memory": long_term}

def route_intent(state: AgentState) -> str:
    if state.get("guardrail_triggered"):
        return END
    intent = state["intent"]
    if intent not in ["order_lookup", "policy", "escalation", "chitchat"]:
        return "escalation"
    return intent

def order_lookup_node(state: AgentState) -> AgentState:
    return order_lookup_agent(state)

def policy_node(state: AgentState) -> AgentState:
    return policy_agent(state)

def escalation_node(state: AgentState) -> AgentState:
    return escalation_agent(state)

def chitchat_node(state: AgentState) -> AgentState:
    return chitchat_agent(state)

def policy_check_node(state: AgentState) -> AgentState:
    """Run after every non-escalation agent to catch policy violations in responses."""
    response = state.get("response", "")
    if not response:
        return {}

    result = check_policy(response)
    if result.get("violation"):
        print(f"[Policy Guardrail] Violation detected: {result['reason']}")
        p = state.get("customer_profile", {})
        safe_response = (
            "I'm sorry, I'm not authorised to make that commitment directly. "
            "I'm escalating this to a human agent who can resolve it for you.\n\n"
            "ESCALATION SUMMARY\n"
            "------------------\n"
            f"Customer ID: {p.get('customer_id', state.get('customer_id', 'Unknown'))}\n"
            f"Customer Name: {p.get('customer_name', 'Unknown')}\n"
            f"Issue: Agent response exceeded policy limits — {result['reason']}\n"
            "Urgency: High\n"
            "Recommended action: Review and manually approve the commitment."
        )
        return {
            "response": safe_response,
            "escalate": True,
            "messages": [{"role": "assistant", "content": safe_response}],
        }

    print(f"[Policy Guardrail] OK")

    # Detect when the policy agent described an escalation in its own words
    # (e.g. "I'll escalate this to a human agent") but didn't set the flag
    escalation_phrases = ["escalat", "human agent", "supervisor", "manual review", "human review"]
    if any(phrase in response.lower() for phrase in escalation_phrases):
        print(f"[Policy Guardrail] Escalation language detected — setting escalate=True")
        return {"escalate": True}

    return {}

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("supervisor",    supervisor_node)
    g.add_node("order_lookup",  order_lookup_node)
    g.add_node("policy",        policy_node)
    g.add_node("escalation",    escalation_node)
    g.add_node("chitchat",      chitchat_node)
    g.add_node("policy_check",  policy_check_node)
    g.set_entry_point("supervisor")
    g.add_conditional_edges("supervisor", route_intent, {
        "order_lookup": "order_lookup",
        "policy":       "policy",
        "escalation":   "escalation",
        "chitchat":     "chitchat",
        END:            END,
    })
    # Order lookup and policy agents pass through the policy check gate
    g.add_edge("order_lookup", "policy_check")
    g.add_edge("policy",       "policy_check")
    g.add_edge("policy_check", END)
    # Chitchat and escalation go straight to END — no policy commitments to check
    g.add_edge("chitchat",     END)
    g.add_edge("escalation",   END)
    return g.compile()

def lookup_customer_by_id(customer_id: str) -> dict:
    import json
    with open("data/orders.json", "r", encoding="utf-8") as f:
        orders = json.load(f)
    order = next((o for o in orders if o["customer_id"].upper() == customer_id.upper()), None)
    if not order:
        return {}
    return {
        "customer_id": order["customer_id"],
        "customer_name": order["customer_name"],
        "email": order["email"],
        "order_id": order["order_id"],
        "item": order["item"],
        "status": order["status"],
        "order_date": order["order_date"],
        "estimated_delivery": order.get("estimated_delivery", "N/A"),
        "total_price": order["total_price"],
    }

if __name__ == "__main__":
    graph = build_graph()

    order_id_input = input("Enter your customer ID (e.g. CUST-115): ").strip().upper()
    profile = lookup_customer_by_id(order_id_input)

    if not profile:
        print(f"No order found for {order_id_input}. Starting as anonymous session.\n")
        customer_id = ""
        greeting = "Hi there! How can I help you today?"
    else:
        customer_id = profile["customer_id"]
        greeting = f"Hi {profile['customer_name']}! How can I help you today?"

    print(f"Bot: {greeting}\n")

    state = {
        "messages": [{"role": "assistant", "content": greeting}],
        "customer_id": customer_id,
        "customer_profile": profile,
        "long_term_memory": {},
        "intent": "",
        "order_id": profile.get("order_id", ""),
        "response": greeting,
        "escalate": False,
        "guardrail_triggered": False,
    }

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("done", "stop"):
            update_memory_after_session(state)
            print("Session ended.")
            break

        state["messages"] = state["messages"] + [{"role": "user", "content": user_input}]
        state["guardrail_triggered"] = False

        result = graph.invoke(state)

        print(f"Bot: {result['response']}\n")

        state["messages"] = result["messages"] + [{"role": "assistant", "content": result["response"]}]
        state["intent"] = result["intent"]
        state["order_id"] = result.get("order_id", "")
        state["escalate"] = result.get("escalate", False)