from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import operator
from agents import order_lookup_agent, policy_agent, escalation_agent, chitchat_agent
from memory import load_memory, update_memory_after_session
from guardrails import check_input, check_policy


load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[List[dict], operator.add]
    customer_id: str
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
        return {
            "intent": "escalation",
            "guardrail_triggered": True,
            "response": "I'm sorry, I can't process that request."
        }

    customer_id = state.get("customer_id", "")
    past = load_memory(customer_id)
    if past:
        print(f"[Memory] Returning customer {customer_id} — past sessions: {past.get('total_sessions', 0)}")

    from langchain_groq import ChatGroq
    classifier = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    prompt = f"""You are an intent classifier for a customer support system.
Classify this message into exactly one of these intents:
- order_lookup: customer asking about order status, tracking, delivery, shipping
- policy: customer asking about returns, refunds, cancellations, damaged items, wrong item
- escalation: complaints, anger, frustration, urgent problems
- chitchat: greetings, small talk, general questions, anything casual

Reply with ONLY one word: order_lookup, policy, escalation, or chitchat.

Message: "{last_msg}"
"""
    result = classifier.invoke(prompt)
    intent = result.content.strip().lower()
    if intent not in ["order_lookup", "policy", "escalation", "chitchat"]:
        intent = "escalation"

    print(f"[Supervisor] Intent: {intent}")
    return {"intent": intent}

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

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("supervisor",   supervisor_node)
    g.add_node("order_lookup", order_lookup_node)
    g.add_node("policy",       policy_node)
    g.add_node("escalation",   escalation_node)
    g.add_node("chitchat",     chitchat_node)
    g.set_entry_point("supervisor")
    g.add_conditional_edges("supervisor", route_intent, {
        "order_lookup": "order_lookup",
        "policy":       "policy",
        "escalation":   "escalation",
        "chitchat":     "chitchat",
        END:            END,
    })
    g.add_edge("order_lookup", END)
    g.add_edge("policy",       END)
    g.add_edge("escalation",   END)
    g.add_edge("chitchat",     END)
    return g.compile()

if __name__ == "__main__":
    graph = build_graph()
    state = {
        "messages": [{"role": "user", "content": "hi"}],
        "customer_id": "CUST-101",
        "intent": "",
        "order_id": "",
        "response": "",
        "escalate": False,
        "guardrail_triggered": False,
    }
    result = graph.invoke(state)
    update_memory_after_session(result)
    print("Intent detected:", result["intent"])
    print("Response:", result["response"])