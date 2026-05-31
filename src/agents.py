import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

def load_orders():
    with open("data/orders.json", "r", encoding="utf-8") as f:
        return json.load(f)

def _profile_context(state) -> str:
    p = state.get("customer_profile", {})
    if not p:
        return ""
    return f"""Customer profile:
Name: {p.get('customer_name', 'Unknown')}
Email: {p.get('email', 'Unknown')}
Customer ID: {p.get('customer_id', 'Unknown')}
"""

def order_lookup_agent(state):
    messages = state["messages"]
    last_msg = messages[-1]["content"]

    orders = load_orders()

    order_id = None
    for word in last_msg.upper().split():
        if word.startswith("ORD-"):
            order_id = word.strip(".,!?")
            break

    if order_id:
        order = next((o for o in orders if o["order_id"] == order_id), None)
    else:
        customer_id = state.get("customer_id", "")
        order = next((o for o in orders if o["customer_id"] == customer_id), None)

    if not order:
        context = "No order found for this customer."
    else:
        context = f"""
Order ID: {order['order_id']}
Item: {order['item']} x{order['quantity']}
Status: {order['status']}
Order date: {order['order_date']}
Estimated delivery: {order.get('estimated_delivery', 'N/A')}
Total: ${order['total_price']}
"""

    history = "\n".join(
        f"{'Customer' if m['role'] == 'user' else 'Agent'}: {m['content']}"
        for m in messages[:-1]
    )
    prompt = f"""You are a helpful customer support agent for NovaMart.
Use the order information below to answer the customer's question.
Be friendly and concise. Remember everything the customer said earlier in this conversation.

{_profile_context(state)}
Order info:
{context}

Conversation so far:
{history}

Customer message: {last_msg}
"""
    response = llm.invoke(prompt)
    return {
        "response": response.content,
        "messages": [{"role": "assistant", "content": response.content}]
    }


def policy_agent(state):
    from rag import get_rag_chain
    messages = state["messages"]
    last_msg = messages[-1]["content"]

    chain, retriever = get_rag_chain()
    answer = chain.invoke(last_msg)

    response = f"{answer}"
    return {
        "response": response,
        "messages": [{"role": "assistant", "content": response}]
    }

def escalation_agent(state):
    messages = state["messages"]
    last_msg = messages[-1]["content"]
    p = state.get("customer_profile", {})

    history = "\n".join(
        f"{'Customer' if m['role'] == 'user' else 'Agent'}: {m['content']}"
        for m in messages[:-1]
    )
    prompt = f"""You are a customer support escalation agent for NovaMart.
A customer has a complaint or issue that needs human attention.
Write a short, structured handoff summary for the human agent who will handle this.

Format your response exactly like this:
ESCALATION SUMMARY
------------------
Customer ID: {p.get('customer_id', state.get('customer_id', 'Unknown'))}
Customer Name: {p.get('customer_name', 'Unknown')}
Email: {p.get('email', 'Unknown')}
Issue: [one sentence description]
Urgency: [Low / Medium / High]
Recommended action: [what the human agent should do]
Customer message: "{last_msg}"

Conversation so far:
{history}
"""
    response = llm.invoke(prompt)
    return {
        "response": response.content,
        "escalate": True,
        "messages": [{"role": "assistant", "content": response.content}]
    }

def chitchat_agent(state):
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    chat_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
    messages = state["messages"]
    p = state.get("customer_profile", {})

    profile_note = ""
    if p.get("customer_name"):
        profile_note = (
            f"The customer's name is {p['customer_name']}, "
            f"email is {p['email']}, "
            f"customer ID is {p['customer_id']}. "
            "Use this information naturally if asked about their details."
        )

    lc_messages = [SystemMessage(content=(
        "You are a friendly customer support agent for NovaMart, an online store. "
        "Respond naturally to the customer's message. Keep it short and warm. "
        "Remember everything the customer has told you in this conversation. "
        "If they seem to need help, let them know you can assist with orders, returns, and shipping. "
        + profile_note
    ))]
    for m in messages:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            lc_messages.append(AIMessage(content=m["content"]))

    response = chat_llm.invoke(lc_messages)
    return {
        "response": response.content,
        "messages": [{"role": "assistant", "content": response.content}]
    }

