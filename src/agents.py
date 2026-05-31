import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

def load_orders():
    with open("data/orders.json", "r", encoding="utf-8") as f:
        return json.load(f)

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

    prompt = f"""You are a helpful customer support agent for NovaMart.
Use the order information below to answer the customer's question.
Be friendly and concise.

Order info:
{context}

Customer message: {last_msg}
"""
    response = llm.invoke(prompt)
    return {
        "response": response.content,
        "messages": messages + [{"role": "assistant", "content": response.content}]
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
        "messages": messages + [{"role": "assistant", "content": response}]
    }

def escalation_agent(state):
    messages = state["messages"]
    last_msg = messages[-1]["content"]

    prompt = f"""You are a customer support escalation agent for NovaMart.
A customer has a complaint or issue that needs human attention.
Write a short, structured handoff summary for the human agent who will handle this.

Format your response exactly like this:
ESCALATION SUMMARY
------------------
Customer ID: {state.get('customer_id', 'Unknown')}
Issue: [one sentence description]
Urgency: [Low / Medium / High]
Recommended action: [what the human agent should do]
Customer message: "{last_msg}"
"""
    response = llm.invoke(prompt)
    return {
        "response": response.content,
        "escalate": True,
        "messages": messages + [{"role": "assistant", "content": response.content}]
    }

def chitchat_agent(state):
    chat_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
    last_msg = state["messages"][-1]["content"]
    prompt = f"""You are a friendly customer support agent for NovaMart, an online store.
Respond naturally to the customer's message. Keep it short and warm.
If they seem to need help, let them know you can assist with orders, returns, and shipping.
Customer: {last_msg}
"""
    response = chat_llm.invoke(prompt)
    return {
        "response": response.content,
        "messages": state["messages"] + [{"role": "assistant", "content": response.content}]
    }

