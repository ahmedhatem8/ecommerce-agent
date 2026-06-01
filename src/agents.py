import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from memory import long_term_context, short_term_history

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

def load_orders():
    with open("data/orders.json", "r", encoding="utf-8") as f:
        return json.load(f)

# ── Mock API tools ────────────────────────────────────────────────────────────

@tool
def lookup_order_by_id(order_id: str) -> str:
    """Look up a specific order by its order ID (e.g. ORD-1001).
    Returns order status, item, delivery date, and total price."""
    orders = load_orders()
    order = next((o for o in orders if o["order_id"].upper() == order_id.upper()), None)
    if not order:
        return f"No order found with ID {order_id}."
    return (
        f"Order ID: {order['order_id']}\n"
        f"Item: {order['item']} x{order['quantity']}\n"
        f"Status: {order['status']}\n"
        f"Order date: {order['order_date']}\n"
        f"Estimated delivery: {order.get('estimated_delivery', 'N/A')}\n"
        f"Total: ${order['total_price']}"
    )

@tool
def lookup_orders_by_customer(customer_id: str) -> str:
    """Look up all orders belonging to a customer by their customer ID (e.g. CUST-115).
    Returns a list of their orders with status and details."""
    orders = load_orders()
    customer_orders = [o for o in orders if o["customer_id"].upper() == customer_id.upper()]
    if not customer_orders:
        return f"No orders found for customer {customer_id}."
    lines = []
    for o in customer_orders:
        lines.append(
            f"- {o['order_id']}: {o['item']} x{o['quantity']} | "
            f"Status: {o['status']} | "
            f"Delivery: {o.get('estimated_delivery', 'N/A')} | "
            f"Total: ${o['total_price']}"
        )
    return f"Orders for {customer_id}:\n" + "\n".join(lines)

ORDER_TOOLS = [lookup_order_by_id, lookup_orders_by_customer]
_tool_map = {t.name: t for t in ORDER_TOOLS}

def _profile_context(state) -> str:
    p = state.get("customer_profile", {})
    if not p:
        return ""
    return (
        f"Customer: {p.get('customer_name', 'Unknown')} "
        f"(ID: {p.get('customer_id', 'Unknown')}, Email: {p.get('email', 'Unknown')})"
    )


def order_lookup_agent(state):
    messages = state["messages"]
    last_msg = messages[-1]["content"]

    short_term = short_term_history(messages)
    long_term = long_term_context(state)
    profile = _profile_context(state)
    customer_id = state.get("customer_id", "")

    # Build context block so the LLM knows who the customer is
    context_lines = []
    if profile:
        context_lines.append(profile)
    if customer_id:
        context_lines.append(f"Customer ID on file: {customer_id}")
    if state.get("order_id"):
        context_lines.append(
            f"Active order on file: {state['order_id']} "
            f"— when the customer asks about 'my order' without specifying an ID, look this up first"
        )
    if long_term:
        context_lines.append(long_term)
    if short_term:
        context_lines.append(f"Conversation so far:\n{short_term}")
    context_block = "\n\n".join(context_lines)

    system_msg = SystemMessage(content=(
        "You are a helpful customer support agent for NovaMart.\n"
        "Use the available tools to look up order information, then answer the customer "
        "in a friendly and concise way. The customer should never have to repeat themselves.\n"
        "CRITICAL: Only report order IDs, item names, prices, and statuses that come "
        "directly from tool results. Never invent, guess, or fabricate any order details. "
        "If a tool returns no results, tell the customer you could not find that order.\n\n"
        + context_block
    ))

    # Build conversation history for the LLM
    lc_messages = [system_msg]
    for m in messages[:-1]:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            lc_messages.append(AIMessage(content=m["content"]))
    lc_messages.append(HumanMessage(content=last_msg))

    # Round 1: LLM decides which tool to call
    llm_with_tools = llm.bind_tools(ORDER_TOOLS)
    ai_msg = llm_with_tools.invoke(lc_messages)

    # Execute any tool calls the LLM requested
    if ai_msg.tool_calls:
        lc_messages.append(ai_msg)
        for tc in ai_msg.tool_calls:
            tool_fn = _tool_map.get(tc["name"])
            if tool_fn:
                result = tool_fn.invoke(tc["args"])
                print(f"[Tool] {tc['name']}({tc['args']}) -> {result[:80]}...")
            else:
                result = f"Unknown tool: {tc['name']}"
            lc_messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

        # Round 2: compose answer from ONLY the verified tool data
        # Using a targeted prompt (not full history) prevents the LLM from
        # inventing order IDs beyond what the tools actually returned
        tool_results_text = "\n\n".join(
            m.content for m in lc_messages if isinstance(m, ToolMessage)
        )
        summary_prompt = (
            "You are a customer support agent for NovaMart.\n"
            "Answer the customer's question using ONLY the exact order data below. "
            "Do NOT add, invent, or reference any order IDs, items, or prices not listed here.\n"
            "Always include the order ID(s) in your response so the customer knows exactly which order you mean.\n\n"
            f"Verified order data from system:\n{tool_results_text}\n\n"
            f"Customer question: {last_msg}\n\n"
            "Friendly, concise answer:"
        )
        final_msg = llm.invoke(summary_prompt)
        response_text = final_msg.content
    else:
        # LLM answered directly without needing a tool call
        response_text = ai_msg.content

    return {
        "response": response_text,
        "messages": [{"role": "assistant", "content": response_text}],
    }


def policy_agent(state):
    from rag import get_hybrid_chain
    import re
    messages = state["messages"]
    last_msg = messages[-1]["content"]

    short_term = short_term_history(messages)
    long_term = long_term_context(state)

    # Enrich the RAG query with conversation context so retrieval is more accurate
    if short_term:
        rag_query = f"Context from conversation:\n{short_term}\n\nCustomer question: {last_msg}"
    else:
        rag_query = last_msg

    # Use retriever directly so we can supplement for multi-product queries
    _, hybrid_retriever = get_hybrid_chain()
    docs = hybrid_retriever(rag_query)

    # When multiple products are mentioned, run a targeted query per product
    # so no product's details get crowded out by the other
    product_names = list(dict.fromkeys(re.findall(
        r'\b(webcam|keyboard|headphones?|monitor|lamp|hub|speaker|camera|tablet|mouse)\b',
        last_msg.lower()
    )))
    if len(product_names) > 1:
        seen = {d.page_content for d in docs}
        for prod in product_names:
            for doc in hybrid_retriever(f"{prod} price warranty features"):
                if doc.page_content not in seen:
                    docs.append(doc)
                    seen.add(doc.page_content)

    context = "\n\n".join(d.page_content for d in docs[:6])

    # Let the LLM answer using the retrieved context
    profile = _profile_context(state)
    prompt = f"""You are a helpful customer support agent for NovaMart.
Use the policy information below to answer the customer's question.
Be friendly, concise, and consistent with everything said earlier in the conversation.
The customer should never have to repeat themselves.
For product questions, include all relevant details from the context: price, key features, availability, and warranty.

{profile}
{long_term}

Policy information (retrieved):
{context}

Conversation so far this session:
{short_term if short_term else "(this is the first message)"}

Customer: {last_msg}"""

    response = llm.invoke(prompt)
    return {
        "response": response.content,
        "messages": [{"role": "assistant", "content": response.content}],
    }


def escalation_agent(state):
    messages = state["messages"]
    last_msg = messages[-1]["content"]
    p = state.get("customer_profile", {})

    short_term = short_term_history(messages)
    long_term = long_term_context(state)

    prompt = f"""You are a customer support escalation agent for NovaMart.
A customer needs to be handed off to a human agent. Write a structured escalation summary.

{long_term}

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

Full conversation this session:
{short_term if short_term else "(no prior messages)"}"""

    response = llm.invoke(prompt)
    return {
        "response": response.content,
        "escalate": True,
        "messages": [{"role": "assistant", "content": response.content}],
    }


def chitchat_agent(state):
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    chat_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)
    messages = state["messages"]
    p = state.get("customer_profile", {})
    long_term = long_term_context(state)

    profile_note = ""
    if p.get("customer_name"):
        profile_note = (
            f"The customer's name is {p['customer_name']}, "
            f"email is {p['email']}, customer ID is {p['customer_id']}. "
        )

    system_content = (
        "You are a friendly customer support agent for NovaMart, an online store. "
        "Respond naturally. Keep it short and warm. "
        "Remember everything the customer has told you in this conversation — they should never repeat themselves. "
        "Always let the customer know you can help with orders, returns, and shipping whenever it fits naturally. "
        "IMPORTANT: You only discuss NovaMart topics. If asked about anything unrelated — recipes, general advice, "
        "or anything not about NovaMart products, orders, returns, or shipping — politely decline and redirect. "
        + profile_note
    )
    if long_term:
        system_content += f"\n\n{long_term}"

    lc_messages = [SystemMessage(content=system_content)]
    for m in messages:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            lc_messages.append(AIMessage(content=m["content"]))

    response = chat_llm.invoke(lc_messages)
    return {
        "response": response.content,
        "messages": [{"role": "assistant", "content": response.content}],
    }
