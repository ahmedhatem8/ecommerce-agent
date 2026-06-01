from flask import Flask, render_template, request, jsonify
import sys
import os
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

app = Flask(__name__)

print("Initializing NovaMart AI Engine...")
from graph import build_graph, lookup_customer_by_id
from memory import update_memory_after_session

graph = build_graph()
sessions = {}
print("LangGraph ready.\n")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/eval')
def eval_dashboard():
    import json as _json
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eval', 'eval_results.json')
    if not os.path.exists(results_path):
        return "<h2>No eval results yet. Run <code>python eval/eval_suite.py</code> first.</h2>", 404
    with open(results_path, 'r', encoding='utf-8') as f:
        data = _json.load(f)
    # If retrieval metrics missing, inject empty placeholders so template renders
    if 'context_precision' not in data.get('summary', {}):
        data['summary']['context_precision'] = None
        data['summary']['context_recall']    = None
    if 'retrieval' not in data:
        data['retrieval'] = []
    return render_template('eval.html', data=data)


@app.route('/api/start', methods=['POST'])
def start_session():
    data = request.get_json()
    customer_id_input = data.get('customer_id', '').strip().upper()
    session_id = str(uuid.uuid4())

    profile = lookup_customer_by_id(customer_id_input) if customer_id_input else {}

    if profile:
        customer_id = profile['customer_id']
        greeting = f"Hi {profile['customer_name']}! How can I help you today?"
    else:
        customer_id = ''
        if customer_id_input:
            greeting = f"I couldn't find customer {customer_id_input}. No worries — I can still help! What do you need?"
        else:
            greeting = "Hi there! Welcome to NovaMart support. Ask me about orders, returns, shipping, or anything else!"

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

    sessions[session_id] = state
    return jsonify({"session_id": session_id, "greeting": greeting})


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    session_id = data.get('session_id')
    user_message = data.get('message', '').strip()

    if not user_message:
        return jsonify({"error": "Empty message"}), 400
    if session_id not in sessions:
        return jsonify({"error": "Session expired. Please refresh the page."}), 400

    state = sessions[session_id]
    state["messages"] = state["messages"] + [{"role": "user", "content": user_message}]
    state["guardrail_triggered"] = False

    try:
        result = graph.invoke(state)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    response = result.get('response', 'I had trouble processing that. Please try again.')
    intent = result.get('intent', '')
    guardrail = result.get('guardrail_triggered', False)

    state["messages"] = result.get("messages", state["messages"]) + [{"role": "assistant", "content": response}]
    state["intent"] = intent
    state["order_id"] = result.get("order_id", state.get("order_id", ""))
    state["escalate"] = result.get("escalate", False)
    # carry forward long-term memory once loaded by supervisor
    if result.get("long_term_memory"):
        state["long_term_memory"] = result["long_term_memory"]
    sessions[session_id] = state

    agent_map = {
        "order_lookup": {"label": "Order Lookup Agent", "color": "#3b82f6", "icon": "📦"},
        "policy":       {"label": "Policy Agent",       "color": "#10b981", "icon": "📋"},
        "escalation":   {"label": "Escalation Agent",   "color": "#ef4444", "icon": "🚨"},
        "chitchat":     {"label": "Chitchat Agent",     "color": "#f59e0b", "icon": "💬"},
    }
    info = agent_map.get(intent, {"label": "AI Agent", "color": "#6b7280", "icon": "🤖"})

    return jsonify({
        "response": response,
        "intent": intent,
        "agent_label": info["label"],
        "agent_color": info["color"],
        "agent_icon": info["icon"],
        "guardrail_triggered": guardrail,
    })


@app.route('/api/end', methods=['POST'])
def end_session():
    data = request.get_json()
    session_id = data.get('session_id')
    if session_id and session_id in sessions:
        try:
            update_memory_after_session(sessions[session_id])
        except Exception:
            pass
        del sessions[session_id]
    return jsonify({"status": "ok"})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
