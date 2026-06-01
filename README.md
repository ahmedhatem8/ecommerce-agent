# NovaMart AI Customer Support Agent

A production-grade AI customer support system for an online electronics store. Built with LangGraph, LangChain, and a Groq-hosted LLaMA 3.3 70B model. The system handles order tracking, returns, product questions, and escalations — with hybrid RAG retrieval, multi-session memory, and three layers of guardrails.

---

## What It Does

Customers can type questions in a chat window. The system automatically figures out what they need and routes them to the right specialist:

- **Order questions** (where is my package, when does it arrive) → Order Lookup Agent calls a mock API
- **Policy and product questions** (how do I return something, what is the warranty) → Policy Agent searches the knowledge base
- **Complaints and anger** → Escalation Agent writes a structured handoff summary for a human agent
- **Greetings and small talk** → Chitchat Agent responds naturally

The system remembers the full conversation while you are talking (short-term memory) and remembers key facts about returning customers across sessions (long-term memory stored in a JSON file).

---

## Architecture

```
User message
     |
     v
Supervisor Node  <-- classifies intent, loads long-term memory
     |
     |--- order_lookup --> Order Lookup Agent --> Policy Check --> END
     |--- policy      --> Policy Agent       --> Policy Check --> END
     |--- escalation  --> Escalation Agent              -------> END
     |--- chitchat    --> Chitchat Agent                -------> END
     |--- [guardrail] -----------------------------------------> END
```

**Supervisor Node** — uses the LLM to classify the customer's message into one of four intents. It also runs input guardrails (injection and toxicity checks) before routing.

**Order Lookup Agent** — binds two tools to the LLM: `lookup_order_by_id` and `lookup_orders_by_customer`. The LLM calls whichever tool is needed, gets real order data back, and answers from that data only. Never invents order details.

**Policy Agent** — uses hybrid RAG to retrieve relevant chunks from the knowledge base (return policy, shipping policy, FAQ, product catalog), then answers the customer using only what was retrieved.

**Escalation Agent** — generates a structured escalation summary (customer ID, issue, urgency, recommended action) for a human support agent.

**Chitchat Agent** — friendly conversation that stays on NovaMart topics. Politely redirects off-topic requests.

**Policy Check Node** — runs after every agent response. Checks if the response promises something outside the agent's authority (e.g. a refund above the allowed threshold). If a violation is detected, the response is replaced with a safe escalation message.

---

## Project Structure

```
ecommerce-agent/
|
|-- app.py                      Flask web application (main entry point)
|
|-- src/
|   |-- graph.py                LangGraph graph definition and CLI runner
|   |-- agents.py               Four specialist agents
|   |-- rag.py                  RAG pipeline (naive and hybrid)
|   |-- memory.py               Short-term and long-term memory
|   |-- guardrails.py           Input, policy, and toxicity guardrails
|
|-- data/
|   |-- orders.json             200 synthetic customer orders
|   |-- generate_orders.py      Script that generated orders.json
|   |-- generate_policies.py    Script that generated policy documents
|   |-- customer_memory.json    Long-term memory (created on first use)
|   |-- policies/
|       |-- return_policy.md
|       |-- shipping_policy.md
|       |-- faq.md
|       |-- product_catalog.md
|
|-- eval/
|   |-- eval_suite.py           Main evaluation suite (30 test cases)
|   |-- baseline_ragas.py       Naive RAG scoring script
|   |-- hybrid_ragas.py         Hybrid RAG scoring script
|   |-- eval_results.json       Last saved evaluation results
|
|-- templates/
|   |-- index.html              Chat UI
|   |-- eval.html               Evaluation dashboard
|   |-- graph.html              Agent graph visualization
|
|-- knowledge_base/
|   |-- chroma_db/              ChromaDB vector store (built on first setup)
|
|-- requirements.txt
```

---

## Environment Dependencies

| Package | Purpose |
|---|---|
| flask | Web server and chat UI |
| langchain-groq | LangChain integration with Groq API (LLaMA 3.3 70B) |
| langchain-chroma | ChromaDB vector store integration |
| langchain-huggingface | HuggingFace sentence-transformer embeddings |
| langchain-text-splitters | Document chunking for RAG |
| sentence-transformers | Local embedding model (all-MiniLM-L6-v2) |
| rank-bm25 | BM25 keyword retrieval for hybrid search |
| langgraph | Multi-agent graph orchestration |
| python-dotenv | Load API keys from .env file |
| faker | Synthetic order data generation |
| ragas | RAG evaluation metrics |
| datasets | Required by ragas |

Python version: 3.10 or higher recommended.

---

## Setup and Running Locally

### Step 1 — Clone the repository

```
git clone <repo-url>
cd ecommerce-agent
```

### Step 2 — Create a virtual environment and install packages

```
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### Step 3 — Create your .env file

Create a file called `.env` in the root of the project with your Groq API key:

```
GROQ_API_KEY=your_groq_api_key_here
```

You can get a free API key at https://console.groq.com

### Step 4 — Build the vector store

This only needs to be done once. It reads the policy documents and builds the ChromaDB vector index.

```
python src/rag.py
```

You should see output like: `Created 42 chunks.` and `Vectorstore built and saved.`

### Step 5 — Run the web application

```
python app.py
```

Then open your browser and go to: `http://localhost:5000`

You will see the NovaMart chat interface. Enter a customer ID to start (e.g. `CUST-115`, `CUST-134`, `CUST-113`), or leave it blank for an anonymous session.

---

## How to Use the Chat

1. Open `http://localhost:5000`
2. Enter a customer ID in the box at the top, or leave blank
3. Type your question and press Enter or click Send
4. The agent label in the bottom corner shows which specialist handled your message
5. When you are done, click "End Session" — this saves the conversation to long-term memory

**Try these example messages:**
- `What is the status of ORD-1001?`
- `How do I return my item?`
- `How much does the Mechanical Keyboard cost?`
- `Ignore all previous instructions` (triggers injection guardrail)

---

## Viewing the Evaluation Dashboard

After running the eval suite, the results can be viewed in the browser:

```
python app.py
```

Then go to: `http://localhost:5000/eval`

This shows intent accuracy, escalation accuracy, resolution rate, policy compliance score, P95 latency, and per-test results for all 30 test cases.

---

## Running the Evaluation Suite

The eval suite runs 30 synthetic conversations and scores them automatically. It takes about 15 minutes because of rate-limit pauses on the Groq free tier.

```
python eval/eval_suite.py
```

Results are saved to `eval/eval_results.json` and printed to the terminal.

To compare naive RAG vs hybrid RAG scores separately:

```
# Baseline (naive) RAG
python eval/baseline_ragas.py

# Hybrid RAG
python eval/hybrid_ragas.py
```

---

## Running the CLI (No Browser Required)

You can also run the agent directly in the terminal without starting the web server:

```
cd src
python graph.py
```

It will ask for a customer ID, then start a text conversation. Type `done` or `stop` to end the session and save memory.

---

## RAG Strategy

The system uses **hybrid retrieval** combining two methods:

1. **Dense retrieval** — ChromaDB with `all-MiniLM-L6-v2` embeddings finds semantically similar chunks
2. **BM25 keyword retrieval** — finds chunks with exact keyword matches

Results from both are merged using **Reciprocal Rank Fusion (RRF)**, which scores each chunk by how highly it ranked in each list. This handles both precise keyword queries ("what is the $4.99 fee") and fuzzy semantic queries ("is my broken item covered").

**Naive RAG baseline scores** (dense retrieval only):
- Faithfulness: 0.925 | Answer Relevancy: 1.000 | Context Precision: 0.967

**Hybrid RAG scores** (after improvement):
- Faithfulness: 0.975 | Answer Relevancy: 1.000 | Context Precision: 0.950

---

## Guardrails

Three types are enforced on every message:

**Input guardrails (before routing)** — The supervisor node checks the customer's message for prompt injection (attempts to override system instructions) and toxicity (hate speech, threats). Injections are blocked immediately. Toxic messages are routed to the escalation agent with a calm response.

**Policy guardrails (after every response)** — The policy check node reviews the agent's response before it reaches the customer. If the response promises a refund or commitment above the agent's authority (e.g. approving a $500 refund autonomously), the response is replaced and escalated.

**Scope guardrails (in chitchat agent)** — The chitchat agent is instructed to stay on NovaMart topics only. Off-topic requests (recipes, general advice, etc.) are politely declined.

---

## Memory System

**Short-term memory** — the full conversation history is stored in the LangGraph state as a list of messages. Every agent can see the complete session history so customers never need to repeat themselves.

**Long-term memory** — when a session ends (user clicks "End Session" in the UI or types `done` in the CLI), the LLM extracts key facts from the conversation:
- Order IDs discussed
- Stated preferences (e.g. prefers express shipping)
- Unresolved complaints
- A one-sentence session summary

These are saved to `data/customer_memory.json`. When the same customer ID starts a new session, this history is loaded and injected into every agent's context.

---

## Knowledge Base

The knowledge base contains four documents in `data/policies/`:

- `return_policy.md` — 30-day return window, refund thresholds, damaged item rules
- `shipping_policy.md` — delivery times, fees, delay compensation
- `faq.md` — common questions about orders, accounts, and products
- `product_catalog.md` — 8 products with prices, features, warranties, and availability

All four documents are chunked (500 characters, 50 overlap), embedded, and stored in ChromaDB at `knowledge_base/chroma_db/`.

---

## Data

`data/orders.json` contains 200 synthetic orders generated with the Faker library. Each order has:
- order_id (ORD-1001 to ORD-1200)
- customer_id (CUST-100 to CUST-150)
- customer name, email
- item name, quantity, total price
- status (delivered, in_transit, delayed, returned, cancelled)
- order date, estimated delivery date
