# Tool Disclosure Document
## NovaMart AI Customer Support Agent — CSAI 422 Capstone

---

### Frameworks

| Tool | Version | Justification | Role in Architecture |
|---|---|---|---|
| LangGraph | 0.2.x | Provides stateful, graph-based multi-agent orchestration with conditional routing | Defines the supervisor node, four specialist agent nodes, and the policy check node; manages shared AgentState across turns |
| LangChain | 0.3.x | Standardized abstractions for LLM calls, tools, prompts, retrievers, and output parsers | Used throughout agents.py, rag.py, and guardrails.py for building chains, binding tools, and invoking the LLM |
| Flask | 3.x | Lightweight Python web framework | Serves the chat UI, evaluation dashboard, and REST API endpoints (/api/start, /api/chat, /api/end) |

---

### Libraries

| Tool | Version | Justification | Role in Architecture |
|---|---|---|---|
| langchain-groq | 0.2.x | Official LangChain adapter for the Groq inference API | Instantiates ChatGroq (LLaMA 3.3 70B) used by all agents, the supervisor classifier, the guardrail checker, and the LLM-as-judge |
| langchain-chroma | 0.1.x | LangChain integration for ChromaDB vector store | Stores and queries dense embeddings of knowledge base chunks in rag.py |
| langchain-huggingface | 0.1.x | LangChain integration for HuggingFace embedding models | Wraps the sentence-transformer model so it can be used as a LangChain embedding function |
| sentence-transformers | 3.x | Provides the all-MiniLM-L6-v2 embedding model locally | Encodes knowledge base chunks and query text into vectors for dense retrieval |
| chromadb | 0.5.x | Vector database that persists and searches embeddings on disk | Stores all document chunk embeddings in knowledge_base/chroma_db; used for dense retrieval in the hybrid pipeline |
| rank-bm25 | 0.2.x | Efficient BM25 keyword scoring over a document corpus | Powers the sparse retrieval leg of the hybrid RAG pipeline; complements dense retrieval for exact keyword matches |
| langchain-text-splitters | 0.3.x | Splits long documents into overlapping chunks | Splits policy and catalog documents into 500-character chunks with 50-character overlap before indexing |
| python-dotenv | 1.x | Loads environment variables from a .env file | Loads the GROQ_API_KEY at startup in every module |
| faker | 30.x | Generates realistic fake names, emails, and dates | Used in data/generate_orders.py to create 200 synthetic customer orders for testing and evaluation |
| ragas | 0.1.x | RAG evaluation framework with standard retrieval metrics | Used as the conceptual basis for our faithfulness, answer relevancy, and context precision scoring scripts (baseline_ragas.py and hybrid_ragas.py) |
| datasets | 3.x | HuggingFace datasets library | Required dependency of ragas |

---

### Models

| Tool | Provider | Justification | Role in Architecture |
|---|---|---|---|
| LLaMA 3.3 70B Versatile | Meta / Groq | High-quality open model available free via Groq with very fast inference; strong instruction-following and JSON output | Powers all agents (order lookup, policy, escalation, chitchat), the supervisor intent classifier, guardrail checks, memory extraction, and the LLM-as-judge in evaluation |
| all-MiniLM-L6-v2 | HuggingFace / sentence-transformers | Small, fast, and accurate embedding model; runs locally with no API cost; well-suited for sentence-level semantic similarity | Encodes knowledge base chunks and retrieval queries for the dense retrieval leg of hybrid RAG |

---

### Datasets

| Dataset | Source | Justification | Role in Architecture |
|---|---|---|---|
| Synthetic order dataset (200 orders) | Generated using Faker library | No real customer data available; Faker produces realistic names, emails, and dates with a fixed seed for reproducibility | Serves as the mock order database that the Order Lookup Agent queries via its tools |
| Return and Refund Policy | Manually authored | Written to match a realistic e-commerce policy with specific thresholds that the guardrails enforce ($100 auto, $100–$300 supervisor, $300+ human) | Indexed in ChromaDB; retrieved by the Policy Agent to answer return and refund questions |
| Shipping Policy | Manually authored | Written to include concrete values (fees, delivery times, delay compensation) that the evaluation suite can verify | Indexed in ChromaDB; retrieved for shipping fee and delivery questions |
| FAQ Document | Manually authored | Covers common account, order, and product questions that do not fit neatly into return or shipping policies | Indexed in ChromaDB; acts as a catch-all retrieval source |
| Product Catalog | Manually authored | Lists 8 products with exact prices, features, warranties, and stock status; anchors all product question answers | Indexed in ChromaDB; retrieved by the Policy Agent for product price, feature, and warranty questions |

---

### AI Assistants Used During Development

| Tool | Provider | How It Was Used |
|---|---|---|
| Claude (claude-sonnet-4-6) | Anthropic | Used as a coding assistant during development for writing, reviewing, and debugging code across all modules |
