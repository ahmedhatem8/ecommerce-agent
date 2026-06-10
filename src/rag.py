import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from rank_bm25 import BM25Okapi

load_dotenv()

CHROMA_PATH = "knowledge_base/chroma_db"
POLICIES_DIR = "data/policies"
KNOWLEDGE_BASE_DIR = "data/knowledge_base"

def load_documents():
    docs = []
    for directory, label in [
        (POLICIES_DIR, "policies"),
        (KNOWLEDGE_BASE_DIR, "knowledge_base"),
    ]:
        if not os.path.isdir(directory):
            continue
        for fname in sorted(os.listdir(directory)):
            if fname.endswith(".md"):
                fpath = os.path.join(directory, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    text = f.read()
                docs.append(Document(
                    page_content=text,
                    metadata={"source": fname, "directory": label},
                ))
    return docs

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def build_vectorstore():
    print("Loading documents...")
    docs = load_documents()
    print("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks.")
    print("Embedding and storing in ChromaDB...")
    vectorstore = Chroma.from_documents(
        chunks,
        get_embeddings(),
        persist_directory=CHROMA_PATH,
    )
    print("Vectorstore built and saved.")
    return vectorstore

def load_vectorstore():
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embeddings(),
    )

def get_rag_chain():
    retriever = load_vectorstore().as_retriever(search_kwargs={"k": 3})
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    prompt = ChatPromptTemplate.from_template("""
You are a helpful customer support agent for NovaMart.
Answer the question based only on the context below.

Context: {context}

Question: {question}
""")
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever

def get_hybrid_chain():
    documents = load_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    tokenized = [chunk.page_content.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized)

    vectorstore = load_vectorstore()
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def hybrid_retrieve(query):
        dense_results = dense_retriever.invoke(query)
        tokenized_query = query.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)
        top_bm25_indices = sorted(range(len(bm25_scores)),
                                  key=lambda i: bm25_scores[i], reverse=True)[:5]
        bm25_results = [chunks[i] for i in top_bm25_indices]

        # Reciprocal Rank Fusion: score each doc by its rank in both result lists
        # Higher score = appeared near the top in more lists = more relevant
        K = 60
        rrf_scores = {}
        for rank, doc in enumerate(dense_results):
            key = doc.page_content
            rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (rank + K)
        for rank, doc in enumerate(bm25_results):
            key = doc.page_content
            rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (rank + K)

        # Collect unique docs and sort by combined RRF score descending
        seen = {}
        for doc in dense_results + bm25_results:
            if doc.page_content not in seen:
                seen[doc.page_content] = doc
        ranked = sorted(seen.values(),
                        key=lambda d: rrf_scores.get(d.page_content, 0),
                        reverse=True)

        return ranked[:4] if ranked else []

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful customer support agent for NovaMart.
    Answer the question based only on the context below.

    Context: {context}

    Question: {question}
    """)

    def hybrid_chain(query):
        docs = hybrid_retrieve(query)
        context = "\n\n".join([d.page_content for d in docs])
        response = (prompt | llm | StrOutputParser()).invoke({
            "context": context,
            "question": query
        })
        return response, docs

    return hybrid_chain, hybrid_retrieve

if __name__ == "__main__":
    build_vectorstore()
    chain, retriever = get_rag_chain()
    question = "What is the return window for items?"
    answer = chain.invoke(question)
    sources = retriever.invoke(question)
    print("\n--- Answer ---")
    print(answer)
    print("\n--- Sources used ---")
    for doc in sources:
        print(f"  - {doc.metadata.get('source', 'unknown')}")