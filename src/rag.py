import os
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

CHROMA_PATH = "knowledge_base/chroma_db"
POLICIES_DIR = "data/policies"

def load_documents():
    from langchain_core.documents import Document
    docs = []
    for fname in os.listdir(POLICIES_DIR):
        if fname.endswith(".md"):
            fpath = os.path.join(POLICIES_DIR, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                text = f.read()
            docs.append(Document(page_content=text, metadata={"source": fname}))
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