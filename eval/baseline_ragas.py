import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from src.rag import get_rag_chain

load_dotenv()

judge = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

eval_data = [
    {"question": "What is the return window for items at NovaMart?",
     "ground_truth": "Items may be returned within 30 days of delivery."},
    {"question": "How long does standard shipping take?",
     "ground_truth": "Standard shipping takes 5 to 7 business days."},
    {"question": "What happens if my order is delayed more than 3 days?",
     "ground_truth": "A $5 store credit is issued automatically for delays over 3 days."},
    {"question": "Can I return a digital product?",
     "ground_truth": "No, digital products and opened software are non-refundable."},
    {"question": "What refund amount requires human escalation?",
     "ground_truth": "Refunds above $300 require a human agent review and must be escalated."},
    {"question": "How much is express shipping?",
     "ground_truth": "Express shipping costs $12.99."},
    {"question": "How do I reset my password?",
     "ground_truth": "Click Forgot password on the login page and a reset link will be sent to your email."},
    {"question": "How many addresses can I save in my account?",
     "ground_truth": "You can save up to 5 addresses in your account settings."},
]

chain, retriever = get_rag_chain()

faithfulness_scores = []
relevancy_scores = []
precision_scores = []

for item in eval_data:
    question = item["question"]
    ground_truth = item["ground_truth"]
    answer = chain.invoke(question)
    sources = retriever.invoke(question)
    context = " ".join([doc.page_content for doc in sources])

    f_prompt = f"""Rate from 0 to 1 how faithful this answer is to the context.
1 = answer only uses information from context.
0 = answer contains information not in context.
Reply with ONLY a number between 0 and 1.
Context: {context}
Answer: {answer}"""
    f_score = float(judge.invoke(f_prompt).content.strip())

    r_prompt = f"""Rate from 0 to 1 how relevant this answer is to the question.
1 = perfectly answers the question.
0 = completely irrelevant.
Reply with ONLY a number between 0 and 1.
Question: {question}
Answer: {answer}"""
    r_score = float(judge.invoke(r_prompt).content.strip())

    p_prompt = f"""Rate from 0 to 1 how precisely the retrieved context contains the answer.
1 = context directly contains the answer.
0 = context is irrelevant to the question.
Reply with ONLY a number between 0 and 1.
Question: {question}
Context: {context}
Ground truth: {ground_truth}"""
    p_score = float(judge.invoke(p_prompt).content.strip())

    faithfulness_scores.append(f_score)
    relevancy_scores.append(r_score)
    precision_scores.append(p_score)
    print(f"Q: {question[:50]}...")
    print(f"   Faithfulness: {f_score} | Relevancy: {r_score} | Precision: {p_score}\n")

print("=" * 50)
print("NAIVE RAG BASELINE SCORES")
print("=" * 50)
print(f"Faithfulness:      {sum(faithfulness_scores)/len(faithfulness_scores):.3f}")
print(f"Answer relevancy:  {sum(relevancy_scores)/len(relevancy_scores):.3f}")
print(f"Context precision: {sum(precision_scores)/len(precision_scores):.3f}")
print("\nSave these numbers for your report.")