import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from rag import get_hybrid_chain

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
    {"question": "how long do i have 2 send back item",
     "ground_truth": "Items must be shipped back within 7 days of receiving the return label."},
    {"question": "wat if my package is late",
     "ground_truth": "A $5 store credit is issued automatically for delays over 3 days."},
    {"question": "can i get money back for broken headphones",
     "ground_truth": "Damaged items get a full refund with photo evidence, no return needed."},
    {"question": "do you ship outside egypt",
     "ground_truth": "Currently NovaMart ships within Egypt and the UAE only."},
    {"question": "how much to ship fast",
     "ground_truth": "Express shipping costs $12.99 and takes 2 to 3 business days."},
    {"question": "i got wrong product what do i do",
     "ground_truth": "Wrong item sent gets a full refund or replacement, no return needed."},
    {"question": "warranty on electronics",
     "ground_truth": "Electronics carry a 1-year manufacturer warranty."},
]

chain, retriever = get_hybrid_chain()

faithfulness_scores = []
relevancy_scores = []
precision_scores = []

for item in eval_data:
    question = item["question"]
    ground_truth = item["ground_truth"]
    answer, sources = chain(question)
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
print("HYBRID RAG SCORES")
print("=" * 50)
print(f"Faithfulness:      {sum(faithfulness_scores)/len(faithfulness_scores):.3f}")
print(f"Answer relevancy:  {sum(relevancy_scores)/len(relevancy_scores):.3f}")
print(f"Context precision: {sum(precision_scores)/len(precision_scores):.3f}")
print("Naive RAG:  F=0.925 | AR=1.000 | CP=0.967")