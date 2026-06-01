"""
eval/hybrid_ragas.py — Hybrid RAG evaluation (BM25 + Dense + RRF)
Same 20 questions as baseline_ragas.py. Saves results to eval/hybrid_rag_results.json.

Usage:
    cd ecommerce-agent
    python eval/hybrid_ragas.py
"""

import sys, os, json, time, statistics
from datetime import datetime

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

from langchain_groq import ChatGroq
from rag import get_hybrid_chain

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hybrid_rag_results.json")

# ── Identical questions to baseline_ragas.py ─────────────────────────────────

EVAL_QUESTIONS = [

    # ── Medium ────────────────────────────────────────────────────────────────

    {
        "id": "RQ-01", "difficulty": "medium", "topic": "returns",
        "question": "What is the return window for items at NovaMart?",
        "ground_truth": "Items may be returned within 30 days of delivery.",
    },
    {
        "id": "RQ-02", "difficulty": "medium", "topic": "shipping",
        "question": "How long does standard shipping take and how much does it cost for orders below $50?",
        "ground_truth": "Standard shipping takes 5 to 7 business days and costs $4.99 for orders below $50.",
    },
    {
        "id": "RQ-03", "difficulty": "medium", "topic": "products",
        "question": "What warranty does the Wireless Headphones come with?",
        "ground_truth": "The Wireless Headphones carry a 1-year manufacturer warranty.",
    },
    {
        "id": "RQ-04", "difficulty": "medium", "topic": "shipping",
        "question": "How much does express shipping cost and how fast is it?",
        "ground_truth": "Express shipping costs $12.99 and takes 2 to 3 business days.",
    },
    {
        "id": "RQ-05", "difficulty": "medium", "topic": "products",
        "question": "Is the 27-inch Monitor currently in stock and what is its price?",
        "ground_truth": "Yes, the Monitor 27 inch is in stock at a price of $299.99.",
    },

    # ── Hard ──────────────────────────────────────────────────────────────────

    {
        "id": "RQ-06", "difficulty": "hard", "topic": "returns",
        "question": "My refund amount is exactly $100. Will it be processed automatically or does it need supervisor approval?",
        "ground_truth": "Refunds between $100 and $300 require supervisor approval. An exact $100 refund falls in that range and must be escalated.",
    },
    {
        "id": "RQ-07", "difficulty": "hard", "topic": "returns",
        "question": "I received the wrong item 40 days after delivery. Can I still get a refund even though the 30-day window has passed?",
        "ground_truth": "Yes. Wrong item exceptions grant a full refund or replacement with no return needed. The 30-day window does not apply.",
    },
    {
        "id": "RQ-08", "difficulty": "hard", "topic": "shipping",
        "question": "My package arrived exactly 2 days past the estimated delivery date. Do I qualify for the $5 store credit?",
        "ground_truth": "No. The $5 store credit is only issued automatically for delays over 3 days. Two days does not qualify.",
    },
    {
        "id": "RQ-09", "difficulty": "hard", "topic": "products",
        "question": "I purchased both the Monitor 27 inch and the Phone Case. What is the warranty period for each?",
        "ground_truth": "The Monitor 27 inch has a 1-year manufacturer warranty. The Phone Case has a 90-day warranty.",
    },
    {
        "id": "RQ-10", "difficulty": "hard", "topic": "shipping",
        "question": "Can I get same-day delivery if I live in Alexandria, Egypt?",
        "ground_truth": "No. Same-day delivery is only available in Cairo for orders placed before 12 PM.",
    },
    {
        "id": "RQ-11", "difficulty": "hard", "topic": "returns",
        "question": "After I receive the return shipping label, how many days do I have to actually ship the item back?",
        "ground_truth": "You must ship the item back within 7 days of receiving the return label.",
    },
    {
        "id": "RQ-12", "difficulty": "hard", "topic": "products",
        "question": "wat is da return window 4 da headphones n exactly how do i start da return process",
        "ground_truth": "The return window is 30 days from delivery. Contact support with your order ID to receive a return label within 24 hours, then ship back within 7 days.",
    },
    {
        "id": "RQ-13", "difficulty": "hard", "topic": "returns",
        "question": "How many business days does it take to receive my refund after NovaMart gets the returned item?",
        "ground_truth": "The refund is issued within 5 business days of NovaMart receiving the returned item.",
    },

    # ── Very Hard ─────────────────────────────────────────────────────────────

    {
        "id": "RQ-14", "difficulty": "very_hard", "topic": "multi-policy",
        "question": "If I order a USB-C Hub and a Desk Lamp together, will I get free shipping? Show the calculation.",
        "ground_truth": "Yes. USB-C Hub is $34.99 and Desk Lamp is $29.99, totalling $64.98 which exceeds the $50 free shipping threshold.",
    },
    {
        "id": "RQ-15", "difficulty": "very_hard", "topic": "multi-policy",
        "question": "My Mechanical Keyboard stopped working after 8 months. Is it still under warranty, and what exactly should I do?",
        "ground_truth": "Yes. The Mechanical Keyboard has a 1-year manufacturer warranty so 8 months is within coverage. Contact support with your order ID to initiate a warranty claim.",
    },
    {
        "id": "RQ-16", "difficulty": "very_hard", "topic": "products",
        "question": "How much does the Bluetooth Speaker cost in your catalog?",
        "ground_truth": "NovaMart does not sell a Bluetooth Speaker. The catalog includes Wireless Headphones, Phone Case, USB-C Hub, Laptop Stand, Mechanical Keyboard, Webcam HD, Desk Lamp, and Monitor 27 inch.",
    },
    {
        "id": "RQ-17", "difficulty": "very_hard", "topic": "multi-policy",
        "question": "I am only buying a Phone Case for $14.99. How much will standard shipping cost me?",
        "ground_truth": "Since $14.99 is below the $50 free shipping threshold, standard shipping will cost $4.99.",
    },
    {
        "id": "RQ-18", "difficulty": "very_hard", "topic": "multi-policy",
        "question": "My order was delayed by 4 days and I also want to return it within the window. Can I get both the $5 store credit and a full refund?",
        "ground_truth": "Yes. The $5 delay credit is issued automatically for delays over 3 days. You can separately initiate a return for a refund within 30 days of delivery.",
    },
    {
        "id": "RQ-19", "difficulty": "very_hard", "topic": "products",
        "question": "What is the original price, the current sale price, and the exact dollar amount saved on the Mechanical Keyboard?",
        "ground_truth": "The Mechanical Keyboard was originally $149.00 and is now $120.00, saving exactly $29.00.",
    },
    {
        "id": "RQ-20", "difficulty": "very_hard", "topic": "orders",
        "question": "I placed an order 90 minutes ago. Can I still change it? And if I want to cancel, is that possible?",
        "ground_truth": "No — order changes are only allowed within 1 hour. Cancellation is still possible if the order has not yet been dispatched.",
    },
]


# ── Scoring ───────────────────────────────────────────────────────────────────

def _safe_float(text: str) -> float:
    try:
        return min(max(float(text.strip()), 0.0), 1.0)
    except Exception:
        return 0.5


def score_question(judge, question: str, answer: str, context: str, ground_truth: str):
    f_prompt = (
        "Rate from 0.0 to 1.0 how faithful this answer is to the retrieved context.\n"
        "1.0 = answer uses ONLY information present in the context.\n"
        "0.0 = answer contains information not found in the context at all.\n"
        "Reply with ONLY a decimal number between 0.0 and 1.0, nothing else.\n\n"
        f"Context:\n{context}\n\nAnswer:\n{answer}"
    )
    r_prompt = (
        "Rate from 0.0 to 1.0 how relevant this answer is to the question.\n"
        "1.0 = fully and directly answers the question.\n"
        "0.0 = completely irrelevant or misses the point entirely.\n"
        "Reply with ONLY a decimal number between 0.0 and 1.0, nothing else.\n\n"
        f"Question:\n{question}\n\nAnswer:\n{answer}"
    )
    p_prompt = (
        "Rate from 0.0 to 1.0 how precisely the retrieved context contains what is needed to answer correctly.\n"
        "1.0 = context directly and completely contains the correct answer.\n"
        "0.0 = context is entirely irrelevant to the question.\n"
        "Reply with ONLY a decimal number between 0.0 and 1.0, nothing else.\n\n"
        f"Question:\n{question}\n\nContext:\n{context}\n\nGround truth:\n{ground_truth}"
    )

    f = _safe_float(judge.invoke(f_prompt).content);  time.sleep(1)
    r = _safe_float(judge.invoke(r_prompt).content);  time.sleep(1)
    p = _safe_float(judge.invoke(p_prompt).content);  time.sleep(1)
    return round(f, 3), round(r, 3), round(p, 3)


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print("=" * 60)
    print("  Hybrid RAG Evaluation — 20 Questions")
    print("=" * 60)

    judge = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    hybrid_chain_fn, _ = get_hybrid_chain()

    results = []
    all_f, all_r, all_p = [], [], []
    diffs: dict = {}

    for i, item in enumerate(EVAL_QUESTIONS, 1):
        if i > 1:
            time.sleep(5)
        print(f"[{i:02d}/20] {item['id']} ({item['difficulty']:<9}) — {item['question'][:55]}...")

        # Use the proper hybrid chain (same LLM + prompt template as baseline)
        answer, docs = hybrid_chain_fn(item["question"])
        context = "\n\n".join(doc.page_content for doc in docs)
        time.sleep(1)

        f, r, p = score_question(judge, item["question"], answer, context, item["ground_truth"])
        all_f.append(f); all_r.append(r); all_p.append(p)
        print(f"           F={f:.3f}  AR={r:.3f}  CP={p:.3f}")

        results.append({
            "id":               item["id"],
            "difficulty":       item["difficulty"],
            "topic":            item["topic"],
            "question":         item["question"],
            "ground_truth":     item["ground_truth"],
            "answer":           answer,
            "faithfulness":     f,
            "answer_relevancy": r,
            "context_precision":p,
        })

        d = item["difficulty"]
        diffs.setdefault(d, {"faithfulness": [], "answer_relevancy": [], "context_precision": [], "n": 0})
        diffs[d]["faithfulness"].append(f)
        diffs[d]["answer_relevancy"].append(r)
        diffs[d]["context_precision"].append(p)
        diffs[d]["n"] += 1

    by_diff = {
        d: {
            "n":                vals["n"],
            "faithfulness":     round(statistics.mean(vals["faithfulness"]),     3),
            "answer_relevancy": round(statistics.mean(vals["answer_relevancy"]), 3),
            "context_precision":round(statistics.mean(vals["context_precision"]),3),
        }
        for d, vals in diffs.items()
    }

    output = {
        "run_name":  "Hybrid RAG (BM25 + Dense + RRF Fusion)",
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "faithfulness":      round(statistics.mean(all_f), 3),
            "answer_relevancy":  round(statistics.mean(all_r), 3),
            "context_precision": round(statistics.mean(all_p), 3),
            "total_questions":   len(results),
        },
        "by_difficulty": by_diff,
        "tests":         results,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("  HYBRID RAG RESULTS")
    print("=" * 60)
    print(f"  Faithfulness:      {output['summary']['faithfulness']:.3f}")
    print(f"  Answer Relevancy:  {output['summary']['answer_relevancy']:.3f}")
    print(f"  Context Precision: {output['summary']['context_precision']:.3f}")
    print(f"\n  Saved → {OUTPUT_PATH}")
    print("=" * 60)
    return output


if __name__ == "__main__":
    run()
