"""
data/ingest_datasets.py — Download and index external datasets into ChromaDB.

Sources:
  1. Amazon Product Reviews (Electronics) — McAuley-Lab/Amazon-Reviews-2023 on Hugging Face
  2. Bitext Customer Support dataset — bitext/Bitext-customer-support-llm-chatbot-training-dataset
  3. E-Commerce Support Style Guide — curated from common support ticket patterns (Kaggle-inspired)

Saves processed markdown to data/knowledge_base/, then rebuilds ChromaDB from all documents.

Usage:
  cd ecommerce-agent
  python data/ingest_datasets.py
"""

import os
import sys
import shutil
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)  # rag.py uses relative paths — must run from project root
sys.path.insert(0, os.path.join(ROOT, "src"))

KB_DIR = "data/knowledge_base"
os.makedirs(KB_DIR, exist_ok=True)


# ── 1. Amazon Electronics ─────────────────────────────────────────────────────

def ingest_amazon_electronics():
    """
    Load Amazon Electronics product metadata from Hugging Face.
    McAuley-Lab/Amazon-Reviews-2023 uses a loading script blocked by datasets v3+,
    so we discover the actual JSONL file paths via list_repo_files and load them
    directly as JSON, bypassing the loading script entirely.
    """
    print("\n[1/3] Loading Amazon Electronics metadata from Hugging Face...")
    try:
        from datasets import load_dataset
        from huggingface_hub import list_repo_files

        print("  Scanning repo for Electronics metadata files...")
        all_files = list(list_repo_files("McAuley-Lab/Amazon-Reviews-2023", repo_type="dataset"))
        meta_files = [
            f for f in all_files
            if "Electronics" in f
            and "meta" in f.lower()
            and not f.endswith(".py")
        ]

        if not meta_files:
            raise ValueError(
                f"No Electronics metadata files found. Repo contains: {all_files[:15]}"
            )

        # Use the first matching file — typically the full metadata JSONL
        chosen = meta_files[0]
        jsonl_hf_path = f"hf://datasets/McAuley-Lab/Amazon-Reviews-2023/{chosen}"
        print(f"  Streaming: {chosen}")

        ds = load_dataset(
            "json",
            data_files={"train": jsonl_hf_path},
            split="train",
            streaming=True,
        )

        products = []
        seen = set()
        for item in ds:
            title = (item.get("title") or "").strip()
            if not title or title in seen:
                continue
            seen.add(title)

            description = " ".join(item.get("description") or [])[:400].strip()
            features = " | ".join(item.get("features") or [])[:400].strip()
            price = str(item.get("price") or "").strip()
            categories = " > ".join(item.get("categories") or []).strip()

            if not description and not features:
                continue

            products.append({
                "title": title,
                "description": description,
                "features": features,
                "price": price,
                "categories": categories,
            })

            if len(products) >= 150:
                break

        if not products:
            raise ValueError("No products with descriptions found in stream.")

        chunk_size = 10
        file_idx = 1
        for i in range(0, len(products), chunk_size):
            batch = products[i : i + chunk_size]
            lines = [f"# Amazon Electronics Product Catalog — Batch {file_idx}\n"]
            for p in batch:
                lines.append(f"## {p['title']}")
                if p["categories"]:
                    lines.append(f"**Category:** {p['categories']}")
                if p["price"]:
                    lines.append(f"**Price:** {p['price']}")
                if p["description"]:
                    lines.append(f"**Description:** {p['description']}")
                if p["features"]:
                    lines.append(f"**Features:** {p['features']}")
                lines.append("")

            path = os.path.join(KB_DIR, f"amazon_electronics_{file_idx:02d}.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            file_idx += 1

        print(f"  Saved {len(products)} products across {file_idx - 1} files.")

    except Exception as e:
        print(f"  Could not load Amazon dataset ({e}). Writing curated fallback...")
        _amazon_fallback()


def _amazon_fallback():
    content = """\
# Extended Electronics Product Catalog — Amazon-Inspired

## Bluetooth Earbuds Pro
**Category:** Electronics > Audio > Earbuds
**Price:** $49.99
**Description:** True wireless earbuds with active noise cancellation, 8-hour battery per charge, and IPX5 water resistance. Charging case adds 24 extra hours.
**Features:** ANC | Touch Controls | USB-C | Bluetooth 5.2 | 6mm Drivers | Multipoint Connection

## Smart LED Strip 16.4ft
**Category:** Electronics > Smart Home > Lighting
**Price:** $24.99
**Description:** App-controlled RGB LED strip compatible with Alexa and Google Home. 16 million colors, music sync mode, cuttable every 2 inches.
**Features:** App + Voice Control | Music Sync | Self-Adhesive | 16M Colors | Scene Modes

## Portable Power Bank 20000mAh
**Category:** Electronics > Chargers > Portable
**Price:** $39.99
**Description:** 20,000mAh capacity with 22.5W PD fast charging. Charges a typical smartphone 4–5 times. Dual USB-A + USB-C output.
**Features:** 22.5W PD | Dual Output | LCD Indicator | 345g | Pass-Through Charging

## 15W Wireless Charging Pad
**Category:** Electronics > Chargers > Wireless
**Price:** $19.99
**Description:** Qi-certified 15W wireless charger for all Qi-enabled devices. Anti-slip surface, LED indicator, 5ft braided cable included.
**Features:** 15W Max | Qi Universal | Anti-Slip | Case Friendly | LED Status

## Laptop Cooling Pad 15.6\"
**Category:** Electronics > Laptop Accessories
**Price:** $27.99
**Description:** Dual 120mm fan cooling pad for laptops up to 15.6 inches. USB-powered, 5 adjustable height levels, RGB underglow lighting.
**Features:** Dual Fans | 5 Heights | RGB | USB Powered | 1200 RPM | Quiet 25dB

## HDMI 2.1 Cable 6ft
**Category:** Electronics > Cables
**Price:** $12.99
**Description:** 48Gbps HDMI 2.1 cable supporting 8K@60Hz and 4K@120Hz. Braided nylon sleeve, gold-plated connectors, backward compatible.
**Features:** 48Gbps | 8K@60Hz | 4K@120Hz | Braided | Gold Connectors | eARC

## USB-C to USB-C Cable 100W
**Category:** Electronics > Cables
**Price:** $14.99
**Description:** 100W PD fast-charging cable with 10Gbps data transfer. 6.6ft length, braided nylon, compatible with laptops, tablets, and phones.
**Features:** 100W PD | 10Gbps Data | 6.6ft | Braided Nylon | Universal Compatibility

## Mechanical Numpad
**Category:** Electronics > Peripherals > Keyboards
**Price:** $35.99
**Description:** Standalone 21-key mechanical numpad with Cherry MX Blue switches. RGB backlighting, USB-C, compatible with Windows and macOS.
**Features:** Cherry MX Blue | RGB | USB-C | 21-Key | Plug-and-Play | Aluminum Plate

## Ergonomic Mouse Vertical
**Category:** Electronics > Peripherals > Mice
**Price:** $32.99
**Description:** Vertical ergonomic wireless mouse reduces wrist strain by 57%. 2.4GHz connection, 6 buttons, DPI 800/1200/1600/2400, 18-month battery.
**Features:** Vertical Design | 2.4GHz | 6 Buttons | 4-Level DPI | 18-Month Battery

## Monitor Arm Single
**Category:** Electronics > Desk Accessories
**Price:** $44.99
**Description:** Fully adjustable single monitor arm for screens 17–32 inches up to 8kg. 360-degree rotation, VESA 75x75 and 100x100, cable management included.
**Features:** 17–32\" Support | 360° Rotation | VESA Compatible | C-Clamp + Grommet | Cable Management

## Mesh WiFi Router System
**Category:** Electronics > Networking
**Price:** $89.99
**Description:** Dual-band mesh WiFi 6 router covering up to 3,000 sq ft. AX1800 speeds, supports 40+ devices, easy app setup, automatic updates.
**Features:** WiFi 6 AX1800 | 3000 sq ft | 40+ Devices | App Setup | Auto Updates | MU-MIMO

## Smart Plug 4-Pack
**Category:** Electronics > Smart Home > Outlets
**Price:** $22.99
**Description:** WiFi smart plugs compatible with Alexa, Google Home, and Apple HomeKit. Schedule, timer, energy monitoring, no hub required.
**Features:** Alexa | Google | HomeKit | Energy Monitor | Scheduling | Compact Design
"""
    path = os.path.join(KB_DIR, "amazon_electronics_01.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("  Wrote curated Amazon Electronics fallback (12 products).")


# ── 2. Bitext Customer Support ─────────────────────────────────────────────────

def ingest_bitext_support():
    """
    Load Bitext Customer Support dataset from Hugging Face.
    load_dataset() crashes during the generation step on some systems, so we
    use hf_hub_download to get the raw CSV and read it with pandas directly.
    """
    print("\n[2/3] Loading Bitext Customer Support dataset from Hugging Face...")
    try:
        import pandas as pd
        from huggingface_hub import hf_hub_download

        print("  Downloading CSV via hf_hub_download...")
        csv_path = hf_hub_download(
            repo_id="bitext/Bitext-customer-support-llm-chatbot-training-dataset",
            filename="Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv",
            repo_type="dataset",
        )
        df = pd.read_csv(csv_path, encoding="utf-8")
        print(f"  Loaded {len(df):,} rows. Columns: {list(df.columns)}")

        # Normalise column names — dataset uses 'instruction' and 'response'
        q_col = next((c for c in df.columns if "instruction" in c.lower()), None)
        a_col = next((c for c in df.columns if "response" in c.lower()), None)
        i_col = next((c for c in df.columns if "intent" in c.lower()), None)

        if not q_col or not a_col:
            raise ValueError(f"Expected instruction/response columns, got: {list(df.columns)}")

        by_intent = defaultdict(list)
        for _, row in df.iterrows():
            intent = str(row[i_col]).strip() if i_col else "general"
            question = str(row[q_col]).strip()
            answer = str(row[a_col]).strip()
            if question and answer and len(by_intent[intent]) < 3:
                by_intent[intent].append({"q": question, "a": answer})

        category_map = {
            "order": [
                "check_invoice", "check_payment_methods", "check_refund_policy",
                "complaint", "contact_customer_service", "contact_human_agent",
                "get_invoice", "get_refund",
            ],
            "shipping": [
                "delivery_options", "delivery_period", "place_order",
                "track_order", "change_order", "cancel_order",
            ],
            "account": [
                "change_username", "change_password", "edit_account",
                "recover_password", "create_account", "delete_account",
            ],
            "returns": [
                "check_refund_policy", "get_refund", "return", "review",
            ],
        }

        saved = 0
        for category, intents in category_map.items():
            lines = [f"# Customer Support FAQ — {category.title()}\n"]
            for intent in intents:
                pairs = by_intent.get(intent, [])
                if not pairs:
                    continue
                lines.append(f"## {intent.replace('_', ' ').title()}")
                for pair in pairs[:2]:
                    lines.append(f"**Customer:** {pair['q']}")
                    lines.append(f"**Support:** {pair['a']}")
                    lines.append("")

            if len(lines) > 2:
                path = os.path.join(KB_DIR, f"bitext_faq_{category}.md")
                with open(path, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines))
                saved += 1
                print(f"  Saved bitext_faq_{category}.md")

        total = sum(len(v) for v in by_intent.values())
        print(f"  Processed {total} Q&A pairs across {len(by_intent)} intents, {saved} files written.")

    except Exception as e:
        print(f"  Could not load Bitext dataset ({e}). Writing curated fallback...")
        _bitext_fallback()


def _bitext_fallback():
    content = """\
# Customer Support FAQ — Orders, Shipping & Returns
## Bitext-Inspired E-Commerce Support Conversations

## Check Order Status
**Customer:** I need to know where my order is right now, I placed it 3 days ago.
**Support:** I'll look that up right away. Please share your order ID or the email on the account and I'll give you the latest tracking status and estimated delivery date.

**Customer:** My order still says processing — is that normal after two days?
**Support:** Orders normally process within 24 hours. Let me check what's happening. Please share your order ID and I'll investigate and update you immediately.

## Track Order
**Customer:** Can you send me the tracking information for my recent order?
**Support:** Of course! Once your order is dispatched, a tracking link is sent to your registered email. If you haven't received it, share your order ID and I'll resend it now.

## Cancel Order
**Customer:** I changed my mind — I want to cancel my order before it ships.
**Support:** I can help with that. Cancellations are possible as long as the order hasn't been dispatched yet. Please give me your order ID and I'll check the dispatch status immediately.

## Change Order
**Customer:** I ordered the wrong color — can I change it?
**Support:** Changes are possible within 1 hour of placing the order. Please contact us right away with your order ID and the change you'd like. After 1 hour, the order may already be in processing.

## Get Refund
**Customer:** I returned my item two weeks ago and haven't received my refund yet.
**Support:** I apologize for the delay. Refunds are normally issued within 5 business days of receiving your return. Please share your order ID and I'll escalate this for immediate review.

**Customer:** How do I get a refund for a damaged product?
**Support:** For damaged items, you're entitled to a full refund without needing to return the product. Please send a photo of the damage along with your order ID and we'll process the refund right away.

## Check Refund Policy
**Customer:** What is your refund policy?
**Support:** We accept returns within 30 days of delivery for unused items in original packaging. Refunds under $100 are processed automatically. Amounts between $100–$300 require supervisor approval, and above $300 require a human agent review.

## Delivery Options
**Customer:** What delivery options do you offer?
**Support:** We offer three options: Standard (5–7 business days, $4.99 or free above $50), Express (2–3 business days, $12.99), and Same-Day delivery in Cairo for orders placed before 12 PM.

## Delivery Period
**Customer:** How long will my order take to arrive?
**Support:** Standard shipping takes 5–7 business days. Express shipping arrives in 2–3 business days. For Same-Day delivery in Cairo, order before 12 PM. You'll receive a tracking link by email once your order ships.

## Check Payment Methods
**Customer:** What payment methods do you accept?
**Support:** We accept Visa, Mastercard, PayPal, and Cash on Delivery (available in Egypt only). All online transactions are secured with SSL encryption.

## Contact Human Agent
**Customer:** I need to speak to a real person about my issue.
**Support:** I completely understand. I'll connect you with a senior support specialist right away. I've documented your case so you won't need to repeat anything. A team member will follow up within 2 hours.

## Complaint
**Customer:** I'm really disappointed with the service — my order has been wrong twice.
**Support:** I sincerely apologize for this experience — this is not the standard we hold ourselves to. I'm escalating your case to our customer experience team right now. They will contact you within 24 hours with a resolution.

## Return
**Customer:** How do I return an item I don't want?
**Support:** To return an item: contact support with your order ID, we'll email you a prepaid return label within 24 hours, ship the item back within 7 days of receiving the label, and your refund will be issued within 5 business days of us receiving the return.

## Recover Password
**Customer:** I forgot my password and can't log in.
**Support:** No problem at all. Click "Forgot Password" on the login page and enter your email. We'll send a secure reset link immediately. The link expires in 24 hours for your security.

## Change Password
**Customer:** How do I update my account password?
**Support:** Log into your account, go to Account Settings > Security, and click "Change Password." You'll need to enter your current password and then set a new one. If you're locked out, use the "Forgot Password" option on the login page.

## Edit Account
**Customer:** I need to update my delivery address.
**Support:** You can save up to 5 delivery addresses in your account. Go to Account Settings > Addresses to add, edit, or remove addresses. Changes take effect immediately for future orders.
"""
    path = os.path.join(KB_DIR, "bitext_faq_general.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("  Wrote curated Bitext FAQ fallback.")


# ── 3. E-Commerce Support Style Guide (Kaggle-inspired tone calibration) ──────

def write_support_style_guide():
    print("\n[3/3] Writing e-commerce support style guide (Kaggle support ticket patterns)...")
    content = """\
# E-Commerce Customer Support Style Guide — NovaMart
## Tone Calibration Reference (Derived from Real Support Ticket Patterns)

## Core Voice Principles
- **Empathy first:** acknowledge the customer's situation before offering solutions.
- **Plain language:** avoid jargon; write at a 7th-grade reading level.
- **Proactive:** anticipate the next question and answer it in the same message.
- **Ownership:** use "I" and "we" — never blame systems, policies, or other departments.
- **Specificity:** give exact timelines, amounts, and steps — never vague promises.

## Response Templates by Scenario

### Delayed / Missing Order
"I completely understand your frustration — waiting longer than expected is unacceptable.
I've checked order [ID]: it's currently [status] and the latest update shows [detail].
[Explanation of cause if available]. Here's what happens next: [concrete next step].
If there's no update within [X hours/days], contact us again and we'll escalate immediately.
I've also applied a $5 store credit to your account for the inconvenience."

### Return / Refund Request
"I'd be happy to help process your return. Your item is eligible for a full refund under
our 30-day return policy. Here's exactly what to do:
1. Reply with your order ID and the reason for return.
2. We'll email a prepaid return label within 24 hours.
3. Drop off the item within 7 days.
4. Your refund will appear within 5 business days of us receiving it.
Is there anything else I can clarify?"

### Damaged or Wrong Item
"I'm really sorry your [product] arrived in that condition — that's not the experience
we want for you. You don't need to return it. Please reply with:
- A photo of the damage/wrong item
- Your order ID
Once I have those, I'll process a full refund or send a replacement immediately.
Which would you prefer?"

### Out-of-Policy Request
"I completely understand your frustration, and I genuinely wish I could do more.
Unfortunately, [policy constraint — stated plainly]. What I can offer instead is [alternative].
Would that work for you? If not, I can connect you with a senior specialist who has
more flexibility in these situations."

### Escalation Handoff
"This situation deserves more than I can offer here, so I'm connecting you with a senior
support specialist who can [specific action — e.g., approve exceptions, investigate billing].
I've documented everything: [summary]. You won't need to repeat yourself.
Expected response: within [timeframe]. Thank you for your patience."

### Guardrail / Unauthorized Request
"I appreciate you reaching out, but I'm not able to [unauthorized action] as it falls
outside what I'm authorized to do. What I can do is [alternative within policy].
If you believe there's been an error, I can escalate to a human specialist who can review
your case with full access. Would you like me to do that?"

## Phrase Reference

### Use These
- "I understand how frustrating this must be."
- "Let me look into this right away."
- "Here's exactly what will happen next."
- "I've made a note on your case."
- "Is there anything else I can help with today?"
- "I want to make sure this is resolved properly."

### Avoid These
- "As per our policy..." → say what you CAN do, not what policy prevents
- "I cannot do that." → replace with what you can offer
- "You should have..." → never blame the customer
- "That's not my department." → always own the problem
- "I don't know." → say "Let me find out for you right now."
- "It is what it is." → always offer a path forward

## Escalation Triggers (Always Pass to Human Agent)
- Refund requests above $300
- Customer mentions legal action or regulatory complaints
- Account security concerns (unauthorized access, fraud)
- Three or more repeated contacts about the same unresolved issue
- Aggressive or threatening language (de-escalate first, then escalate)
- Any request involving another customer's personal data
"""
    path = os.path.join(KB_DIR, "support_style_guide.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("  Saved support_style_guide.md")


# ── 4. Rebuild ChromaDB ───────────────────────────────────────────────────────

def rebuild_vectorstore():
    print("\n[Rebuilding ChromaDB with all documents...]")
    from dotenv import load_dotenv
    load_dotenv(".env")

    from rag import get_embeddings, CHROMA_PATH, POLICIES_DIR, KNOWLEDGE_BASE_DIR
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    all_docs = []
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
                all_docs.append(Document(
                    page_content=text,
                    metadata={"source": fname, "directory": label},
                ))

    print(f"  Loaded {len(all_docs)} documents ({POLICIES_DIR} + {KNOWLEDGE_BASE_DIR}).")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(all_docs)
    print(f"  Split into {len(chunks)} chunks.")

    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("  Cleared old ChromaDB.")

    print("  Embedding and storing... (this may take a few minutes)")
    Chroma.from_documents(chunks, get_embeddings(), persist_directory=CHROMA_PATH)
    print(f"  ChromaDB rebuilt successfully with {len(chunks)} chunks.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  NovaMart Dataset Ingestion Pipeline")
    print("=" * 60)

    ingest_amazon_electronics()
    ingest_bitext_support()
    write_support_style_guide()
    rebuild_vectorstore()

    print("\n" + "=" * 60)
    print("  All datasets ingested. ChromaDB rebuilt.")
    print("  Run the app or eval suite to verify.")
    print("=" * 60)
