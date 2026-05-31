import os
os.makedirs("data/policies", exist_ok=True)

policies = {
    "return_policy.md": """# Return & Refund Policy — NovaMart

## Eligibility
- Items may be returned within 30 days of delivery.
- Items must be unused, in original packaging, and include the receipt.
- Digital products and opened software are non-refundable.

## Refund Process
- Refunds under $100 are processed automatically by our support agent.
- Refunds between $100 and $300 require supervisor approval (escalation required).
- Refunds above $300 require a human agent review — the AI agent MUST escalate.

## How to Return
1. Contact support with your order ID.
2. Receive a return shipping label by email within 24 hours.
3. Ship the item back within 7 days of receiving the label.
4. Refund is issued within 5 business days of receiving the return.

## Exceptions
- Damaged items: full refund with photo evidence, no return needed.
- Wrong item sent: full refund or replacement, no return needed.
""",

    "shipping_policy.md": """# Shipping Policy — NovaMart

## Delivery Times
- Standard shipping: 5-7 business days.
- Express shipping: 2-3 business days.
- Same-day delivery: available in Cairo only for orders before 12 PM.

## Shipping Fees
- Free standard shipping on orders above $50.
- Standard shipping fee: $4.99 for orders below $50.
- Express shipping fee: $12.99.

## Tracking
- A tracking link is emailed once your order is dispatched.
- Tracking updates every 24 hours.

## Delays
- If your order is delayed beyond the estimated date, contact support with your order ID.
- We will investigate and provide an update within 12 hours.
- Compensation: a $5 store credit is issued automatically for delays over 3 days.
""",

    "faq.md": """# Frequently Asked Questions — NovaMart

## Orders
Q: Can I change my order after placing it?
A: Yes, within 1 hour of placing the order. Contact support immediately with your order ID.

Q: Can I cancel my order?
A: Yes, if the order is not yet dispatched. Once in transit, cancellation is not possible.

Q: What payment methods do you accept?
A: Visa, Mastercard, PayPal, and cash on delivery (Egypt only).

## Account
Q: How do I reset my password?
A: Click Forgot password on the login page. A reset link is sent to your email.

Q: Can I have multiple delivery addresses?
A: Yes, you can save up to 5 addresses in your account settings.

## Products
Q: Are your products under warranty?
A: Electronics carry a 1-year manufacturer warranty. Other items have a 90-day warranty.

Q: Do you sell internationally?
A: Currently we ship within Egypt and the UAE only.
""",
}

for filename, content in policies.items():
    path = f"data/policies/{filename}"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Created {path}")

print("All policy documents created.")