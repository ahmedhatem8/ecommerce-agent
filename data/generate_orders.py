import json
import random
from faker import Faker
from datetime import datetime, timedelta

fake = Faker()
random.seed(42)

STATUSES = ["delivered", "in_transit", "delayed", "returned", "cancelled"]
ITEMS = [
    {"name": "Wireless Headphones", "price": 89.99},
    {"name": "Phone Case", "price": 14.99},
    {"name": "USB-C Hub", "price": 34.99},
    {"name": "Laptop Stand", "price": 45.00},
    {"name": "Mechanical Keyboard", "price": 120.00},
    {"name": "Webcam HD", "price": 59.99},
    {"name": "Desk Lamp", "price": 29.99},
    {"name": "Monitor 27 inch", "price": 299.99},
]

def random_date(days_back=90):
    d = datetime.now() - timedelta(days=random.randint(0, days_back))
    return d.strftime("%Y-%m-%d")

orders = []
for i in range(1, 201):
    item = random.choice(ITEMS)
    qty = random.randint(1, 3)
    status = random.choices(
        STATUSES, weights=[50, 25, 10, 10, 5]
    )[0]
    order = {
        "order_id": f"ORD-{1000 + i}",
        "customer_id": f"CUST-{random.randint(100, 150)}",
        "customer_name": fake.name(),
        "email": fake.email(),
        "item": item["name"],
        "quantity": qty,
        "total_price": round(item["price"] * qty, 2),
        "status": status,
        "order_date": random_date(90),
        "estimated_delivery": random_date(30) if status == "in_transit" else None,
    }
    orders.append(order)

with open("data/orders.json", "w") as f:
    json.dump(orders, f, indent=2)

print(f"Generated {len(orders)} orders.")