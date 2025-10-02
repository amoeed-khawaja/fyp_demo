import random

# Simple mock ad recommender
ADS = {
    "sportswear": ["Nike Shoes", "Adidas Tracksuit"],
    "electronics": ["Smartphone Discount", "Laptop Offer"],
    "fashion": ["Designer Dress", "Casual Outfit Sale"],
    "food": ["Pizza Deal", "Coffee Shop Offer"]
}

def recommend_ad(customer_id, visits):
    """
    Very basic ad recommender:
    - Chooses ad category based on shop visits (if available)
    - Otherwise picks a random category
    """
    if visits:
        last_shop = visits[-1]["shop"]
        if last_shop in ADS:
            return random.choice(ADS[last_shop])
    # Fallback: random category
    category = random.choice(list(ADS.keys()))
    return random.choice(ADS[category])
