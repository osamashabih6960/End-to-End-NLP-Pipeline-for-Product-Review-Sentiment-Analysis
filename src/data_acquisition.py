"""
data_acquisition.py
--------------------
Step 1: DATA ACQUISITION

In a real production system this module would pull data from one (or a mix) of:
    - Internal DB / data warehouse (order + review tables)
    - Public datasets (Amazon/Flipkart reviews, Kaggle)
    - Web scraping (Scrapy/BeautifulSoup) subject to ToS
    - Third-party APIs (Play Store / App Store review APIs)
    - Human annotation / crowdsourcing for labelling raw text

Since this project needs to run fully offline & reproducibly, this script
GENERATES a realistic, labelled e-commerce product-review dataset using
templated natural language + controlled random noise (typos, punctuation,
emojis, varied length). This mimics the *shape* of a real acquired dataset
(class imbalance, noisy text, short + long reviews) closely enough to train
and demo a full pipeline end-to-end.

Swap this file out for a real DB/API/scraper call in production -
every downstream module only depends on the resulting CSV schema:
    review_id, text, rating, sentiment
"""

import random
import csv
import os

random.seed(42)

POSITIVE_TEMPLATES = [
    "I absolutely love this {product}, it works perfectly and {reason}.",
    "This {product} exceeded my expectations, {reason}.",
    "Best {product} I have ever bought, {reason}!",
    "Highly recommend this {product}, {reason}.",
    "The {product} is amazing, {reason}.",
    "Great value for money, this {product} {reason}.",
    "Super happy with this purchase, the {product} {reason}.",
    "Five stars! The {product} {reason}.",
    "Works like a charm, {reason}.",
    "Fantastic quality {product}, {reason} :)",
]

NEGATIVE_TEMPLATES = [
    "This {product} is terrible, {reason}.",
    "Very disappointed with the {product}, {reason}.",
    "Waste of money, the {product} {reason}.",
    "Do not buy this {product}, {reason}.",
    "Worst {product} ever, {reason}.",
    "The {product} broke within days, {reason}.",
    "Poor quality {product}, {reason}.",
    "I regret buying this {product}, {reason}.",
    "Completely useless, {reason}.",
    "Not worth it at all, {reason} :(",
]

NEUTRAL_TEMPLATES = [
    "The {product} is okay, nothing special, {reason}.",
    "It's an average {product}, {reason}.",
    "The {product} does what it says, {reason}.",
    "Not bad, not great, {reason}.",
    "It arrived on time, the {product} {reason}.",
    "Decent {product} for the price, {reason}.",
    "The {product} is fine I guess, {reason}.",
    "Standard {product}, {reason}.",
    "It works, but {reason}.",
    "Neutral experience overall, {reason}.",
]

PRODUCTS = [
    "phone case", "laptop", "wireless earbuds", "kitchen blender", "backpack",
    "office chair", "smart watch", "bluetooth speaker", "running shoes",
    "coffee maker", "desk lamp", "keyboard", "monitor", "book", "t-shirt",
    "camera", "tablet", "charger", "gaming mouse", "water bottle",
]

POS_REASONS = [
    "the build quality is excellent", "delivery was super fast",
    "customer service was very helpful", "battery life is outstanding",
    "it looks even better in person", "the packaging was premium",
    "it fits perfectly", "the sound quality is crisp", "it saved me so much time",
    "the price was very reasonable for the quality",
]

NEG_REASONS = [
    "it stopped working after a week", "the packaging was damaged",
    "customer support never responded", "the material feels cheap",
    "it did not match the description", "the size was completely wrong",
    "it arrived very late", "the battery drains too fast",
    "there were missing parts", "it is overpriced for what you get",
]

NEU_REASONS = [
    "the shipping took a bit longer than expected", "it looks similar to the pictures",
    "the instructions were a little unclear", "it does the basic job",
    "there is nothing extra to mention", "the price is about average",
    "the colour was slightly different than shown", "it is comparable to other brands",
    "some features were unnecessary", "it works as described, nothing more",
]

EMOJIS_POS = ["😊", "🔥", "👍", "❤️", ""]
EMOJIS_NEG = ["😡", "👎", "😢", ""]
TYPOS = {"the": "teh", "really": "realy", "product": "produtc", "great": "gr8"}


def _inject_noise(text, p=0.12):
    words = text.split()
    for i, w in enumerate(words):
        key = w.lower().strip(".,!")
        if key in TYPOS and random.random() < p:
            words[i] = TYPOS[key]
    return " ".join(words)


def _make_review(templates, reasons, product, emoji_pool):
    text = random.choice(templates).format(
        product=product, reason=random.choice(reasons)
    )
    if random.random() < 0.3:
        text += " " + random.choice(emoji_pool)
    if random.random() < 0.15:
        text = text.upper() if random.random() < 0.3 else text
    text = _injectnoise_wrapper(text)
    return text.strip()


def _injectnoise_wrapper(text):
    return _inject_noise(text)


def generate_dataset(n_per_class=400):
    rows = []
    idx = 1
    for _ in range(n_per_class):
        product = random.choice(PRODUCTS)
        text = _make_review(POSITIVE_TEMPLATES, POS_REASONS, product, EMOJIS_POS)
        rows.append((idx, text, random.choice([4, 5]), "positive")); idx += 1

        product = random.choice(PRODUCTS)
        text = _make_review(NEGATIVE_TEMPLATES, NEG_REASONS, product, EMOJIS_NEG)
        rows.append((idx, text, random.choice([1, 2]), "negative")); idx += 1

        product = random.choice(PRODUCTS)
        text = _make_review(NEUTRAL_TEMPLATES, NEU_REASONS, product, [""])
        rows.append((idx, text, 3, "neutral")); idx += 1

    random.shuffle(rows)
    return rows


def acquire_data(output_path="data/reviews.csv", n_per_class=400):
    rows = generate_dataset(n_per_class=n_per_class)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["review_id", "text", "rating", "sentiment"])
        writer.writerows(rows)
    print(f"[data_acquisition] Wrote {len(rows)} labelled reviews -> {output_path}")
    return output_path


if __name__ == "__main__":
    acquire_data()
