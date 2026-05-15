"""
prompts.py — Standardized prompt bank (40 prompts across 5 categories)

Each prompt has:
  id          — unique slug used as a filename-safe key
  category    — factual | reasoning | code | creative | edge
  text        — the raw user prompt
  expected_json_keys (optional) — for Phase 2 schema validation smoke-tests
"""

PROMPTS: list[dict] = [
    # ── FACTUAL (8) ──────────────────────────────────────────────────────────
    {"id": "f01", "category": "factual",
     "text": "What is the capital of Australia?"},
    {"id": "f02", "category": "factual",
     "text": "In what year did the Berlin Wall fall?"},
    {"id": "f03", "category": "factual",
     "text": "What is the speed of light in a vacuum (m/s)?"},
    {"id": "f04", "category": "factual",
     "text": "Name the four fundamental forces of nature."},
    {"id": "f05", "category": "factual",
     "text": "What programming language was Python named after?"},
    {"id": "f06", "category": "factual",
     "text": "How many bones are in the adult human body?"},
    {"id": "f07", "category": "factual",
     "text": "What does HTTP stand for?"},
    {"id": "f08", "category": "factual",
     "text": "Who wrote 'Pride and Prejudice'?"},

    # ── REASONING (8) ────────────────────────────────────────────────────────
    {"id": "r01", "category": "reasoning",
     "text": "If a train travels 120 km in 1.5 hours, what is its average speed in km/h?"},
    {"id": "r02", "category": "reasoning",
     "text": "A bat and a ball cost $1.10 in total. The bat costs $1 more than the ball. How much does the ball cost?"},
    {"id": "r03", "category": "reasoning",
     "text": "All roses are flowers. Some flowers fade quickly. Can we conclude that some roses fade quickly? Explain."},
    {"id": "r04", "category": "reasoning",
     "text": "You have a 3-litre jug and a 5-litre jug. How do you measure exactly 4 litres of water?"},
    {"id": "r05", "category": "reasoning",
     "text": "Rank these sorting algorithms by worst-case time complexity: Bubble Sort, Merge Sort, Quick Sort."},
    {"id": "r06", "category": "reasoning",
     "text": "If you flip a fair coin 3 times, what is the probability of getting exactly 2 heads?"},
    {"id": "r07", "category": "reasoning",
     "text": "A store reduces a $200 jacket by 30%, then later raises the sale price by 20%. What is the final price?"},
    {"id": "r08", "category": "reasoning",
     "text": "Explain why 0.1 + 0.2 ≠ 0.3 in most floating-point systems."},

    # ── CODE (8) ─────────────────────────────────────────────────────────────
    {"id": "c01", "category": "code",
     "text": "Write a Python function that checks if a string is a palindrome."},
    {"id": "c02", "category": "code",
     "text": "Write a SQL query to find the top 5 customers by total spend from an 'orders' table with columns customer_id and amount."},
    {"id": "c03", "category": "code",
     "text": "Implement a binary search function in Python with O(log n) complexity."},
    {"id": "c04", "category": "code",
     "text": "Write a Python decorator that logs function execution time."},
    {"id": "c05", "category": "code",
     "text": "Given a list of integers, return all pairs that sum to a target value using O(n) time."},
    {"id": "c06", "category": "code",
     "text": "Write a bash one-liner to count the number of unique IP addresses in a log file."},
    {"id": "c07", "category": "code",
     "text": "Implement an LRU cache in Python using built-in data structures."},
    {"id": "c08", "category": "code",
     "text": "Write a regex pattern to validate an email address, with a short explanation of each part."},

    # ── CREATIVE (8) ─────────────────────────────────────────────────────────
    {"id": "cr01", "category": "creative",
     "text": "Write a three-sentence product description for a smart water bottle that tracks hydration."},
    {"id": "cr02", "category": "creative",
     "text": "Create a short analogy explaining recursion to a 10-year-old."},
    {"id": "cr03", "category": "creative",
     "text": "Write a haiku about machine learning."},
    {"id": "cr04", "category": "creative",
     "text": "Give the opening line of a thriller novel set on the International Space Station."},
    {"id": "cr05", "category": "creative",
     "text": "Suggest three unique names for a coffee shop that specialises in cold brew."},
    {"id": "cr06", "category": "creative",
     "text": "Describe a colour to someone who has been blind from birth."},
    {"id": "cr07", "category": "creative",
     "text": "Write a one-paragraph pitch for a mobile app that gamifies learning chess."},
    {"id": "cr08", "category": "creative",
     "text": "Invent a proverb about debugging code and explain its meaning."},

    # ── EDGE / STRESS (8) ────────────────────────────────────────────────────
    {"id": "e01", "category": "edge",
     "text": "Translate 'The quick brown fox jumps over the lazy dog' into exactly three languages."},
    {"id": "e02", "category": "edge",
     "text": "List the prime numbers between 1 and 50."},
    {"id": "e03", "category": "edge",
     "text": "What is 17 × 23?"},
    {"id": "e04", "category": "edge",
     "text": "Continue this sequence: 2, 3, 5, 8, 13, 21, … (next three values)."},
    {"id": "e05", "category": "edge",
     "text": "Summarise the plot of 'The Great Gatsby' in exactly two sentences."},
    {"id": "e06", "category": "edge",
     "text": "Respond only with 'YES' or 'NO': Is the Earth older than the Sun?"},
    {"id": "e07", "category": "edge",
     "text": "Give me a word that rhymes with 'orange'."},
    {"id": "e08", "category": "edge",
     "text": "Write the NATO phonetic alphabet in order."},
]

# Quick look-up by id
PROMPT_BY_ID: dict[str, dict] = {p["id"]: p for p in PROMPTS}

# Category groupings
CATEGORIES: list[str] = sorted({p["category"] for p in PROMPTS})
