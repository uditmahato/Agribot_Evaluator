import os, json, random
from dotenv import load_dotenv

def load_env():
    load_dotenv()
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY missing. Create .env from .env.example.")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return key, model

def extract_json(text: str):
    """Robustly extract first {...} JSON object from text."""
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end+1])
        raise

def ab_swap(a_text: str, b_text: str):
    """Randomly assign which model becomes Answer A vs B."""
    if random.random() < 0.5:
        return {"A": a_text, "B": b_text}, {"A": "base", "B": "tuned"}
    return {"A": b_text, "B": a_text}, {"A": "tuned", "B": "base"}
