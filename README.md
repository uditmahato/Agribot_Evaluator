# AGRIBOT_EVALUATOR

LLM-as-judge evaluator for agronomy Q&A (pairwise A/B). Uses OpenAI API with `.env`.

## Setup
```bash
git clone <your-repo>
cd AGRIBOT_EVALUATOR
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
cp .env.example .env && nano .env   # paste your OPENAI_API_KEY
