# AGRIBOT_EVALUATOR

LLM-as-judge evaluator for agronomy Q&A (pairwise A/B). Uses OpenAI API with `.env`.

## Setup
```bash
git clone <your-repo>
cd AGRIBOT_EVALUATOR
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
cp .env && nano .env   # paste your OPENAI_API_KEY

python -m src.run_pairwise \
  --csv data/llm_eval_prompts_template.csv \
  --pairwise_prompt prompts/judge_pairwise.txt \
  --out_prefix judge_pairwise_run1
