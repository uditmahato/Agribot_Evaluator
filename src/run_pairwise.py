import os, json, argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from .utils import load_env, extract_json, ab_swap
from .judge import make_client, judge_pairwise


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROMPTS = ROOT / "prompts"
DATA = ROOT / "data"
OUT = ROOT / "outputs"

def read_text(p):
    return Path(p).read_text(encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DATA / "llm_eval_prompts_template.csv"))
    ap.add_argument("--pairwise_prompt", default=str(PROMPTS / "judge_pairwise.txt"))
    ap.add_argument("--out_prefix", default="judge_pairwise")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Reproducibility
    import random, numpy as np
    random.seed(args.seed); np.random.seed(args.seed)

    key, model = load_env()
    client = make_client(key)
    sys_prompt = read_text(args.pairwise_prompt)
    df = pd.read_csv(args.csv)

    OUT.mkdir(parents=True, exist_ok=True)
    raw_path = OUT / f"{args.out_prefix}_raw.jsonl"
    sum_path = OUT / f"{args.out_prefix}_summary.csv"
    lang_path = OUT / f"{args.out_prefix}_by_language.csv"

    wins = {"tuned": 0, "base": 0, "TIE": 0}
    by_lang = {}
    rows = []

    with raw_path.open("w", encoding="utf-8") as fout:
        for _, row in tqdm(df.iterrows(), total=len(df)):
            lang = row["language"]
            q = row["question"]
            base = row["base_answer"]
            tuned = row["tuned_answer"]

            ab_texts, mapping = ab_swap(base, tuned)
            A, B = ab_texts["A"], ab_texts["B"]

            content = judge_pairwise(client, model, sys_prompt, lang, q, A, B)

            try:
                data = extract_json(content)
            except Exception:
                # if judge output malformed, mark TIE with empty scores
                data = {"winner":"TIE", "parse_error": True}

            winner = data.get("winner", "TIE")
            if winner in ("A", "B"):
                winner = mapping[winner]  # map back to base/tuned

            wins[winner] = wins.get(winner, 0) + 1
            by_lang.setdefault(lang, {"tuned":0,"base":0,"TIE":0})
            by_lang[lang][winner] += 1

            rec = {
                "id": row.get("id", _),
                "language": lang,
                "winner": winner,
                "assignment": mapping,
                "raw": data
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            rows.append(rec)

    # Summary CSV
    n = len(df)
    summary = pd.DataFrame([{
        "n": n,
        "tuned_win": wins["tuned"],
        "base_win": wins["base"],
        "tie": wins["TIE"],
        "tuned_win_rate": round(wins["tuned"]/n, 4) if n else 0.0
    }])
    summary.to_csv(sum_path, index=False)

    # Per-language CSV
    lang_rows = []
    for l, c in by_lang.items():
        total = sum(c.values())
        lang_rows.append({
            "language": l,
            "n": total,
            "tuned_win": c["tuned"],
            "base_win": c["base"],
            "tie": c["TIE"],
            "tuned_win_rate": round(c["tuned"]/total, 4) if total else 0.0
        })
    pd.DataFrame(lang_rows).to_csv(lang_path, index=False)

    print("Saved:")
    print("  Raw:", raw_path)
    print("  Summary:", sum_path)
    print("  By-language:", lang_path)

if __name__ == "__main__":
    main()
