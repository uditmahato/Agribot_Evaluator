import argparse, pandas as pd
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="outputs/*_summary.csv")
    args = ap.parse_args()

    files = list(Path(".").glob(args.glob))
    if not files:
        print("No summary files found.")
        return

    dfs = [pd.read_csv(f) for f in files]
    total = pd.concat(dfs, ignore_index=True)
    print(total)
    print("\nOverall tuned win-rate:", round(total["tuned_win"].sum() / total["n"].sum(), 4))

if __name__ == "__main__":
    main()
