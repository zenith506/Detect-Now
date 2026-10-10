import csv
import sys
from collections import defaultdict


def read(path):
    with open(path, newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    key = {r["file"]: r for r in read(sys.argv[1])}
    verdicts = {}
    for row in read(sys.argv[2]):
        value = (row.get("verdict") or "").strip().lower()
        if value in ("real", "fake"):
            verdicts[row["file"].strip()] = value

    counts = defaultdict(lambda: [0, 0])  # category -> [correct, answered]
    unanswered = 0
    for name, info in key.items():
        if name not in verdicts:
            unanswered += 1
            continue
        c = counts[info["category"]]
        c[1] += 1
        c[0] += verdicts[name] == info["true_label"]

    labels = {
        "real": "Real images called real",
        "face_swap": "Face swaps caught",
        "ai_edit": "AI edits caught",
    }
    print(f"{'Category':28s} {'correct':>10s} {'rate':>8s}")
    print("-" * 48)
    for cat in ("real", "face_swap", "ai_edit"):
        ok, n = counts[cat]
        rate = f"{ok / n:.1%}" if n else "-"
        print(f"{labels[cat]:28s} {ok:>4d}/{n:<5d} {rate:>8s}")

    total_ok = sum(c[0] for c in counts.values())
    total_n = sum(c[1] for c in counts.values())
    real_ok, real_n = counts["real"]
    fake_ok = total_ok - real_ok
    fake_n = total_n - real_n
    if real_n and fake_n:
        print("-" * 48)
        print(f"{'Overall accuracy':28s} {total_ok:>4d}/{total_n:<5d} {total_ok / total_n:>8.1%}")
        balanced = (real_ok / real_n + fake_ok / fake_n) / 2
        print(f"{'Balanced accuracy':28s} {'':>10s} {balanced:>8.1%}")
    if unanswered:
        print(f"\nNo answer for {unanswered} images.")


if __name__ == "__main__":
    main()
