"""Recalculate saved CSV metrics; no model inference or training is performed."""
import csv
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"


def rows(name):
    with (RESULTS / name).open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def auc(labels, scores):
    ordered = sorted(zip(scores, labels))
    positive_ranks = 0.0
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][0] == ordered[start][0]:
            end += 1
        mean_rank = (start + 1 + end) / 2
        positive_ranks += mean_rank * sum(label for _, label in ordered[start:end])
        start = end
    positive = sum(labels)
    negative = len(labels) - positive
    return (positive_ranks - positive * (positive + 1) / 2) / (positive * negative)


def main():
    for filename, threshold, expected in [
        ("test_scores.csv", 0.45, (847, 100, 67, 880)),
        ("test_v2_scores.csv", 0.40, (833, 114, 206, 741)),
    ]:
        records = rows(filename)
        labels = [r["true_label"] == "fake" for r in records]
        scores = [float(r["fake_score"]) for r in records]
        predicted = [score >= threshold for score in scores]
        tn = sum(not y and not p for y, p in zip(labels, predicted))
        fp = sum(not y and p for y, p in zip(labels, predicted))
        fn = sum(y and not p for y, p in zip(labels, predicted))
        tp = sum(y and p for y, p in zip(labels, predicted))
        assert (tn, fp, fn, tp) == expected, filename
        print(f"{filename}: n={len(records)}, threshold={threshold:.2f}, "
              f"TN/FP/FN/TP={tn}/{fp}/{fn}/{tp}, "
              f"accuracy={(tn + tp) / len(records):.4%}, AUC={auc(labels, scores):.6f}")
    key = {r["file"]: r for r in rows("answer_key.csv")}
    for filename, threshold, expected in [
        ("results_detectnow_old.csv", 0.45, (40, 17, 9)),
        ("results_detectnow_v2.csv", 0.40, (46, 32, 2)),
    ]:
        records = rows(filename)
        assert len(records) == len(key) == 139
        assert {r["file"] for r in records} == set(key)
        counts = []
        for category in ("real", "face_swap", "ai_edit"):
            group = [r for r in records if key[r["file"]]["category"] == category]
            positive = sum(float(r["score"]) >= threshold for r in group)
            counts.append(len(group) - positive if category == "real" else positive)
            print(f"{filename}: {category}, n={len(group)}, fake calls={positive}")
        assert tuple(counts) == expected, filename
        print(f"Real/swap accuracy={(counts[0] + counts[1]) / 100:.0%}")


if __name__ == "__main__":
    main()
