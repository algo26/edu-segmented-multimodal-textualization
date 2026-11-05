"""Compute the statistics reported in Tables 5, 6 and 7 of the manuscript.

Table 5  95% percentile bootstrap confidence intervals (10,000 resamples) for each
         configuration, and paired two-sided tests between configurations:
         exact McNemar tests on TR and NQA, permutation tests with 100,000 random
         sign flips on VS, and Holm-adjusted p-values over the five comparisons
         of each subtask.
Table 6  Improvement of full fusion (C1) over each single-channel configuration
         (C2, C3), in percentage points (pp) and in relative terms.
Table 7  Accuracy of C4 under alternative duration thresholds (tau1, tau2) of the
         frame sampling rule, and exact McNemar tests against the nominal C4.

Input: one CSV file in long format with one row per configuration and item:

    subtask,item_id,config,value
    TR,test_AWB-6:topic_reasoning:in-what-environment...,C1,1
    NQA,test_needle_0:needle_qa:how-many-bats...,C4,0
    VS,test_subPlot_3,C2,7

    subtask  TR (Topic Reasoning), NQA (Needle Question Answering) or VS (Video
             Summarization). The long names topic_reasoning, needle_qa and
             video_summarization are also accepted.
    item_id  question or video identifier; it pairs the configurations item by item.
    config   C1 to C5, and C4_6_12 and C4_10_20 for the threshold variants of C4.
    value    1 or 0 (answer correct or not) for TR and NQA; the GPT-4 judge score
             for VS (completeness plus reliability, from 0 to 10).

Usage:
    python statistics_tables.py results_long.csv [--out-dir results]

Every random procedure uses its own generator seeded with --seed (default 2026),
so the output does not depend on the order in which the comparisons are run.
Requirements: numpy, scipy.
"""
import argparse
import csv
import os
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest

SUBTASK_ALIASES = {"TR": "TR", "topic_reasoning": "TR",
                   "NQA": "NQA", "needle_qa": "NQA",
                   "VS": "VS", "video_summarization": "VS"}
SUBTASKS = ["TR", "NQA", "VS"]
CONFIGS = ["C1", "C2", "C3", "C4", "C5"]
COMPARISONS = [("C1", "C2"), ("C1", "C3"), ("C1", "C4"), ("C1", "C5"), ("C4", "C5")]
THRESHOLD_VARIANTS = [("C4_6_12", "(6 s, 12 s)"), ("C4", "(8 s, 16 s), nominal"), ("C4_10_20", "(10 s, 20 s)")]


def load(path):
    """Return data[subtask][config] = {item_id: value}."""
    data = defaultdict(lambda: defaultdict(dict))
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            subtask = SUBTASK_ALIASES[row["subtask"].strip()]
            item, config = row["item_id"].strip(), row["config"].strip()
            if item in data[subtask][config]:
                raise ValueError(f"duplicated item {item} for {config} in {subtask}")
            data[subtask][config][item] = float(row["value"])
    return data


def paired(data, subtask, a, b):
    """Values of configurations a and b on the items they share, in the same order."""
    da, db = data[subtask][a], data[subtask][b]
    if set(da) != set(db):
        raise ValueError(f"{subtask}: {a} and {b} do not cover the same items")
    items = sorted(da)
    return np.array([da[i] for i in items]), np.array([db[i] for i in items])


def bootstrap_ci(values, n_resamples=10_000, seed=2026, level=0.95):
    """Percentile bootstrap interval of the mean, resampling items with replacement."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_resamples, len(values)))
    means = values[idx].mean(axis=1)
    alpha = (1 - level) / 2
    return np.percentile(means, [100 * alpha, 100 * (1 - alpha)])


def mcnemar_exact(x, y):
    """Exact two-sided McNemar test for paired binary outcomes.

    b counts the items that x answers correctly and y does not, c the reverse.
    Under the null hypothesis each discordant item is equally likely to fall in
    either group, so b follows a Binomial(b + c, 0.5) distribution.
    """
    x, y = np.asarray(x).astype(bool), np.asarray(y).astype(bool)
    b, c = int(np.sum(x & ~y)), int(np.sum(~x & y))
    p = 1.0 if b + c == 0 else binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue
    return b, c, min(p, 1.0)


def permutation_test(x, y, n_permutations=100_000, seed=2026):
    """Two-sided paired permutation test on the mean difference (random sign flips)."""
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    observed = abs(d.mean())
    rng = np.random.default_rng(seed)
    count = 0
    for start in range(0, n_permutations, 10_000):  # blocks keep memory low
        n = min(10_000, n_permutations - start)
        signs = rng.choice([-1.0, 1.0], size=(n, d.size))
        null = np.abs((signs * d).mean(axis=1))
        count += int(np.sum(null >= observed - 1e-12))
    return (count + 1) / (n_permutations + 1)


def holm(pvalues):
    """Holm step-down adjustment; returns adjusted p-values in the input order."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adjusted[i] = min(running, 1.0)
    return adjusted


def fmt_p(p):
    return "$<$0.0001" if p < 0.0001 else f"{p:.4f}"


def table5(data, seed, out):
    rows_ci, rows_tests = [], []
    for subtask in SUBTASKS:
        for config in CONFIGS:
            vals = np.array(list(data[subtask][config].values()))
            if vals.size == 0:
                continue
            lo, hi = bootstrap_ci(vals, seed=seed)
            scale = 1 if subtask == "VS" else 100
            rows_ci.append((subtask, config, len(vals), scale * vals.mean(), scale * lo, scale * hi))
        family = []
        for a, b in COMPARISONS:
            if not (data[subtask][a] and data[subtask][b]):
                continue
            x, y = paired(data, subtask, a, b)
            if subtask == "VS":
                family.append([subtask, f"{a} vs {b}", "", "", "paired permutation", permutation_test(x, y, seed=seed)])
            else:
                bb, cc, p = mcnemar_exact(x, y)
                family.append([subtask, f"{a} vs {b}", bb, cc, "exact McNemar", p])
        if family:
            for row, adj in zip(family, holm([r[-1] for r in family])):
                rows_tests.append(row + [adj])

    with open(os.path.join(out, "table5_intervals.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subtask", "config", "n", "estimate", "ci_low", "ci_high"])
        w.writerows([[s, c, n, f"{e:.2f}", f"{lo:.2f}", f"{hi:.2f}"] for s, c, n, e, lo, hi in rows_ci])
    with open(os.path.join(out, "table5_tests.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subtask", "comparison", "b", "c", "test", "p", "p_holm"])
        w.writerows([[s, comp, b, c, t, f"{p:.6f}", f"{q:.6f}"] for s, comp, b, c, t, p, q in rows_tests])

    print("Table 5. Confidence intervals")
    for s, c, n, e, lo, hi in rows_ci:
        unit = "" if s == "VS" else "%"
        print(f"  {s:3s} {c}: {e:.2f}{unit} [{lo:.2f}, {hi:.2f}]  (N = {n})")
    print("Table 5. Paired tests, p (Holm-adjusted p)")
    for s, comp, b, c, t, p, q in rows_tests:
        extra = f"b = {b}, c = {c}, " if t == "exact McNemar" else ""
        print(f"  {s:3s} {comp}: {extra}{fmt_p(p)} ({fmt_p(q)})")


def table6(data, out):
    rows = []
    for subtask in SUBTASKS:
        means = {c: np.mean(list(data[subtask][c].values())) for c in ("C1", "C2", "C3") if data[subtask][c]}
        if len(means) < 3:
            continue
        scale = 1 if subtask == "VS" else 100
        for single in ("C2", "C3"):
            gain = scale * (means["C1"] - means[single])
            rel = 100 * (means["C1"] - means[single]) / means[single]
            rows.append((subtask, single, scale * means[single], scale * means["C1"], gain, rel))
    with open(os.path.join(out, "table6_modality_gain.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subtask", "single_channel", "single_score", "c1_score", "absolute_gain", "relative_gain_percent"])
        w.writerows([[s, c, f"{a:.2f}", f"{b:.2f}", f"{g:.2f}", f"{r:.2f}"] for s, c, a, b, g, r in rows])
    print("Table 6. Improvement of C1 over the single-channel configurations")
    for s, c, a, b, g, r in rows:
        unit = "" if s == "VS" else " pp"
        print(f"  {s:3s} C1 over {c}: {g:+.2f}{unit} ({r:+.2f}%)")


def table7(data, out):
    rows = []
    for subtask in ("TR", "NQA"):
        if not data[subtask]["C4"]:
            continue
        for config, label in THRESHOLD_VARIANTS:
            if not data[subtask][config]:
                continue
            vals = np.array(list(data[subtask][config].values()))
            if config == "C4":
                rows.append((subtask, label, 100 * vals.mean(), "", "", None))
            else:
                b, c, p = mcnemar_exact(*paired(data, subtask, config, "C4"))
                rows.append((subtask, label, 100 * vals.mean(), b, c, p))
    if not rows:
        return
    with open(os.path.join(out, "table7_threshold_tests.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subtask", "schedule", "accuracy", "b_variant_only", "c_nominal_only", "p_mcnemar"])
        w.writerows([[s, l, f"{a:.2f}", b, c, "" if p is None else f"{p:.6f}"] for s, l, a, b, c, p in rows])
    print("Table 7. Threshold variants of C4 (exact McNemar against the nominal schedule)")
    for s, l, a, b, c, p in rows:
        tail = "" if p is None else f"  b = {b}, c = {c}, p = {fmt_p(p)}"
        print(f"  {s:3s} {l}: {a:.2f}%{tail}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("results", help="CSV file in long format (see the module docstring)")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    data = load(args.results)
    table5(data, args.seed, args.out_dir)
    table6(data, args.out_dir)
    table7(data, args.out_dir)
    print(f"CSV files written to {args.out_dir}")


if __name__ == "__main__":
    main()
