"""Statistics for the MedQA multi-agent results (read-only, Python stdlib only).

Usage (from the analysis/ folder, next to summarize.py):
    python3 stats.py ../results qwen4 qwen8 qwen14 qwen32 gemma12 llama8

Per model it prints:
  1. vote accuracy with bootstrap 95% CIs (resampling questions)
  2. exact McNemar tests on paired votes (r1 vs r2, SC-6 vs r2, CoT vs r2, r2 vs planted)
  3. unanimous precision r1 vs r2 with bootstrap CI of the difference
  4. r1 -> r2 state transitions (unanimous-right / unanimous-wrong / split)
  5. AUROC of disagreement as a predictor of a wrong vote (+ bootstrap CI), and r2 - r1
  6. risk-coverage: answer only when agreement >= level
Also writes <results>/risk_coverage.csv for plotting later.
Votes use the same rule as summarize.py (majority, tie -> earliest agent/sample).
"""
import csv
import json
import random
import sys
from collections import Counter
from math import factorial


def comb(n, k):  # math.comb needs Python 3.8+; this works on older versions too
    return factorial(n) // (factorial(k) * factorial(n - k))
from pathlib import Path

from summarize import group, load, vote

B = 2000  # bootstrap resamples
SEED = 0


# ---------- helpers ----------
def agree_frac(v):
    """Fraction of agents/samples that gave the voted answer (1.0 = unanimous)."""
    w = vote(v)
    return 0.0 if w is None else sum(p == w for p in v) / len(v)


def no_majority(v):
    """True when the vote had to use the tie rule (top count shared)."""
    c = Counter(p for p in v if p)
    if not c:
        return True
    top = sorted(c.values(), reverse=True)
    return len(top) > 1 and top[0] == top[1]


def mcnemar(a_ok, b_ok, ids):
    """Exact two-sided McNemar. Returns (a right b wrong, a wrong b right, p)."""
    b = sum(a_ok[i] and not b_ok[i] for i in ids)
    c = sum(b_ok[i] and not a_ok[i] for i in ids)
    n = b + c
    if n == 0:
        return b, c, 1.0
    p = 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return b, c, min(1.0, p)


def auroc(score, wrong, ids):
    """P(score of a wrong vote > score of a right vote), ties count 1/2.
    Returns None when only one class is present."""
    pos = [score[i] for i in ids if wrong[i]]
    neg = [score[i] for i in ids if not wrong[i]]
    if not pos or not neg:
        return None
    # rank-based (Mann-Whitney) so it is fast enough for bootstrap
    allv = sorted(pos + neg)
    rank, j = {}, 0
    while j < len(allv):
        k = j
        while k < len(allv) and allv[k] == allv[j]:
            k += 1
        rank[allv[j]] = (j + k + 1) / 2  # average 1-based rank
        j = k
    rsum = sum(rank[s] for s in pos)
    return (rsum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def unan_prec(s, gold, ids):
    ag = [i for i in ids if s[i][0] and len(set(s[i])) == 1]
    if not ag:
        return None
    return sum(s[i][0] == gold[i] for i in ag) / len(ag)


def ci(vals):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return (float("nan"), float("nan"))
    lo = vals[int(0.025 * (len(vals) - 1))]
    hi = vals[int(0.975 * (len(vals) - 1))]
    return lo, hi


def boot(ids, fn, rng):
    """Bootstrap fn(list_of_ids) over questions; returns (point, lo, hi)."""
    point = fn(ids)
    reps = [fn([rng.choice(ids) for _ in ids]) for _ in range(B)]
    lo, hi = ci(reps)
    return point, lo, hi


def fmt(x, pct=True):
    if x is None:
        return "  n/a"
    return f"{100 * x:5.1f}" if pct else f"{x:.3f}"


def state(v, g):
    if v[0] and len(set(v)) == 1:
        return "unan_right" if v[0] == g else "unan_wrong"
    return "split"


# ---------- per model ----------
def analyze(d, rc_rows):
    rng = random.Random(SEED)
    gold = {r["id"]: r["gold"] for r in load(d / "single.jsonl")}
    ids = sorted(gold)
    n = len(ids)

    cot = {r["id"]: r["pred"] for r in load(d / "cot.jsonl")}
    single = {r["id"]: r["pred"] for r in load(d / "single.jsonl")}
    sc = group(load(d / "sc.jsonl"), "sample")
    r1 = group(load(d / "round1.jsonl"), "agent")
    r2 = group(load(d / "round2.jsonl"), "agent")
    sets = {"SC-3": {i: v[:3] for i, v in sc.items()}, "SC-6": sc, "round1": r1, "round2": r2}

    # planted: 3 answers per question (2 honest round-2 + planted agent's reply)
    pfile = d / "planted_round2.jsonl"
    if pfile.exists():
        pg = {r["id"]: r for r in load(d / "planted_gen.jsonl")}
        pv = {}
        for r in load(pfile):
            pv.setdefault(r["id"], {})[r["agent"]] = r["pred"]
        for i, dd in pv.items():
            dd[pg[i]["agent"]] = pg[i]["pred"]
        sets["planted"] = {i: [dd[k] for k in sorted(dd)] for i, dd in pv.items()}

    votes = {k: {i: vote(v) for i, v in s.items()} for k, s in sets.items()}
    ok = {k: {i: votes[k].get(i) == gold[i] for i in ids} for k in votes}
    ok["single"] = {i: single.get(i) == gold[i] for i in ids}
    ok["CoT"] = {i: cot.get(i) == gold[i] for i in ids}

    print(f"\n===== {d.name} ({n} questions, bootstrap B={B}) =====")

    # 1. accuracy + CIs
    print("1) vote accuracy %  [95% CI]")
    for k in ["single", "CoT"] + list(sets):
        p, lo, hi = boot(ids, lambda s, k=k: sum(ok[k][i] for i in s) / len(s), rng)
        print(f"   {k:8} {fmt(p)}  [{fmt(lo)}, {fmt(hi)}]")
    ties = {k: sum(no_majority(s[i]) for i in ids) for k, s in sets.items()}
    print("   tie-rule used (no strict majority): "
          + "  ".join(f"{k} {v}" for k, v in ties.items()))

    # 2. McNemar
    print("2) exact McNemar (A right/B wrong, A wrong/B right, p)")
    pairs = [("round1", "round2"), ("SC-6", "round2"), ("CoT", "round2"), ("SC-3", "round1")]
    if "planted" in sets:
        pairs.append(("round2", "planted"))
    for a, b in pairs:
        x, y, p = mcnemar(ok[a], ok[b], ids)
        print(f"   {a:7} vs {b:8}: {x:3} / {y:3}   p = {p:.4f}")
    dacc = boot(ids, lambda s: (sum(ok["round2"][i] for i in s) - sum(ok["round1"][i] for i in s)) / len(s), rng)
    print(f"   acc diff round2 - round1: {fmt(dacc[0])} pts  [{fmt(dacc[1])}, {fmt(dacc[2])}]")

    # 3. unanimous precision
    print("3) unanimous precision %  [95% CI]")
    for k in sets:
        p, lo, hi = boot(ids, lambda s, k=k: unan_prec(sets[k], gold, s), rng)
        print(f"   {k:8} {fmt(p)}  [{fmt(lo)}, {fmt(hi)}]")

    def dprec(s):
        a, b = unan_prec(r1, gold, s), unan_prec(r2, gold, s)
        return None if a is None or b is None else b - a
    p, lo, hi = boot(ids, dprec, rng)
    print(f"   diff round2 - round1: {fmt(p)} pts  [{fmt(lo)}, {fmt(hi)}]")
    uw = boot(ids, lambda s: sum(state(r2[i], gold[i]) == "unan_wrong" for i in s)
              - sum(state(r1[i], gold[i]) == "unan_wrong" for i in s), rng)
    print(f"   unanimous-wrong count change r2 - r1: {uw[0]:+d}  [{uw[1]:+d}, {uw[2]:+d}]")

    # 4. transitions
    print("4) r1 -> r2 transitions (rows r1, cols r2)")
    names = ["unan_right", "unan_wrong", "split"]
    t = Counter((state(r1[i], gold[i]), state(r2[i], gold[i])) for i in ids)
    print(f"   {'':12}" + "".join(f"{c:>12}" for c in names))
    for r in names:
        print(f"   {r:12}" + "".join(f"{t[(r, c)]:12}" for c in names))
    print(f"   new unanimous-wrong in r2: from split {t[('split', 'unan_wrong')]}, "
          f"from unan_right {t[('unan_right', 'unan_wrong')]}")

    # 5. AUROC
    print("5) AUROC: disagreement (1 - agreement fraction) -> vote wrong  [95% CI]")
    dis = {k: {i: 1 - agree_frac(s[i]) for i in ids} for k, s in sets.items()}
    wrong = {k: {i: not ok[k][i] for i in ids} for k in sets}
    for k in sets:
        p, lo, hi = boot(ids, lambda s, k=k: auroc(dis[k], wrong[k], s), rng)
        print(f"   {k:8} {fmt(p, False)}  [{fmt(lo, False)}, {fmt(hi, False)}]")

    def dauc(s):
        a, b = auroc(dis["round1"], wrong["round1"], s), auroc(dis["round2"], wrong["round2"], s)
        return None if a is None or b is None else b - a
    p, lo, hi = boot(ids, dauc, rng)
    print(f"   diff round2 - round1: {fmt(p, False)}  [{fmt(lo, False)}, {fmt(hi, False)}]")

    # 6. risk-coverage
    print("6) risk-coverage: answer only if agreement >= level")
    for k, s in sets.items():
        af = {i: agree_frac(s[i]) for i in ids}
        levels = sorted(set(af.values()), reverse=True)
        parts = []
        for lv in levels:
            kept = [i for i in ids if af[i] >= lv - 1e-9]
            acc = sum(ok[k][i] for i in kept) / len(kept)
            cov = len(kept) / n
            parts.append(f"{lv:.2f}: cov {100 * cov:5.1f} acc {100 * acc:5.1f}")
            rc_rows.append([d.name, k, f"{lv:.4f}", len(kept), n, f"{cov:.4f}", f"{acc:.4f}"])
        print(f"   {k:8} " + " | ".join(parts))


if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    rows = []
    for m in sys.argv[2:] or ["qwen32", "qwen8", "gemma12", "llama8"]:
        analyze(root / m, rows)
    out = root / "risk_coverage.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "method", "agree_level", "kept", "n", "coverage", "accuracy"])
        w.writerows(rows)
    print(f"\nwrote {out}")
