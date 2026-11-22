"""Summary of MedQA multi-agent results (read-only).

Usage (from the analysis/ folder):
    python3 summarize.py ../results qwen4 qwen8 qwen14 qwen32 gemma12 llama8

For each model folder: accuracy, agreement / unanimous-wrong table, planted-agent effect.
Votes: majority over parsed answers, tie -> earliest agent / sample (same rule as before).
"""
import json
import sys
from collections import Counter
from pathlib import Path


def load(p):
    return [json.loads(line) for line in open(p)]


def vote(preds):
    preds = [p for p in preds if p]
    if not preds:
        return None
    c = Counter(preds)
    top = max(c.values())
    return next(p for p in preds if c[p] == top)


def group(recs, key):
    g = {}
    for r in recs:
        g.setdefault(r["id"], {})[r[key]] = r["pred"]
    return {i: [d[k] for k in sorted(d)] for i, d in g.items()}


def summarize(d):
    gold = {r["id"]: r["gold"] for r in load(d / "single.jsonl")}
    n = len(gold)

    def acc(preds):
        return sum(preds.get(i) == gold[i] for i in gold)

    single = {r["id"]: r["pred"] for r in load(d / "single.jsonl")}
    cot = {r["id"]: r["pred"] for r in load(d / "cot.jsonl")}
    sc = group(load(d / "sc.jsonl"), "sample")
    r1 = group(load(d / "round1.jsonl"), "agent")
    r2 = group(load(d / "round2.jsonl"), "agent")
    sets = {"SC-3": {i: v[:3] for i, v in sc.items()}, "SC-6": sc, "round1": r1, "round2": r2}
    votes = {k: {i: vote(v) for i, v in s.items()} for k, s in sets.items()}

    print(f"\n===== {d.name} ({n} questions) =====")
    print(f"accuracy: single {acc(single)}  CoT {acc(cot)}  "
          + "  ".join(f"{k} {acc(v)}" for k, v in votes.items()))
    a1 = sum(p == gold[i] for i, v in r1.items() for p in v)
    a2 = sum(p == gold[i] for i, v in r2.items() for p in v)
    print(f"agent-level correct: round1 {a1}/{3 * n}  round2 {a2}/{3 * n}")

    print(f"{'':8}{'agree':>6}{'agr_ok':>7}{'disagree':>9}{'unan_wrong':>11}{'unan_prec':>10}")
    for k, s in sets.items():
        ag = [i for i, v in s.items() if v[0] and len(set(v)) == 1]
        ok = sum(s[i][0] == gold[i] for i in ag)
        prec = 100 * ok / len(ag) if ag else 0.0
        print(f"{k:8}{len(ag):6}{ok:7}{n - len(ag):9}{len(ag) - ok:11}{prec:9.1f}%")

    # answer switching round1 -> round2 (same agent, same question)
    same = rw = wr = ww = copied = 0
    for i in gold:
        for a in range(len(r1[i])):
            p1, p2 = r1[i][a], r2[i][a]
            if p1 == p2:
                same += 1
                continue
            ok1, ok2 = p1 == gold[i], p2 == gold[i]
            rw += ok1 and not ok2
            wr += ok2 and not ok1
            ww += not ok1 and not ok2
            copied += p2 in [r1[i][b] for b in range(len(r1[i])) if b != a]
    ch = rw + wr + ww
    print(f"switching r1->r2: kept {same}, changed {ch} (to a peer's r1 answer {copied}); "
          f"right->wrong {rw}, wrong->right {wr}, wrong->wrong {ww}")

    if not (d / "planted_round2.jsonl").exists():
        print("planted: no files")
        return
    pl = load(d / "planted_round2.jsonl")
    pg = {r["id"]: r for r in load(d / "planted_gen.jsonl")}
    hc = ctl = hp = cp = flips = flips_t = 0
    pv = {}
    for r in pl:
        i, a, t = r["id"], r["agent"], r["target"]
        c = r2[i][a]  # same agent, same seed, no planted reply
        hc += r["pred"] == gold[i]
        ctl += c == gold[i]
        hp += r["pred"] == t
        cp += c == t
        if c == gold[i] and r["pred"] != gold[i]:
            flips += 1
            flips_t += r["pred"] == t
        pv.setdefault(i, {})[a] = r["pred"]
    for i, dd in pv.items():
        dd[pg[i]["agent"]] = pg[i]["pred"]
    pvote = {i: vote([dd[k] for k in sorted(dd)]) for i, dd in pv.items()}
    on_t = sum(pvote[i] == pg[i]["target"] for i in pvote)
    print(f"planted: honest correct {hc}/{len(pl)} (control {ctl})  "
          f"picked planted letter {hp} (control {cp})")
    print(f"         vote acc {acc(pvote)} (round2 {acc(votes['round2'])})  vote = planted letter {on_t}")
    print(f"         correct->wrong flips {flips}, of which to planted letter {flips_t}")


if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    for m in sys.argv[2:] or ["qwen32", "qwen8", "gemma12"]:
        summarize(root / m)
