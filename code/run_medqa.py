"""Multi-agent discussion vs self-consistency on MedQA: all main setups for one model.

Usage (from the repository root, on a machine with a GPU and vLLM installed):
  python code/run_medqa.py --model qwen8 --limit 20   # pilot: first 20 questions
  python code/run_medqa.py --model qwen8              # full run: 300 questions

Input : data/medqa_300.jsonl          (made by code/prepare_medqa.py)
Output: results/<model>[_pilot]/<stage>.jsonl, one line per LLM call.
Set the environment variable PROJ to use another root folder (default: current folder).

Stages (run in this order; a finished stage file is skipped, so a run can be restarted):
  single  P1, 1 call, greedy         -> plain accuracy
  cot     P2, 1 call, greedy         -> chain-of-thought accuracy
  sc      P2, 6 samples, temp 0.7    -> SC-3 = samples 0-2, SC-6 = samples 0-5
  round1  P3, 3 role agents, temp 0.7 (independent)            -> same number of calls as SC-3
  round2  P4, each agent sees the other two and revises        -> total calls equal to SC-6
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

PROJ = Path(os.environ.get("PROJ", "."))
DATA = PROJ / "data" / "medqa_300.jsonl"
RESULTS = PROJ / "results"

MODELS = {
    # key: (Hugging Face repo, GPUs, chat_template_kwargs)
    "qwen4": ("Qwen/Qwen3-4B", 1, {"enable_thinking": False}),    # thinking off: reasoning set by the prompt
    "qwen8": ("Qwen/Qwen3-8B", 1, {"enable_thinking": False}),
    "qwen14": ("Qwen/Qwen3-14B", 1, {"enable_thinking": False}),
    "qwen32": ("Qwen/Qwen3-32B", 2, {"enable_thinking": False}),
    "llama8": ("meta-llama/Llama-3.1-8B-Instruct", 1, {}),        # gated on Hugging Face
    "gemma12": ("google/gemma-3-12b-it", 1, {}),                  # gated on Hugging Face
}

ROLES = ["internist", "clinical pharmacologist", "pathologist"]
SC_SAMPLES = 6
MAX_TOKENS = 1024      # per reply
MAX_MODEL_LEN = 8192   # prompt + reply context
TEMPERATURE = 0.7
FORMAT = "End your reply with a final line of the form: ANSWER: X (X is one of A, B, C, D)."

P1_SINGLE = ("Answer the following USMLE question. Reply with only the letter of the correct option.\n"
             + FORMAT)
P2_COT = ("You are a medical expert. Think step by step about the question, then give a short "
          "explanation (3-5 sentences) and your final choice.\n" + FORMAT)


def you_are(role):
    """'You are an internist' / 'You are a pathologist' (correct article)."""
    return f"You are {'an' if role[0] in 'aeiou' else 'a'} {role}."


def p3_role(role):
    # identical to P2 except for the role line, so the role is the only difference from SC
    return (f"{you_are(role)} Think step by step about the question, then give a short "
            "explanation (3-5 sentences) and your final choice.\n" + FORMAT)


def question_text(q):
    opts = "\n".join(f"{k}. {v}" for k, v in sorted(q["options"].items()))
    return f"Question:\n{q['question']}\n\nOptions:\n{opts}"


def p4_revise(role, q, own, others):
    """others = list of (role, response text) for the two colleagues."""
    colleagues = "\n\n".join(f"Colleague {i + 1} ({r}):\n{t}" for i, (r, t) in enumerate(others))
    return (f"{you_are(role)}\n\n{question_text(q)}\n\n"
            f"Your earlier answer:\n{own}\n\n"
            f"Here are the answers and reasoning of two colleagues:\n\n{colleagues}\n\n"
            "Reconsider the question. You may keep or change your answer. "
            "Give a short explanation (3-5 sentences).\n" + FORMAT)


ANS_RE = re.compile(r"ANSWER\s*\**\s*[:：]\s*\**\s*\(?([ABCD])\b", re.I)


def parse_answer(text):
    """Last 'ANSWER: X' in the reply; None if missing."""
    m = ANS_RE.findall(text or "")
    return m[-1].upper() if m else None


def chat(instruction, q=None):
    """One-turn user message. If q is given, the question is appended after the instruction."""
    content = instruction if q is None else f"{instruction}\n\n{question_text(q)}"
    return [{"role": "user", "content": content}]


def run_stage(llm, tmpl, path, items, **chat_kw):
    """items = list of (record_dict, messages, SamplingParams). Writes one jsonl line per call."""
    if path.exists():
        print(f"skip {path.name} (exists)", flush=True)
        return [json.loads(l) for l in open(path)]
    t0 = time.time()
    outs = llm.chat([m for _, m, _ in items], [p for _, _, p in items],
                    chat_template_kwargs=tmpl, use_tqdm=False, **chat_kw)
    recs = []
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        for (rec, msgs, _), o in zip(items, outs):
            text = o.outputs[0].text
            rec = dict(rec, pred=parse_answer(text), response=text,
                       prompt_tokens=len(o.prompt_token_ids), output_tokens=len(o.outputs[0].token_ids))
            recs.append(rec)
            f.write(json.dumps(rec) + "\n")
    tmp.rename(path)  # only a complete stage gets the final name
    ok = sum(r["pred"] is not None for r in recs)
    print(f"{path.name}: {len(recs)} calls, {ok} parsed, {time.time() - t0:.0f}s", flush=True)
    return recs


def base(q):
    return {"id": q["id"], "gold": q["answer_idx"], "step": q["step"]}


def main():
    from vllm import LLM, SamplingParams

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions (pilot)")
    args = ap.parse_args()
    repo, gpus, tmpl = MODELS[args.model]

    qs = [json.loads(l) for l in open(DATA)]
    if args.limit:
        qs = qs[:args.limit]
    out = RESULTS / f"{args.model}{'_pilot' if args.limit else ''}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"{args.model}: {len(qs)} questions -> {out}", flush=True)

    llm = LLM(model=repo, tensor_parallel_size=gpus, max_model_len=MAX_MODEL_LEN,
              gpu_memory_utilization=0.90, seed=0)
    greedy = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)

    run_stage(llm, tmpl, out / "single.jsonl",
              [(base(q), chat(P1_SINGLE, q), greedy) for q in qs])
    run_stage(llm, tmpl, out / "cot.jsonl",
              [(base(q), chat(P2_COT, q), greedy) for q in qs])
    run_stage(llm, tmpl, out / "sc.jsonl",
              [(dict(base(q), sample=k), chat(P2_COT, q),
                SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_TOKENS, seed=1000 * k + i))
               for i, q in enumerate(qs) for k in range(SC_SAMPLES)])
    r1 = run_stage(llm, tmpl, out / "round1.jsonl",
                   [(dict(base(q), agent=a, role=role), chat(p3_role(role), q),
                     SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_TOKENS, seed=1000 * a + i))
                    for i, q in enumerate(qs) for a, role in enumerate(ROLES)])

    r1_by = {(r["id"], r["agent"]): r["response"] for r in r1}
    items = []
    for i, q in enumerate(qs):
        for a, role in enumerate(ROLES):
            others = [(ROLES[b], r1_by[(q["id"], b)]) for b in range(len(ROLES)) if b != a]
            msgs = [{"role": "user", "content": p4_revise(role, q, r1_by[(q["id"], a)], others)}]
            items.append((dict(base(q), agent=a, role=role), msgs,
                          SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_TOKENS, seed=5000 + 1000 * a + i)))
    run_stage(llm, tmpl, out / "round2.jsonl", items)
    print("finished", out, flush=True)


if __name__ == "__main__":
    main()
