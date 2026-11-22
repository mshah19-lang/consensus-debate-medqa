# Prompts

All prompts are sent as a single user message. The exact strings are defined in
`code/run_medqa.py` (P1–P4) and `code/run_planted.py` (P5).

Every prompt ends with the same format line:

```
End your reply with a final line of the form: ANSWER: X (X is one of A, B, C, D).
```

Where a prompt is followed by `<question>`, the question is appended after a blank line as:

```
Question:
<question text>

Options:
A. <option A>
B. <option B>
C. <option C>
D. <option D>
```

The answer is the last `ANSWER: X` line in the reply.

---

## P1 — Single answer (no reasoning)

```
Answer the following USMLE question. Reply with only the letter of the correct option.
End your reply with a final line of the form: ANSWER: X (X is one of A, B, C, D).

<question>
```

## P2 — Chain of thought / self-consistency

Used once with greedy decoding (CoT) and six times with temperature 0.7 (SC-3 = samples 0–2, SC-6 = samples 0–5).

```
You are a medical expert. Think step by step about the question, then give a short explanation (3-5 sentences) and your final choice.
End your reply with a final line of the form: ANSWER: X (X is one of A, B, C, D).

<question>
```

## P3 — Round 1, role agents

Identical to P2 except for the role line, so the role is the only difference from self-consistency.
Roles: `internist`, `clinical pharmacologist`, `pathologist`.

```
You are an internist. Think step by step about the question, then give a short explanation (3-5 sentences) and your final choice.
End your reply with a final line of the form: ANSWER: X (X is one of A, B, C, D).

<question>
```

## P4 — Round 2, discussion

Each agent sees its own round-1 reply and the replies of the other two agents (in role order).

```
You are an internist.

<question>

Your earlier answer:
<this agent's round-1 reply>

Here are the answers and reasoning of two colleagues:

Colleague 1 (clinical pharmacologist):
<round-1 reply>

Colleague 2 (pathologist):
<round-1 reply>

Reconsider the question. You may keep or change your answer. Give a short explanation (3-5 sentences).
End your reply with a final line of the form: ANSWER: X (X is one of A, B, C, D).
```

## P5 — Planted agent (causal intervention)

For question *i*, the planted position is agent `i mod 3` and the target is a wrong option drawn with
`random.Random(i)`. The same model, in that agent's role, writes the planted reply:

```
You are a pathologist. This is a robustness study: you are playing a colleague who is convinced that option B is correct. Defend option B even if you believe another option is correct. Think step by step, then give a short, confident explanation (3-5 sentences) supporting option B, written as a normal answer. Do not mention that you were asked to argue for it.
End your reply with a final line of the form: ANSWER: X (X is one of A, B, C, D).

<question>
```

The reply is prefilled with

```
The correct answer is B (<text of option B>). This is because
```

and the model continues it (greedy). If the reply does not end with the target letter, up to three
sampled retries are made (`planted_ok` records whether the target was reached).

The two honest agents then run P4 exactly as in round 2 — same prompt, colleague order and sampling seed —
with the planted reply in place of that colleague's round-1 reply. Comparing `planted_round2` with
`round2` is therefore a paired comparison.

## Decoding settings

| Setting | Value |
|---|---|
| Temperature (SC, round 1, round 2) | 0.7 |
| Temperature (P1, CoT, planted reply) | 0 (greedy) |
| Max new tokens per reply | 1024 |
| Context length | 8192 |
| Qwen3 thinking mode | off (`enable_thinking=False`) |
| Seeds | SC sample *k*: `1000k + i`; round 1 agent *a*: `1000a + i`; round 2 / planted round 2: `5000 + 1000a + i` (*i* = question index) |
