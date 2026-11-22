# Talking Away Uncertainty: Multi-Agent LLM Debate Turns Disagreement into Confident Errors in Medical Question Answering

Code, prompts and model outputs for the paper.

We compare three role-based LLM agents (internist, clinical pharmacologist, pathologist) before and
after one round of discussion against self-consistency with the same number of model calls, on 300
MedQA questions and six open-weight models. Disagreement among the agents is treated as a signal that
the majority answer is wrong (AUROC, unanimous precision, risk–coverage). A paired causal intervention
replaces one agent's round-1 reply with an argument for a wrong option and measures how the honest
agents respond.

## Repository layout

```
code/
  prepare_medqa.py   builds data/medqa_300.jsonl from the official MedQA test file
  run_medqa.py       single answer, CoT, self-consistency, round 1, round 2 (one model)
  run_planted.py     planted wrong agent + paired round 2 (needs round1.jsonl of the same model)
analysis/
  summarize.py       accuracy, agreement table, planted-agent effect
  stats.py           bootstrap CIs, McNemar tests, transitions, AUROC, risk-coverage
  make_figures.py    Figure 1
results/<model>/     all model outputs used in the paper (one JSON line per LLM call)
prompts.md           every prompt, decoding settings and seeds
```

## Models

| Key | Hugging Face model |
|---|---|
| `qwen4` | Qwen/Qwen3-4B |
| `qwen8` | Qwen/Qwen3-8B |
| `qwen14` | Qwen/Qwen3-14B |
| `qwen32` | Qwen/Qwen3-32B (2 GPUs) |
| `gemma12` | google/gemma-3-12b-it |
| `llama8` | meta-llama/Llama-3.1-8B-Instruct |

Qwen3 thinking mode is switched off. Runs used vLLM on NVIDIA A40 (48 GB) GPUs.

## Reproducing the analysis (no GPU needed)

The outputs in `results/` are enough to recompute every number and the figure:

```bash
cd analysis
python3 summarize.py ../results qwen4 qwen8 qwen14 qwen32 gemma12 llama8
python3 stats.py     ../results qwen4 qwen8 qwen14 qwen32 gemma12 llama8   # also writes ../results/risk_coverage.csv
python3 make_figures.py .                                                   # needs matplotlib
```

Statistics use only the Python standard library (bootstrap B = 2,000, seed 0).

## Re-running the experiments

1. **Data.** MedQA is not redistributed here. Download it from <https://github.com/jind11/MedQA> and
   use `data_clean/questions/US/4_options/phrases_no_exclude_test.jsonl` (1,273 questions). Then:
   ```bash
   python code/prepare_medqa.py path/to/phrases_no_exclude_test.jsonl
   ```
   This writes `data/medqa_300.jsonl` (300 questions sampled with seed 42, original order).
2. **Environment.** `pip install -r requirements.txt`. Llama 3.1 and Gemma 3 are gated on Hugging Face:
   accept their licences and log in with `huggingface-cli login` first.
3. **Runs** (from the repository root, one model at a time):
   ```bash
   python code/run_medqa.py   --model qwen8    # writes results/qwen8/{single,cot,sc,round1,round2}.jsonl
   python code/run_planted.py --model qwen8    # writes results/qwen8/{planted_gen,planted_round2}.jsonl
   ```
   Add `--limit 20` for a quick pilot (written to `results/<model>_pilot/`). Finished stages are
   skipped, so an interrupted run can be restarted. Set `PROJ=/some/folder` to use another root folder.

## Output format

One JSON object per line. Common fields: `id` (q000–q299), `gold` (correct letter), `step`
(USMLE step), `pred` (parsed letter or `null`), `response` (full reply), `prompt_tokens`, `output_tokens`.

| File | Calls | Extra fields |
|---|---|---|
| `single.jsonl` | 300 | – |
| `cot.jsonl` | 300 | – |
| `sc.jsonl` | 1,800 | `sample` (0–5) |
| `round1.jsonl` | 900 | `agent` (0–2), `role` |
| `round2.jsonl` | 900 | `agent`, `role` |
| `planted_gen.jsonl` | 300 | `agent`, `role`, `target`, `attempt`, `prefill`, `planted_ok` |
| `planted_round2.jsonl` | 600 | `agent`, `role`, `planted_agent`, `target`, `planted_ok`, `planted_attempt` |

Votes are the majority over parsed answers; ties go to the earliest agent or sample.

## Citation

Citation details will be added after publication.

## Licence

Code: see `LICENSE`. MedQA is distributed by its authors under its own terms.
