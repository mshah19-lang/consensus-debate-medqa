"""Build data/medqa_300.jsonl, the 300 MedQA questions used in the paper.

MedQA (Jin et al., 2021) is not redistributed here. Download it from the authors' repository
(https://github.com/jind11/MedQA) and use the US, 4-option test file:
    data_clean/questions/US/4_options/phrases_no_exclude_test.jsonl   (1,273 questions)

Usage (from the repository root):
    python code/prepare_medqa.py path/to/phrases_no_exclude_test.jsonl

Selection: 300 question indices drawn with random.Random(42).sample over the 1,273 test questions,
kept in their original order. Output fields per line:
    {"id": "q000".., "orig_index", "question", "options": {"A".."D"}, "answer_idx", "step"}
("step" is the USMLE step from MedQA's meta_info.)
"""
import json
import random
import sys
from pathlib import Path

SEED = 42
N = 300


def main():
    src = Path(sys.argv[1])
    rows = [json.loads(l) for l in open(src, encoding="utf-8")]
    if len(rows) != 1273:
        print(f"warning: expected 1273 questions, found {len(rows)}; the selection will differ")
    idx = sorted(random.Random(SEED).sample(range(len(rows)), N))
    out = Path("data") / "medqa_300.jsonl"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for k, j in enumerate(idx):
            r = rows[j]
            f.write(json.dumps({"id": f"q{k:03d}", "orig_index": j, "question": r["question"],
                                "options": r["options"], "answer_idx": r["answer_idx"],
                                "step": r["meta_info"]}) + "\n")
    print(f"wrote {out}: {N} questions")


if __name__ == "__main__":
    main()
