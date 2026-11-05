# Reproducibility materials

This directory contains the per-video Video Summarization (VS) scores used in
the manuscript and the scripts supplied to generate its data figures and table
statistics.

## GPT-4 VS judge scores

The files in `data/vs_gpt4_scores/` are the original benchmark prediction logs.
Each file contains 40 MLVU Video Summarization examples and records the generated
summary, reference summary, completeness score, reliability score, combined
score, judge explanation, and `judge_model`. All 200 records identify the judge
as `gpt-4-turbo`.

`data/results_long_vs.csv` consolidates the 200 combined scores into the long
format accepted directly by `scripts/statistics_tables.py`.

| Configuration | File | Completeness | Reliability | Combined score |
| --- | --- | ---: | ---: | ---: |
| C1 — Full fusion | `c1_full_fusion.jsonl` | 1.925 | 2.350 | 4.275 |
| C2 — Captions only | `c2_captions_only.jsonl` | 1.800 | 2.300 | 4.100 |
| C3 — Transcript only | `c3_transcript_only.jsonl` | 1.575 | 1.800 | 3.375 |
| C4 — Speech-anchored, fixed windows | `c4_speech_anchored_fixed.jsonl` | 2.050 | 2.575 | 4.625 |
| C5 — Speech-anchored, variable windows | `c5_speech_anchored_variable.jsonl` | 2.075 | 2.475 | 4.550 |

The combined score is the sum of the 1–5 completeness and reliability scores.
The manuscript displays the combined means rounded to two decimal places: 4.28,
4.10, 3.38, 4.63, and 4.55 for C1 through C5, respectively.

## Scripts

- `scripts/make_figures.py` generates the manuscript's data figures as vector
  PDF files.
- `scripts/statistics_tables.py` computes bootstrap confidence intervals,
  paired exact McNemar tests, paired permutation tests, Holm-adjusted p-values,
  modality gains, and threshold-sensitivity results.
- `scripts/transcript_statistics.py` computes the transcription and segment
  duration statistics reported in the manuscript.
- `scripts/crop_frame_thumbnails.py` extracts the thumbnails used in the two
  qualitative frame figures.

The scripts were copied without modification from the manuscript materials.
Their command-line usage and expected input formats are documented in their
module docstrings.

To reproduce the VS confidence intervals and paired tests from the consolidated
score file:

```bash
python reproducibility/scripts/statistics_tables.py \
  reproducibility/data/results_long_vs.csv \
  --out-dir reproducibility/generated/vs_statistics
```

## Dependencies

Install the additional plotting and statistical packages with:

```bash
python -m pip install -r reproducibility/requirements.txt
```

The score files are research outputs, not runtime caches. They are intentionally
versioned even though benchmark-generated JSONL files are ignored elsewhere in
the repository.
