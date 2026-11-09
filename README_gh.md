# Same Bytes, Different Scores — evidence package

Paper for the *2026 Gemma 4 Developer Agent* competition, paper track.

**Title:** Same Bytes, Different Scores: What a Code-Agent Leaderboard Measures
**Authors:** Wang Chen and Youtao Liu (South China University of Technology)

## Contents

| File | What it is |
|---|---|
| `Same-Bytes-Different-Scores.pdf` | Final paper (8 pages, 4 figures) |
| `fig1_lineage.png` | Figure 1 — lineages and public scores |
| `fig2_lb_sampling.png` | Figure 2 — score vs submission count |
| `fig3_local_arms.png` | Figure 3 — local paired blocks vs leaderboard |
| `fig5_flow.png` | Figure 4 — how one submission is scored / what we measure |
| `evidence.csv` | 108-row claim table (`part=A` partner-reported, `part=B` our measurements) |
| `recompute.py` | Deterministic recomputation of every Part B number from the CSV |
| `LICENSE` | Apache-2.0 (code); data under CC-BY-4.0 |

## Reproduce

```bash
python recompute.py --csv evidence.csv
```

`recompute.py` prints each Part B claim id with its value; no network access is required.

## Data availability

Raw traces are not redistributed. `Archive A` (partner team) is cited by line number in
the co-author manuscript and is not part of this package. `Archive B` numbers are
recomputable from `evidence.csv` alone.
