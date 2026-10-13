# statsheet

**Statistical significance engine for AI benchmarks.**

Turns raw benchmark numbers into defensible claims:

> Model B wins by 3.2% ± 1.1% (p<0.05)

instead of "Model B seems better".

## Why

A point estimate without an interval is marketing. A 2-point delta on 50 samples is noise — but executives make million-dollar routing decisions on exactly those numbers. statsheet bootstraps the sampling distribution of any metric difference, runs paired permutation tests, and reports effect sizes with the sample-size context needed to judge them.

## What it does

- **Bootstrap confidence intervals** on any metric and on the delta between two models
- **Paired permutation tests** when both models ran on the same cases (the stronger, correct test for benchmark data)
- **Effect sizes** — Cohen's d (parametric) and Cliff's delta (non-parametric, robust to outliers)
- **Sample-size planning** — "how many samples do I need to detect a 3% delta?"
- **The headline** — one string that says who won, by how much, with what uncertainty, at what significance

## Quick start

```bash
pip install -e .  # or just: python statsheet.py

# compare two benchmark files (JSONL: {"metric": 0.83} per line)
python statsheet.py compare baseline.jsonl treatment.jsonl --names baseline,treatment

# paired comparison from one file ({"baseline": 0.8, "treatment": 0.84} per line)
python statsheet.py report bench.jsonl --baseline baseline --treatment treatment
```

```python
from statsheet import compare, compare_paired_file

r = compare_paired_file("bench.jsonl", "baseline", "treatment")
print(r.headline())   # treatment wins by 4.1% ± 0.8% (p<0.05, n=50)
print(r.verdict)      # treatment wins decisively
print(r.to_dict())    # full machine-readable result
```

## The API

```python
compare(a, b, ...)                    # the headline object
bootstrap_ci(samples)                 # percentile CI for any statistic
paired_permutation_test(a, b)         # p-value, paired
permutation_test(a, b)                # p-value, unpaired
cohens_d(a, b)                        # parametric effect size
cliffs_delta(a, b)                    # non-parametric effect size
required_sample_size(delta, std)      # planning: n per group
```

## Design decisions

- **Auto-pairing**: same-length inputs are treated as paired (the stronger test). Override with `paired=False`.
- **+1 correction** on permutation p-values: an observed result as extreme as itself is always possible.
- **Deterministic seeds** everywhere: same data → same answer, always.
- **No dependencies**: stdlib only. Statistics this basic shouldn't need a 200MB install.

## Part of the TechGuardCoders portfolio

Extends [Batcher](https://github.com/TechGuardCoders/Batcher) and [Green-Zone](https://github.com/TechGuardCoders/Green-Zone): their published benchmark numbers can now be re-analyzed with confidence intervals and significance tests. Every future benchmark in the org reports through statsheet.

## Tests

```bash
python -m pytest tests/ -q   # 24 tests
```

The test suite encodes the core claim: **a small delta on a tiny sample must never be called a win** (`test_tiny_sample_is_noise`).
