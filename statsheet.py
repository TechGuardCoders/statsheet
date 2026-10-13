"""statsheet — statistical significance engine for AI benchmarks.

Turns raw benchmark numbers into defensible claims:

    "Model B wins by 3.2% ± 1.1% (p<0.05)"

Core idea: benchmark samples are draws from a distribution. A point estimate
without an interval is marketing. statsheet bootstraps the sampling distribution
of any metric difference, runs paired permutation tests, and reports effect
sizes with sample-size context.

Works on any JSONL/JSON records with a metric per sample:
    {"model": "a", "metric": 0.83}
    {"model": "b", "metric": 0.86}
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

__version__ = "0.1.0"


# ---------------------------------------------------------------- loading

def load_records(path: str | Path, metric_key: str = "metric") -> list[float]:
    """Load a JSONL (or JSON array) file of records, extracting one metric."""
    path = Path(path)
    records: list[float] = []
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl" or "\n" in text.strip() and not text.lstrip().startswith("["):
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records.append(float(rec[metric_key]))
    else:
        for rec in json.loads(text):
            records.append(float(rec[metric_key]))
    if not records:
        raise ValueError(f"no records with metric {metric_key!r} in {path}")
    return records


def load_paired(path: str | Path, key_a: str, key_b: str,
                metric_key: str = "metric") -> tuple[list[float], list[float]]:
    """Load paired samples: records carrying both models' metrics on the same case."""
    path = Path(path)
    a, b = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if key_a in rec and key_b in rec:
            a.append(float(rec[key_a][metric_key] if isinstance(rec[key_a], dict) else rec[key_a]))
            b.append(float(rec[key_b][metric_key] if isinstance(rec[key_b], dict) else rec[key_b]))
    if not a:
        raise ValueError(f"no paired records with keys {key_a}/{key_b} in {path}")
    return a, b


# ---------------------------------------------------------------- bootstrap

def bootstrap_ci(samples: list[float], statistic=sum / len if False else None,
                 stat=None, n_boot: int = 10_000, confidence: float = 0.95,
                 seed: int = 42) -> tuple[float, float, float]:
    """Percentile bootstrap CI for a statistic over samples.

    Returns (point_estimate, ci_low, ci_high).
    """
    if stat is None:
        stat = _mean
    rng = random.Random(seed)
    n = len(samples)
    point = stat(samples)
    boots = []
    for _ in range(n_boot):
        resample = [samples[rng.randrange(n)] for _ in range(n)]
        boots.append(stat(resample))
    boots.sort()
    alpha = (1.0 - confidence) / 2.0
    lo_idx = max(0, int(alpha * n_boot))
    hi_idx = min(n_boot - 1, int((1 - alpha) * n_boot))
    return point, boots[lo_idx], boots[hi_idx]


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


# ---------------------------------------------------------------- comparison

@dataclass
class ComparisonResult:
    """The headline object: everything a defensible claim needs."""
    name_a: str
    name_b: str
    metric_a: float
    metric_b: float
    delta: float                    # b - a
    delta_ci: tuple[float, float]   # bootstrap CI of the delta
    ci_level: float
    p_value: float                  # paired permutation test
    effect_size: float              # cohens d (pooled)
    n_a: int
    n_b: int
    n_paired: int = 0
    paired: bool = False
    verdict: str = ""

    def headline(self) -> str:
        """e.g. 'Model B wins by 3.2% ± 1.1% (p<0.05)'"""
        return format_comparison(self)

    def to_dict(self) -> dict:
        return {
            "a": self.name_a, "b": self.name_b,
            "metric_a": round(self.metric_a, 6), "metric_b": round(self.metric_b, 6),
            "delta": round(self.delta, 6),
            "delta_ci": [round(self.delta_ci[0], 6), round(self.delta_ci[1], 6)],
            "ci_level": self.ci_level,
            "p_value": round(self.p_value, 6),
            "effect_size_cohens_d": round(self.effect_size, 4),
            "n_a": self.n_a, "n_b": self.n_b, "n_paired": self.n_paired,
            "paired": self.paired,
            "verdict": self.verdict,
            "headline": self.headline(),
        }


def compare(a: list[float], b: list[float], name_a: str = "A", name_b: str = "B",
            confidence: float = 0.95, n_boot: int = 10_000, seed: int = 42,
            paired: bool | None = None) -> ComparisonResult:
    """Compare two samples of a metric. If lengths match and paired is None,
    auto-detect pairing (same length => treat as paired, the stronger test)."""
    if paired is None:
        paired = len(a) == len(b) and len(a) > 1

    mean_a, mean_b = _mean(a), _mean(b)
    delta = mean_b - mean_a

    # bootstrap CI on the delta
    rng = random.Random(seed)
    n_boot_deltas = []
    na, nb = len(a), len(b)
    for _ in range(n_boot):
        ra = [a[rng.randrange(na)] for _ in range(na)]
        rb = [b[rng.randrange(nb)] for _ in range(nb)]
        n_boot_deltas.append(_mean(rb) - _mean(ra))
    n_boot_deltas.sort()
    alpha = (1.0 - confidence) / 2.0
    lo = n_boot_deltas[max(0, int(alpha * n_boot))]
    hi = n_boot_deltas[min(n_boot - 1, int((1 - alpha) * n_boot))]

    # significance
    if paired and len(a) == len(b):
        p = paired_permutation_test(a, b, seed=seed)
    else:
        p = permutation_test(a, b, seed=seed)

    d = cohens_d(a, b)

    # verdict
    ci_excludes_zero = (lo > 0) or (hi < 0)
    if delta > 0 and ci_excludes_zero and p < (1 - confidence):
        verdict = f"{name_b} wins decisively"
    elif delta < 0 and ci_excludes_zero and p < (1 - confidence):
        verdict = f"{name_a} wins decisively"
    elif p < (1 - confidence):
        verdict = "statistically significant but CI touches zero — report with caution"
    else:
        verdict = "no statistically significant difference"

    return ComparisonResult(
        name_a=name_a, name_b=name_b,
        metric_a=mean_a, metric_b=mean_b,
        delta=delta, delta_ci=(lo, hi), ci_level=confidence,
        p_value=p, effect_size=d,
        n_a=na, n_b=nb, n_paired=len(a) if paired else 0, paired=paired,
        verdict=verdict,
    )


def format_comparison(r: ComparisonResult) -> str:
    """The headline: 'Model B wins by 3.2% ± 1.1% (p<0.05)'."""
    pct = r.delta * 100
    half_width = (r.delta_ci[1] - r.delta_ci[0]) / 2 * 100
    winner = r.name_b if r.delta > 0 else r.name_a
    p_str = f"p<{1 - r.ci_level:.2f}" if r.p_value < (1 - r.ci_level) else f"p={r.p_value:.3f}"
    if abs(r.delta) < 1e-12:
        return "no measurable difference"
    return (f"{winner} wins by {abs(pct):.1f}% ± {abs(half_width):.1f}% ({p_str}, "
            f"n={r.n_paired if r.paired else min(r.n_a, r.n_b)})")


# ---------------------------------------------------------------- tests

def paired_permutation_test(a: list[float], b: list[float], n_perm: int = 10_000,
                            seed: int = 42) -> float:
    """Exact-ish paired permutation test on the mean difference.
    H0: the per-case differences are symmetric around 0."""
    diffs = [bi - ai for ai, bi in zip(a, b)]
    observed = abs(_mean(diffs))
    rng = random.Random(seed)
    n = len(diffs)
    if n == 0:
        return 1.0
    count = 0
    for _ in range(n_perm):
        flipped = [d * (1 if rng.random() < 0.5 else -1) for d in diffs]
        if abs(_mean(flipped)) >= observed:
            count += 1
    return (count + 1) / (n_perm + 1)  # +1 correction


def permutation_test(a: list[float], b: list[float], n_perm: int = 10_000,
                     seed: int = 42) -> float:
    """Two-sample permutation test on the difference of means."""
    observed = abs(_mean(a) - _mean(b))
    pooled = a + b
    na = len(a)
    rng = random.Random(seed)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        if abs(_mean(pooled[:na]) - _mean(pooled[na:])) >= observed:
            count += 1
    return (count + 1) / (n_perm + 1)


# ---------------------------------------------------------------- effect sizes

def cohens_d(a: list[float], b: list[float]) -> float:
    """Cohen's d with pooled SD. |d|<0.2 negligible, 0.2 small, 0.5 medium, 0.8 large."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    ma, mb = _mean(a), _mean(b)
    var_a = sum((x - ma) ** 2 for x in a) / (na - 1)
    var_b = sum((x - mb) ** 2 for x in b) / (nb - 1)
    pooled_sq = ((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2)
    if pooled_sq <= 0:
        return 0.0
    return (mb - ma) / math.sqrt(pooled_sq)


def cliffs_delta(a: list[float], b: list[float]) -> float:
    """Non-parametric effect size in [-1, 1]. Robust to outliers/scale."""
    if not a or not b:
        return 0.0
    gt = sum(1 for x in a for y in b if x > y)
    lt = sum(1 for x in a for y in b if x < y)
    return (gt - lt) / (len(a) * len(b))


# ---------------------------------------------------------------- power

def required_sample_size(delta: float, std: float, alpha: float = 0.05,
                         power: float = 0.8) -> int:
    """Approximate per-group sample size to detect `delta` with std `std`.
    Uses the standard two-sample z-approximation."""
    if delta == 0 or std == 0:
        return 0
    # z quantiles (hardcoded for common cases; fine for planning)
    z_alpha = 1.96 if alpha == 0.05 else 2.58 if alpha == 0.01 else 1.96
    z_power = 0.84 if power == 0.8 else 1.28 if power == 0.9 else 0.84
    n = 2 * ((z_alpha + z_power) * std / delta) ** 2
    return max(2, math.ceil(n))


# ---------------------------------------------------------------- report

def compare_files(path_a: str, path_b: str, name_a: str = "A", name_b: str = "B",
                  metric_key: str = "metric", **kwargs) -> ComparisonResult:
    a = load_records(path_a, metric_key)
    b = load_records(path_b, metric_key)
    return compare(a, b, name_a=name_a, name_b=name_b, **kwargs)


def compare_paired_file(path: str, key_a: str, key_b: str,
                        metric_key: str = "metric", **kwargs) -> ComparisonResult:
    a, b = load_paired(path, key_a, key_b, metric_key)
    return compare(a, b, name_a=key_a, name_b=key_b, paired=True, **kwargs)


# ---------------------------------------------------------------- CLI

def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="statsheet — significance for benchmarks")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compare", help="compare two JSONL files")
    c.add_argument("file_a"); c.add_argument("file_b")
    c.add_argument("--names", default="A,B")
    c.add_argument("--metric", default="metric")
    c.add_argument("--unpaired", action="store_true")

    r = sub.add_parser("report", help="paired comparison from one JSONL")
    r.add_argument("file")
    r.add_argument("--baseline", required=True)
    r.add_argument("--treatment", required=True)
    r.add_argument("--metric", default="metric")

    args = p.parse_args()
    if args.cmd == "compare":
        na, nb = args.names.split(",")
        res = compare_files(args.file_a, args.file_b, na, nb, args.metric,
                            paired=not args.unpaired)
    else:
        res = compare_paired_file(args.file, args.baseline, args.treatment, args.metric)
    print(json.dumps(res.to_dict(), indent=2))


if __name__ == "__main__":
    main()
