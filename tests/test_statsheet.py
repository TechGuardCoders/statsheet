"""statsheet tests — the engine must be honest about noise."""
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from statsheet import (
    bootstrap_ci, compare, compare_paired_file, cohens_d, cliffs_delta,
    format_comparison, load_paired, load_records, paired_permutation_test,
    permutation_test, required_sample_size,
)


# ---------------------------------------------------------------- fixtures

@pytest.fixture
def identical():
    rng = random.Random(1)
    return [rng.gauss(0.8, 0.05) for _ in range(50)], [rng.gauss(0.8, 0.05) for _ in range(50)]


@pytest.fixture
def clear_winner():
    rng = random.Random(2)
    return [rng.gauss(0.80, 0.03) for _ in range(60)], [rng.gauss(0.86, 0.03) for _ in range(60)]


@pytest.fixture
def tiny_sample():
    rng = random.Random(3)
    return [rng.gauss(0.8, 0.05) for _ in range(8)], [rng.gauss(0.83, 0.05) for _ in range(8)]


# ---------------------------------------------------------------- loading

def test_load_jsonl(tmp_path):
    f = tmp_path / "a.jsonl"
    f.write_text('{"metric": 0.5}\n{"metric": 0.7}\n')
    assert load_records(f) == [0.5, 0.7]


def test_load_json_array(tmp_path):
    f = tmp_path / "a.json"
    f.write_text('[{"metric": 0.5}, {"metric": 0.7}]')
    assert load_records(f) == [0.5, 0.7]


def test_load_paired(tmp_path):
    f = tmp_path / "p.jsonl"
    f.write_text('{"baseline": 0.5, "treatment": 0.6}\n{"baseline": 0.7, "treatment": 0.9}\n')
    a, b = load_paired(f, "baseline", "treatment")
    assert a == [0.5, 0.7] and b == [0.6, 0.9]


def test_load_empty_raises(tmp_path):
    f = tmp_path / "e.jsonl"
    f.write_text("")
    with pytest.raises(ValueError):
        load_records(f)


# ---------------------------------------------------------------- bootstrap

def test_bootstrap_ci_contains_point(identical):
    a, _ = identical
    point, lo, hi = bootstrap_ci(a, n_boot=2000)
    assert lo <= point <= hi


def test_bootstrap_ci_narrows_with_more_samples():
    rng = random.Random(4)
    small = [rng.gauss(0.8, 0.1) for _ in range(10)]
    big = [rng.gauss(0.8, 0.1) for _ in range(2000)]
    _, lo_s, hi_s = bootstrap_ci(small, n_boot=2000)
    _, lo_b, hi_b = bootstrap_ci(big, n_boot=2000)
    assert (hi_b - lo_b) < (hi_s - lo_s)


def test_bootstrap_ci_deterministic_with_seed(identical):
    a, _ = identical
    r1 = bootstrap_ci(a, n_boot=500, seed=7)
    r2 = bootstrap_ci(a, n_boot=500, seed=7)
    assert r1 == r2


# ---------------------------------------------------------------- comparison

def test_clear_winner_detected(clear_winner):
    a, b = clear_winner
    r = compare(a, b, "baseline", "treatment")
    assert r.delta > 0
    assert r.p_value < 0.05
    assert "treatment wins" in r.verdict.lower()


def test_identical_no_difference(identical):
    a, b = identical
    r = compare(a, b, "a", "b")
    assert r.verdict.startswith("no statistically significant")


def test_tiny_sample_is_noise(tiny_sample):
    """THE core claim: a 2-point delta on 8 samples must NOT be called a win."""
    a, b = tiny_sample
    r = compare(a, b, "a", "b")
    # whatever the delta, with n=8 the verdict must not be "decisively"
    assert "decisively" not in r.verdict


def test_headline_format(clear_winner):
    a, b = clear_winner
    r = compare(a, b, "baseline", "treatment")
    h = r.headline()
    assert "wins by" in h
    assert "±" in h
    assert "p<" in h
    assert "n=" in h


def test_headline_names_winner(clear_winner):
    a, b = clear_winner
    r = compare(a, b, "baseline", "treatment")
    assert "treatment wins" in r.headline()


def test_paired_stronger_than_unpaired():
    """Paired test on correlated data should detect what unpaired misses."""
    rng = random.Random(5)
    # cases vary a lot but treatment consistently beats baseline per case
    a, b = [], []
    for _ in range(40):
        base = rng.gauss(0.5, 0.3)
        a.append(base)
        b.append(base + 0.05 + rng.gauss(0, 0.01))
    r_paired = compare(a, b, "a", "b", paired=True)
    r_unpaired = compare(a, b, "a", "b", paired=False)
    assert r_paired.p_value <= r_unpaired.p_value


def test_result_serializable(clear_winner):
    a, b = clear_winner
    r = compare(a, b)
    d = r.to_dict()
    json.dumps(d)  # must not raise
    assert d["headline"] == r.headline()


# ---------------------------------------------------------------- tests (stats)

def test_paired_permutation_identical_is_nonsignificant(identical):
    a, b = identical
    assert paired_permutation_test(a, b) > 0.05


def test_paired_permutation_detects_shift(clear_winner):
    a, b = clear_winner
    assert paired_permutation_test(a, b) < 0.01


def test_permutation_test_symmetric():
    rng = random.Random(6)
    a = [rng.gauss(0, 1) for _ in range(30)]
    b = [rng.gauss(0, 1) for _ in range(30)]
    assert permutation_test(a, b) > 0.05


# ---------------------------------------------------------------- effect sizes

def test_cohens_d_small_for_same_distribution(identical):
    """Two draws from the SAME distribution can differ by chance; d must stay
    in the negligible-to-small band, never large."""
    a, b = identical
    assert abs(cohens_d(a, b)) < 0.8


def test_cohens_d_large_for_clear_shift(clear_winner):
    a, b = clear_winner
    assert cohens_d(a, b) > 0.8  # 2 SD shift with 0.03 SD noise


def test_cliffs_delta_bounds():
    assert cliffs_delta([1, 2, 3], [10, 20, 30]) == -1.0
    assert cliffs_delta([10, 20, 30], [1, 2, 3]) == 1.0
    assert cliffs_delta([1, 2], [1, 2]) == 0.0


# ---------------------------------------------------------------- power

def test_required_sample_size_grows_with_noise():
    small_noise = required_sample_size(delta=0.03, std=0.02)
    big_noise = required_sample_size(delta=0.03, std=0.2)
    assert big_noise > small_noise


def test_required_sample_size_shrinks_with_delta():
    small_delta = required_sample_size(delta=0.01, std=0.05)
    big_delta = required_sample_size(delta=0.1, std=0.05)
    assert small_delta > big_delta


# ---------------------------------------------------------------- file roundtrip

def test_compare_paired_file_end_to_end(tmp_path):
    f = tmp_path / "bench.jsonl"
    rng = random.Random(9)
    lines = []
    for _ in range(50):
        base = rng.gauss(0.8, 0.05)
        lines.append(json.dumps({"baseline": base, "treatment": base + 0.04}))
    f.write_text("\n".join(lines))
    r = compare_paired_file(f, "baseline", "treatment")
    assert r.paired
    assert r.delta > 0
    assert "treatment wins" in r.headline()


def test_cli_compare(tmp_path):
    fa, fb = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    rng = random.Random(11)
    fa.write_text("\n".join(json.dumps({"metric": rng.gauss(0.8, 0.03)}) for _ in range(40)))
    fb.write_text("\n".join(json.dumps({"metric": rng.gauss(0.86, 0.03)}) for _ in range(40)))
    out = subprocess.run(
        [sys.executable, str(Path(__file__).parent.parent / "statsheet.py"),
         "compare", str(fa), str(fb), "--names", "old,new"],
        capture_output=True, text=True)
    assert out.returncode == 0
    data = json.loads(out.stdout)
    assert data["headline"]
    assert "new wins" in data["headline"]
