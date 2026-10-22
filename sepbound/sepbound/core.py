"""Separability checks for published leaderboard scores.

Definitions follow Sections III and V of "LLM Agent Leaderboards Often Cannot Separate
Their Top Two Entries" (Proposition 2, the item-sampling and rerun bounds, Algorithm 1). All scores are
proportions in [0, 1]; all bounds are returned in proportion units (multiply by 100 for percentage points).

Model: each score is a mean over n exchangeable, independently drawn tasks, so its standard error is the
binomial SE_j = sqrt(P_j (1 - P_j) / n_j). Two entries scored on the same tasks have a correlation rho between
their per-task outcomes, so SE_D = sqrt(SE_1^2 + SE_2^2 - 2 rho SE_1 SE_2) (normal approximation).
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from . import _norm

Z95 = 1.96

__all__ = [
    "item_sampling_bound", "separable", "order_confidence", "leader_posterior", "leader_set",
    "rerun_bound", "tiers", "check_leaderboard", "to_proportions", "z_bonferroni",
]


# --------------------------------------------------------------------------- helpers

def _scalar(x):
    """0-dim array -> float, anything else unchanged."""
    return float(x) if np.ndim(x) == 0 else x


def _prop(p, name="score"):
    p = np.asarray(p, dtype=float)
    if np.any(~np.isfinite(p)) or np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError(f"{name} must be proportions in [0, 1]; use to_proportions() for percentages")
    return p


def _count(n, name="n"):
    n = np.asarray(n, dtype=float)
    if np.any(~np.isfinite(n)) or np.any(n <= 0.0):
        raise ValueError(f"{name} must be positive")
    return n


def _rho(rho):
    rho = np.asarray(rho, dtype=float)
    if np.any(~np.isfinite(rho)) or np.any(rho < -1.0) or np.any(rho > 1.0):
        raise ValueError("rho must lie in [-1, 1]")
    return rho


def _se(p, n):
    return np.sqrt(p * (1.0 - p) / n)


def _se_d(p1, p2, n1, n2, rho):
    s1, s2 = _se(p1, n1), _se(p2, n2)
    return np.sqrt(np.maximum(s1 ** 2 + s2 ** 2 - 2.0 * rho * s1 * s2, 0.0))


def to_proportions(scores):
    """Convert scores to proportions. If any value exceeds 1 the input is read as percent (divided by 100)."""
    s = np.asarray(scores, dtype=float)
    if np.any(~np.isfinite(s)) or np.any(s < 0):
        raise ValueError("scores must be finite and non-negative")
    if np.any(s > 1.0):
        if np.any(s > 100.0):
            raise ValueError("scores above 100 are neither proportions nor percentages")
        s = s / 100.0
    return s


def z_bonferroni(k: int, alpha: float = 0.05) -> float:
    """Two-sided normal critical value for k Bonferroni-adjusted comparisons: Phi^-1(1 - alpha / (2 k))."""
    if k < 1:
        raise ValueError("k must be at least 1")
    return _norm.ppf(1.0 - alpha / (2.0 * k))


# --------------------------------------------------------------------------- pairwise quantities

def item_sampling_bound(p1, p2, n, rho=0.0, z=Z95, *, n2=None):
    """Item-sampling bound z * SE_D, in proportion units (Proposition 2 with binomial variances).

    n is the task count of entry 1 (and of entry 2 unless n2 is given). Inputs may be arrays (elementwise).
    """
    p1, p2 = _prop(p1, "p1"), _prop(p2, "p2")
    n1 = _count(n)
    n2 = n1 if n2 is None else _count(n2, "n2")
    return _scalar(z * _se_d(p1, p2, n1, n2, _rho(rho)))


def separable(p1, p2, n, rho=0.0, z=Z95, *, n2=None):
    """True iff the gap is not below the item-sampling bound and is not zero.

    This is the census rule: a pair is below the bound when gap < bound or gap == 0 (equal scores are always
    below it); otherwise it is separable. Below the bound is not evidence that the entries do not differ.
    """
    gap = np.abs(_prop(p1, "p1") - _prop(p2, "p2"))
    b = item_sampling_bound(p1, p2, n, rho, z, n2=n2)
    out = ~((gap < b) | (gap == 0.0))
    return bool(out) if np.ndim(out) == 0 else out


def order_confidence(p1, p2, n, rho=0.0, *, n2=None):
    """c = Phi(gap / SE_D): flat-prior posterior probability (normal approximation) that the higher-scoring
    entry has the higher true score. SE_D = 0: c = 1 for a positive gap, 0.5 for a tie."""
    p1, p2 = _prop(p1, "p1"), _prop(p2, "p2")
    n1 = _count(n)
    n2 = n1 if n2 is None else _count(n2, "n2")
    gap = np.abs(p1 - p2)
    sd = _se_d(p1, p2, n1, n2, _rho(rho))
    with np.errstate(divide="ignore", invalid="ignore"):
        zz = np.where(sd > 0, gap / np.where(sd > 0, sd, 1.0), np.where(gap > 0, np.inf, 0.0))
    return _scalar(_norm.cdf(zz))


def rerun_bound(p1, p2, n, f, k=1, z=Z95, *, k2=None, n2=None):
    """Rerun bound for fixed tasks and independent reruns:

        z * sqrt( f/n * ( P1(1-P1)/k1 + P2(1-P2)/k2 ) ),

    with within-task variance v_j = f P_j (1 - P_j). k is the runs per entry (entry 2 too unless k2 is given).
    Leaderboards do not publish f; the caller must supply it (the paper transports f = 0.369 from
    Terminal-Bench 2.0 as an assumption).
    """
    p1, p2 = _prop(p1, "p1"), _prop(p2, "p2")
    if not 0.0 <= f <= 1.0:
        raise ValueError("f must lie in [0, 1]")
    n1 = _count(n)
    n2 = n1 if n2 is None else _count(n2, "n2")
    k1 = _count(k, "k")
    k2 = k1 if k2 is None else _count(k2, "k2")
    var = f * (p1 * (1.0 - p1) / (n1 * k1) + p2 * (1.0 - p2) / (n2 * k2))
    return _scalar(z * np.sqrt(var))


# --------------------------------------------------------------------------- whole-leaderboard quantities

def _per_entry_n(n, m):
    n = _count(n)
    if n.ndim == 0:
        return np.full(m, float(n))
    if n.shape != (m,):
        raise ValueError("n must be a scalar or have one value per entry")
    return n


def _rank(scores, names=None):
    """Indices best-first; ties by name when names are given, else by input position."""
    if names is not None:
        return sorted(range(len(scores)), key=lambda i: (-scores[i], names[i], i))
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))


def leader_posterior(scores, n, rho=0.0, draws=20000, seed=0):
    """Flat-prior posterior probability that the observed leader has the highest true score.

    True scores ~ N(observed, Sigma) with Sigma_ii = SE_i^2 (binomial, at the observed scores) and
    Sigma_ij = rho SE_i SE_j; draws are clipped to [0, 1]. With tied leading scores the observed leader is
    the tied set: a draw counts when its highest true score belongs to any member. Monte Carlo, so the
    result has a standard error of about sqrt(P(1-P)/draws); fix `seed` for reproducibility.
    """
    mu = _prop(scores, "scores")
    m = mu.size
    if m < 2:
        return 1.0
    n_ = _per_entry_n(n, m)
    rho = float(_rho(rho))
    se = _se(mu, n_)
    rng = np.random.default_rng(seed)
    if 0.0 <= rho < 1.0:
        z0 = rng.standard_normal((draws, 1))
        zi = rng.standard_normal((draws, m))
        noise = se * (math.sqrt(rho) * z0 + math.sqrt(1.0 - rho) * zi)
    else:  # negative or unit rho: general covariance through its eigendecomposition
        cov = rho * np.outer(se, se)
        np.fill_diagonal(cov, se ** 2)
        w, V = np.linalg.eigh(cov)
        noise = (rng.standard_normal((draws, m)) * np.sqrt(np.clip(w, 0.0, None))) @ V.T
    top = np.clip(mu + noise, 0.0, 1.0).argmax(axis=1)
    tied = mu == mu.max()
    return float(tied[top].mean())


def leader_set(scores, n, rho=0.0, alpha=0.05, bonferroni=True):
    """Indices (into `scores`, leader first, then by rank) of entries not separated from the leader,
    including the leader. An entry is in the set when its gap to the leader is below z* SE_D or zero, with
    z* = Phi^-1(1 - alpha / 2) = 1.96 (bonferroni=False) or Phi^-1(1 - alpha / (2 (m - 1))) (bonferroni=True).
    Under the normal approximation with known SEs and rho, the Bonferroni set contains an entry with the
    highest true score with probability at least 1 - alpha / 2.
    """
    s = _prop(scores, "scores")
    m = s.size
    if m == 0:
        return []
    order = _rank(s)
    lead = order[0]
    if m == 1:
        return [lead]
    n_ = _per_entry_n(n, m)
    rho = _rho(rho)
    zstar = z_bonferroni(m - 1, alpha) if bonferroni else _norm.ppf(1.0 - alpha / 2.0)
    out = [lead]
    for j in order[1:]:
        gap = s[lead] - s[j]
        bound = zstar * _se_d(s[lead], s[j], n_[lead], n_[j], rho)
        if gap < bound or gap == 0.0:
            out.append(j)
    return out


def tiers(scores, n, rho=0.0, z=Z95, names: Optional[Sequence[str]] = None):
    """Greedy tiers (Algorithm 1). Entries are ranked best first (ties by name if given, else by position);
    entry 1 opens tier 1, and an entry joins the current tier while it is not separable from the tier's best
    entry (gap < bound or gap == 0). Returns a list of tiers, each a list of indices into `scores`;
    len(result) is the tier count L. Tiers are greedy partitions, not equivalence classes.
    """
    s = _prop(scores, "scores")
    m = s.size
    if m == 0:
        return []
    n_ = _per_entry_n(n, m)
    rho = _rho(rho)
    order = _rank(s, list(names) if names is not None else None)
    out = [[order[0]]]
    head = order[0]
    for j in order[1:]:
        gap = s[head] - s[j]
        bound = z * _se_d(s[head], s[j], n_[head], n_[j], rho)
        if gap < bound or gap == 0.0:
            out[-1].append(j)
        else:
            out.append([j])
            head = j
    return out


# --------------------------------------------------------------------------- report

def _pp(x, d=2):
    return f"{100.0 * x:.{d}f}"


def check_leaderboard(names, scores, n, rho=0.0, *, draws=20000, seed=0, z=Z95, alpha=0.05, max_rows=15):
    """Check a leaderboard from its published scores. `scores` are proportions; `n` is a scalar or per entry.

    Returns a dict with: m, n, rho, order (names best first), top1_vs_2 and top1_vs_5 (gap, bound, separable,
    order_confidence; the latter None when m < 5), leader_set and leader_set_uncorrected (names),
    z_bonferroni, leader_posterior, tiers (lists of names), n_tiers, and `text`, a printable report.
    """
    s = _prop(scores, "scores")
    names = [str(x) for x in names]
    m = s.size
    if len(names) != m:
        raise ValueError("names and scores must have the same length")
    if m < 2:
        raise ValueError("need at least two entries")
    n_ = _per_entry_n(n, m)
    r = _rank(s, names)

    def pair(i, j):
        a, b = r[i], r[j]
        return {"a": names[a], "b": names[b], "gap": float(s[a] - s[b]),
                "bound": float(item_sampling_bound(s[a], s[b], n_[a], rho, z, n2=n_[b])),
                "separable": bool(separable(s[a], s[b], n_[a], rho, z, n2=n_[b])),
                "order_confidence": float(order_confidence(s[a], s[b], n_[a], rho, n2=n_[b]))}

    top12 = pair(0, 1)
    top15 = pair(0, 4) if m >= 5 else None
    ls_b = leader_set(s, n_, rho, alpha, True)
    ls_u = leader_set(s, n_, rho, alpha, False)
    post = leader_posterior(s, n_, rho, draws, seed)
    tr = tiers(s, n_, rho, z, names)
    tier_of = {i: t for t, grp in enumerate(tr, 1) for i in grp}
    zb = z_bonferroni(m - 1, alpha)
    uniform_n = bool(np.all(n_ == n_[0]))

    rep = {
        "m": m, "n": float(n_[0]) if uniform_n else n_.tolist(), "rho": float(rho),
        "order": [names[i] for i in r], "top1_vs_2": top12, "top1_vs_5": top15,
        "leader_set": [names[i] for i in ls_b], "leader_set_uncorrected": [names[i] for i in ls_u],
        "z_bonferroni": zb, "leader_posterior": post,
        "tiers": [[names[i] for i in g] for g in tr], "n_tiers": len(tr),
    }

    L = [f"{m} entries, n = {n_[0]:g}" + ("" if uniform_n else " (varies by entry)") + f", rho = {float(rho):g}"]
    for lab, p in (("#1 vs #2", top12), ("#1 vs #5", top15)):
        if p is None:
            L.append(f"{lab}: needs at least 5 entries")
            continue
        verdict = "separable" if p["separable"] else "below the bound"
        L.append(f"{lab}: gap {_pp(p['gap'])} pp, bound {_pp(p['bound'])} pp -> {verdict}; "
                 f"order confidence {p['order_confidence']:.3f}")
    shown = ", ".join(names[i] for i in ls_b[:10]) + (" ..." if len(ls_b) > 10 else "")
    L.append(f"Leader set (Bonferroni, z* = {zb:.3f}): {len(ls_b)} of {m}: {shown}")
    L.append(f"Leader set (uncorrected, z* = {_norm.ppf(1 - alpha / 2):.3f}): {len(ls_u)} of {m}")
    L.append(f"P(observed leader has the highest true score) = {post:.3f}  (flat prior, {draws} draws, seed {seed})")
    L.append(f"Tiers: {len(tr)}")
    L.append("")
    w = max(4, min(30, max(len(x) for x in names)))
    L.append(f"{'rank':>4}  {'entry':<{w}}  {'score %':>8}  {'gap to #1':>10}  {'bound':>7}  {'vs #1':<16}  {'tier':>4}")
    ld = r[0]
    for k, i in enumerate(r[:max_rows], 1):
        if i == ld:
            gp = bd = vs = "-"
        else:
            gp = _pp(s[ld] - s[i])
            bd = _pp(float(item_sampling_bound(s[ld], s[i], n_[ld], rho, z, n2=n_[i])))
            vs = "separable" if separable(s[ld], s[i], n_[ld], rho, z, n2=n_[i]) else "below the bound"
        L.append(f"{k:>4}  {names[i][:w]:<{w}}  {_pp(s[i]):>8}  {gp:>10}  {bd:>7}  {vs:<16}  {tier_of[i]:>4}")
    if m > max_rows:
        L.append(f"... {m - max_rows} more entries")
    rep["text"] = "\n".join(L)
    return rep
