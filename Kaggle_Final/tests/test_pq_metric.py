"""Acceptance tests 5 and 6 (E0): the PQ replica (Spec S4, S5; Arch section 4.5).

* unit tests on hand-made masks, so every expected number can be checked by hand;
* reference == fast on randomised cases (test 5c);
* human-versus-human reproduces the measured S5 values (test 5b), on the real annotations;
* the "when does an extra prediction raise PQ" rule of S4, re-implemented against this replica (test 6).
"""
import itertools

import numpy as np
import pytest

from filament import pq_metric as pq

HW = 64


def rect(y0, y1, x0, x1, hw=HW):
    m = np.zeros((hw, hw), np.uint8)
    m[y0:y1, x0:x1] = 1
    return m


def run_both(gt, pred, version_image=None):
    """gt: {version: (N,H,W)}, pred: {image: (N,H,W)}; returns (reference, fast) results."""
    vi = version_image or {v: v for v in gt}
    ref = pq.score_reference(gt, pred, vi)
    fast = pq.score_fast({v: pq.masks_to_rles(m) for v, m in gt.items()},
                         {i: pq.masks_to_rles(m) for i, m in pred.items()}, vi)
    return ref, fast


def assert_same(a, b):
    assert (a.tp, a.fp, a.fn) == (b.tp, b.fp, b.fn)
    assert a.pq == pytest.approx(b.pq, abs=1e-12)
    assert a.sum_iou == pytest.approx(b.sum_iou, abs=1e-12)
    assert a.mean_pixel_dice == pytest.approx(b.mean_pixel_dice, abs=1e-12)
    assert a.iou_matched.shape == b.iou_matched.shape
    assert np.allclose(np.sort(a.dice_pairs), np.sort(b.dice_pairs), atol=1e-12)
    assert np.array_equal(np.sort(a.gt_partners), np.sort(b.gt_partners))
    assert np.array_equal(np.sort(a.pred_partners), np.sort(b.pred_partners))


# ------------------------------------------------------------------ hand-checked cases
def test_perfect_prediction_is_one_and_empty_is_zero():
    gt = {"a-img": np.stack([rect(0, 10, 0, 10), rect(20, 30, 20, 40)])}
    ref, fast = run_both(gt, {"img": gt["a-img"]}, {"a-img": "img"})
    assert ref.pq == fast.pq == 1.0 and ref.sq == 1.0 and ref.rq == 1.0
    assert (ref.tp, ref.fp, ref.fn) == (2, 0, 0)
    ref, fast = run_both(gt, {}, {"a-img": "img"})
    assert ref.pq == fast.pq == 0.0
    assert (ref.tp, ref.fp, ref.fn) == (0, 0, 2) == (fast.tp, fast.fp, fast.fn)


def test_hand_computed_case():
    # version has GT A (area 100), B (area 100); image has predictions P (hits A with IoU 0.6) and Q (no GT).
    A, B = rect(0, 10, 0, 10), rect(20, 30, 20, 30)
    P = rect(0, 10, 0, 6) | rect(0, 0, 0, 0)  # area 60 inside A: inter 60, union 100 -> IoU 0.6
    Q = rect(40, 50, 40, 50)
    ref, fast = run_both({"v": np.stack([A, B])}, {"img": np.stack([P, Q])}, {"v": "img"})
    for r in (ref, fast):
        assert (r.tp, r.fp, r.fn) == (1, 1, 1)
        assert r.sum_iou == pytest.approx(0.6)
        assert r.pq == pytest.approx(0.6 / (1 + 0.5 + 0.5))
        assert r.sq == pytest.approx(0.6) and r.rq == pytest.approx(0.5)
        assert r.iou_matched.tolist() == [pytest.approx(0.6)]
    assert_same(ref, fast)


def test_iou_exactly_half_is_not_a_hit():
    A = rect(0, 10, 0, 10)  # area 100
    P = rect(0, 5, 0, 10)  # area 50 inside A: IoU exactly 0.5
    ref, fast = run_both({"v": A[None]}, {"i": P[None]}, {"v": "i"})
    assert (ref.tp, ref.fp, ref.fn) == (0, 1, 1) == (fast.tp, fast.fp, fast.fn)
    P2 = rect(0, 6, 0, 10)  # IoU 0.6
    ref, fast = run_both({"v": A[None]}, {"i": P2[None]}, {"v": "i"})
    assert (ref.tp, ref.fp, ref.fn) == (1, 0, 0) == (fast.tp, fast.fp, fast.fn)


def test_versions_are_scored_separately_and_pooled():
    A = rect(0, 10, 0, 10)
    gt = {"1-img": A[None], "2-img": rect(30, 40, 30, 40)[None]}  # two annotators, same image
    pred = {"img": A[None]}
    ref, fast = run_both(gt, pred, {"1-img": "img", "2-img": "img"})
    # version 1: TP; version 2: FN + FP -> TP 1, FP 1, FN 1, PQ = 1 / 2
    assert (ref.tp, ref.fp, ref.fn) == (1, 1, 1) == (fast.tp, fast.fp, fast.fn)
    assert ref.pq == fast.pq == pytest.approx(0.5)
    assert ref.n_versions == 2 and ref.n_images == 1


def test_prediction_only_images_are_ignored_and_missing_images_are_fn():
    A = rect(0, 10, 0, 10)
    ref, fast = run_both({"v": A[None]}, {"i": A[None], "other": A[None]}, {"v": "i"})
    assert (ref.tp, ref.fp, ref.fn) == (1, 0, 0) == (fast.tp, fast.fp, fast.fn)
    ref, fast = run_both({"v": A[None]}, {"other": A[None]}, {"v": "i"})
    assert (ref.tp, ref.fp, ref.fn) == (0, 0, 1) == (fast.tp, fast.fp, fast.fn)


def test_partner_counts_and_dice_distribution():
    A = rect(0, 10, 0, 10)
    P1, P2 = rect(0, 10, 0, 5), rect(0, 10, 5, 10)  # two predictions split A in two halves (IoU 0.5 each: no hit)
    ref, fast = run_both({"v": A[None]}, {"i": np.stack([P1, P2])}, {"v": "i"})
    assert ref.gt_partners.tolist() == [2] and ref.pred_partners.tolist() == [1, 1]
    assert ref.one_to_many_hist() == {2: 1}
    assert (ref.tp, ref.fp, ref.fn) == (0, 2, 1)
    assert np.allclose(np.sort(ref.dice_pairs), [2 * 0.5 / 1.5] * 2)  # Dice of IoU 0.5 = 2/3
    assert_same(ref, fast)


def test_bootstrap_is_image_level_and_paired_difference_works():
    rng = np.random.default_rng(1)
    gt, pred_good, pred_bad, vi = {}, {}, {}, {}
    for i in range(30):
        m = rect(0, 10, 0, 10)
        for k in range(1 + i % 3):  # 1..3 versions per image
            gt["%d-img%d" % (k, i)] = m[None]
            vi["%d-img%d" % (k, i)] = "img%d" % i
        pred_good["img%d" % i] = m[None]
        pred_bad["img%d" % i] = m[None] if rng.random() < 0.5 else rect(30, 40, 30, 40)[None]
    good = pq.score_reference(gt, pred_good, vi)
    bad = pq.score_reference(gt, pred_bad, vi)
    b = pq.bootstrap_pq(bad, n_boot=200, seed=0)
    assert b["lo"] <= bad.pq <= b["hi"] and b["n_images"] == 30
    d = pq.paired_bootstrap_diff(good, bad, n_boot=200, seed=0)
    assert d["diff"] > 0 and d["lo"] > 0


def test_unique_matching_differs_from_host_only_for_double_hits():
    A = rect(0, 10, 0, 10)
    P1, P2 = rect(0, 10, 0, 8), rect(0, 10, 0, 7)  # two overlapping predictions, both IoU > 0.5 with A
    gt, pred = {"v": A[None]}, {"i": np.stack([P1, P2])}
    host = pq.score_reference(gt, pred, {"v": "i"})
    uniq = pq.score_reference(gt, pred, {"v": "i"}, matching="unique")
    assert (host.tp, host.fp, host.fn) == (2, 0, 0)  # both cells are hits
    assert (uniq.tp, uniq.fp, uniq.fn) == (1, 1, 0)  # only the better one counts; the other is a false positive
    assert uniq.sum_iou == pytest.approx(0.8)
    fast = pq.score_fast({"v": pq.masks_to_rles(A[None])}, {"i": pq.masks_to_rles(np.stack([P1, P2]))},
                         {"v": "i"}, matching="unique")
    assert (fast.tp, fast.fp, fast.fn) == (1, 1, 0)
    with pytest.raises(ValueError):
        pq.score_reference(gt, pred, {"v": "i"}, matching="nope")


# ------------------------------------------------------------------ test 5(c): reference == fast on random cases
def _random_masks(rng, n, hw=48, disjoint=False):
    occupied = np.zeros((hw, hw), bool)
    out = []
    for _ in range(n):
        h, w = rng.integers(3, 20, 2)
        y, x = rng.integers(0, hw - h), rng.integers(0, hw - w)
        m = np.zeros((hw, hw), np.uint8)
        m[y:y + h, x:x + w] = 1
        if disjoint:
            m[occupied] = 0
            if m.sum() == 0:
                continue
            occupied |= m.astype(bool)
        out.append(m)
    return np.stack(out) if out else np.zeros((0, hw, hw), np.uint8)


def test_reference_equals_fast_on_random_cases():
    rng = np.random.default_rng(123)
    n_cases = 400
    n_hits = 0
    for case in range(n_cases):
        gt, pred, vi = {}, {}, {}
        for im in range(int(rng.integers(1, 4))):
            name = "img%d" % im
            if rng.random() < 0.9:  # sometimes an image has no prediction at all
                pred[name] = _random_masks(rng, int(rng.integers(0, 7)), disjoint=rng.random() < 0.7)
            for v in range(int(rng.integers(1, 4))):
                gt["%d-%s" % (v, name)] = _random_masks(rng, int(rng.integers(1, 7)), disjoint=rng.random() < 0.8)
                vi["%d-%s" % (v, name)] = name
        # make some predictions equal or near a GT mask so that hits occur
        for v, m in gt.items():
            name = vi[v]
            if name in pred and rng.random() < 0.6 and len(m):
                near = m[: int(rng.integers(1, len(m) + 1))].copy()
                if rng.random() < 0.5:
                    near = np.roll(near, int(rng.integers(0, 3)), axis=1)
                pred[name] = np.concatenate([pred[name], near]) if len(pred[name]) else near
        ref, fast = run_both(gt, pred, vi)
        assert_same(ref, fast)
        n_hits += ref.tp
    print("\n[E0-5c] reference == fast on %d random cases (%d hits in total)" % (n_cases, n_hits))
    assert n_hits > 100  # the cases did exercise the matching


# ------------------------------------------------------------------ test 6: the PQ rule of S4 against the replica
def _rule_case(rng):
    """Build one image with ``nv`` versions; return (items_before, items_after, matched_iou_sum, n_match, nv).

    Geometry: a grid of 30-px cells; a prediction or GT filament is a 20 x w rectangle in one cell.  A GT matched to
    a prediction is the same rectangle trimmed to height h, so IoU = h / 20.  The extra candidate sits in its own
    cell; in a version it hits a currently-unmatched GT row (placed in the candidate's cell) with probability 0.5.
    """
    cell, G = 30, 6
    nv, n_gt, n_pred = int(rng.integers(1, 4)), int(rng.integers(1, 9)), int(rng.integers(1, 9))
    cand_cell = n_pred
    free_cells = list(range(n_pred + 1, n_pred + 1 + n_gt))

    def cell_rect(c, h=20, w=20):
        y0, x0 = (c // G) * cell, (c % G) * cell
        m = np.zeros((G * cell, G * cell), np.uint8)
        m[y0:y0 + h, x0:x0 + w] = 1
        return m

    preds = [cell_rect(c) for c in range(n_pred)]
    base_gt, after_gt = [], []
    matched_sum, n_match = 0.0, 0
    for _ in range(nv):
        rows, targets = [], []
        perm = list(rng.permutation(n_pred))
        used_pred_cells = set()
        free = list(free_cells)
        for j in range(n_gt):
            r = rng.random()
            if perm and r < 0.5:  # matched to a base prediction with IoU h/20 in [0.55, 1.0]
                p = int(perm.pop())
                used_pred_cells.add(p)
                rows.append(("match", cell_rect(p, h=int(rng.integers(11, 21)))))
            elif r < 0.75:  # a GT row that the candidate may hit: lives in the candidate's cell
                rows.append(("cand", None))
            else:  # unmatched: free cell, or a weak overlap (h <= 10 -> IoU <= 0.5, boundary included) with an unused pred
                unused = [p for p in range(n_pred) if p not in used_pred_cells and p in perm]
                if unused and rng.random() < 0.5:
                    p = int(rng.choice(unused))
                    perm.remove(p)
                    used_pred_cells.add(p)
                    rows.append(("weak", cell_rect(p, h=int(rng.integers(1, 11)))))
                else:
                    rows.append(("free", cell_rect(free.pop())))
        # at most one row can sit in the candidate's cell without overlapping another GT of the version
        cand_rows = [i for i, (kind, _) in enumerate(rows) if kind == "cand"]
        keep = cand_rows[0] if cand_rows else None
        gt_base, gt_after = [], []
        hit_here = False
        for i, (kind, m) in enumerate(rows):
            if kind == "cand":
                if i == keep:
                    h = int(rng.integers(11, 21))
                    m = cell_rect(cand_cell, h=h)
                    hit_here = True
                    matched_sum += h / 20.0
                    n_match += 1
                else:
                    m = cell_rect(free.pop() if free else free_cells[0])
            gt_base.append(m)
            gt_after.append(m)
        if not hit_here:
            pass
        base_gt.append(np.stack(gt_base))
        after_gt.append(np.stack(gt_after))
    cand = cell_rect(cand_cell)
    return base_gt, after_gt, np.stack(preds), np.concatenate([np.stack(preds), cand[None]]), matched_sum, n_match, nv


def test_extra_prediction_rule_2000_random_cases():
    """Spec S4: adding one disjoint prediction changes the denominator by exactly 0.5 per version and the numerator by
    the sum of its matched IoUs; PQ rises iff mean_v(match x IoU) > PQ / 2.  2,000 random cases, 0 violations."""
    rng = np.random.default_rng(0)
    trials, bad_den, bad_num, bad_rule, rises = 2000, 0, 0, 0, 0
    for _ in range(trials):
        base_gt, after_gt, preds0, preds1, msum, nmatch, nv = _rule_case(rng)
        # a GT row whose cell was reused by the "free" fallback may overlap another row of the same version; that is
        # allowed (versions can overlap, S5) and does not change the arithmetic being tested
        gt = {"v%d-img" % v: g for v, g in enumerate(base_gt)}
        vi = {k: "img" for k in gt}
        r0 = pq.score_fast({k: pq.masks_to_rles(g) for k, g in gt.items()}, {"img": pq.masks_to_rles(preds0)}, vi)
        r1 = pq.score_fast({k: pq.masks_to_rles(g) for k, g in gt.items()}, {"img": pq.masks_to_rles(preds1)}, vi)
        d0 = r0.tp + 0.5 * r0.fp + 0.5 * r0.fn
        d1 = r1.tp + 0.5 * r1.fp + 0.5 * r1.fn
        bad_den += abs((d1 - d0) - 0.5 * nv) > 1e-9
        bad_num += abs((r1.sum_iou - r0.sum_iou) - msum) > 1e-9
        crit = (msum / nv) > r0.pq / 2 + 1e-12
        crit_eq = abs(msum / nv - r0.pq / 2) <= 1e-12
        if not crit_eq:
            bad_rule += (r1.pq > r0.pq) != crit
        rises += r1.pq > r0.pq
    print("\n[E0-6] PQ rule: %d random cases; denominator violations %d; numerator violations %d; "
          "rule violations %d; PQ rose in %d cases" % (trials, bad_den, bad_num, bad_rule, rises))
    assert bad_den == 0 and bad_num == 0 and bad_rule == 0
    assert 0 < rises < trials  # both outcomes occurred


# ------------------------------------------------------------------ tests on the real annotations
DUPLICATE_VERSION = "010403-20160810234034Lh"  # contains two filaments with IoU 0.724 to each other (diagnosed in E0)


@pytest.mark.data
def test_own_filaments_as_prediction_score_one(annotations, gt_rles):
    """Test 5(a), on the real data: each version scored against its own filaments gives PQ 1; empty gives 0.

    One version has two near-duplicate filaments (IoU 0.724).  The host's hit-cell counting credits both predictions
    against both GT filaments, so its PQ is below 1 under rule "host"; under one-to-one matching it is 1.
    """
    below = {}
    for v, r in gt_rles.items():
        it = pq.ScoreItem(v, annotations.image_of[v], r, r)
        res = pq.score_items_fast([it])
        if res.pq != 1.0:
            below[v] = (res.pq, res.tp, res.fp, res.fn)
            assert pq.score_items_fast([it], matching="unique").pq == 1.0
    print("\n[E0-5a] own filaments as prediction, all %d versions, rule 'host': %d versions with PQ below 1: %s"
          % (len(gt_rles), len(below), {k: tuple(round(x, 4) for x in t) for k, t in below.items()}))
    print("[E0-5a] same with one-to-one matching: all versions PQ 1.0")
    assert set(below) == {DUPLICATE_VERSION}
    sub = sorted(gt_rles)[::10]
    empty = pq.score_items_fast([pq.ScoreItem(v, annotations.image_of[v], gt_rles[v], []) for v in sub])
    print("[E0-5a] empty prediction, %d versions: PQ %.1f (TP %d FP %d FN %d)"
          % (len(sub), empty.pq, empty.tp, empty.fp, empty.fn))
    assert empty.pq == 0.0 and empty.tp == 0 and empty.fp == 0 and empty.fn == sum(len(gt_rles[v]) for v in sub)


def _human_pairs(annotations, gt_rles):
    """All unordered pairs (v1, v2) of versions of the same image, in JSON order (the analysis used v1 as GT)."""
    pairs = []
    for name in annotations.image_names:
        vs = annotations.versions_of[name]
        pairs += [(name, a, b) for a, b in itertools.combinations(vs, 2)]
    return pairs


@pytest.mark.data
def test_human_pairs_reproduce_spec_values(annotations, gt_rles):
    """Test 5(b): one annotator version against another over all multi-version images (598 pairs; S5).

    S5 values: micro PQ 0.340, SQ 0.634, RQ 0.536, TP 2,144; FP + FN = 3,711 (FP 1,851 and FN 1,860 with the earlier
    version as GT; the directions swap when the pair is reversed; PQ, SQ, RQ, TP equal in both).
    Those values came from one-to-one matching, so they are asserted exactly under matching="unique"; the host rule
    gives the same PQ, SQ, RQ to 3 decimals and TP 2,146 / FP 1,849 (two GT filaments hit twice, see above).
    """
    pairs = _human_pairs(annotations, gt_rles)
    n_images = len({p[0] for p in pairs})
    mk = lambda swap: [pq.ScoreItem("%s|%s" % (a, b), n, gt_rles[b if swap else a], gt_rles[a if swap else b])  # noqa: E731
                       for n, a, b in pairs]
    out = {}
    for matching in ("host", "unique"):
        for lab, swap in (("earlier version = GT", False), ("reversed", True)):
            r = pq.score_items_fast(mk(swap), matching=matching)
            out[(matching, swap)] = r
            print("\n[E0-5b] rule %-6s %d pairs / %d images, %-20s: PQ %.4f SQ %.4f RQ %.4f | TP %d FP %d FN %d (FP+FN %d)"
                  % (matching, len(pairs), n_images, lab, r.pq, r.sq, r.rq, r.tp, r.fp, r.fn, r.fp + r.fn))
    print("[E0-5b] expected (S5): PQ 0.340 SQ 0.634 RQ 0.536 TP 2144 FP+FN 3711 (FP 1851, FN 1860)")
    assert len(pairs) == 598 and n_images == 296
    u, ur, h = out[("unique", False)], out[("unique", True)], out[("host", False)]
    assert (round(u.pq, 3), round(u.sq, 3), round(u.rq, 3)) == (0.340, 0.634, 0.536)
    assert (u.tp, u.fp, u.fn) == (2144, 1851, 1860) and u.fp + u.fn == 3711
    assert (ur.tp, ur.fp, ur.fn) == (2144, 1860, 1851)
    assert (round(h.pq, 3), round(h.sq, 3), round(h.rq, 3)) == (0.340, 0.634, 0.536)
    assert (h.tp, h.fp, h.fn) == (2146, 1849, 1860)


@pytest.mark.data
def test_reference_equals_fast_on_all_human_pairs(annotations, gt_rles):
    """Test 5(c) on real masks: the dense reference gives the same numbers as the RLE implementation (598 pairs)."""
    from pycocotools import mask as M

    pairs = _human_pairs(annotations, gt_rles)
    dense = lambda rl: (np.transpose(M.decode(rl), (2, 0, 1)) if len(rl) else np.zeros((0, 2048, 2048), np.uint8))  # noqa: E731
    tp = fp = fn = 0
    si_ref = si_fast = 0.0
    max_dice_diff = 0.0
    for name, a, b in pairs:
        it_f = pq.ScoreItem("%s|%s" % (a, b), name, gt_rles[a], gt_rles[b])
        it_r = pq.ScoreItem("%s|%s" % (a, b), name, dense(gt_rles[a]), dense(gt_rles[b]))
        f, r = pq.score_items_fast([it_f]), pq.score_items_reference([it_r])
        assert (f.tp, f.fp, f.fn) == (r.tp, r.fp, r.fn), (a, b)
        assert abs(f.sum_iou - r.sum_iou) < 1e-9 and abs(f.mean_pixel_dice - r.mean_pixel_dice) < 1e-9
        assert np.array_equal(np.sort(f.gt_partners), np.sort(r.gt_partners))
        tp, fp, fn, si_ref, si_fast = tp + r.tp, fp + r.fp, fn + r.fn, si_ref + r.sum_iou, si_fast + f.sum_iou
    print("\n[E0-5c] reference == fast on all %d human pairs: TP %d FP %d FN %d, sum IoU %.9f vs %.9f"
          % (len(pairs), tp, fp, fn, si_ref, si_fast))
