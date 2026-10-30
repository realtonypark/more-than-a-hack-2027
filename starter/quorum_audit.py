"""P1 ATD audit A1+A2: margin distribution + C0/T3 anchor diff (needs bundle).

A1: visibility-class histogram (dual/single-H/single-A/none) overall and on
    graded frames; margin stats per class; d1 on dropped anchors. Answers
    w1-pr1code U3 (single-team anchor share).
A2: legacy (C0) vs quorum (T3) anchor sets: counts, drop share, T1==T3 and
    T2==T3 equivalence checks, team-flip blocks >=25f, collapse guard.

No GT contact: label-only. No file writes; report goes to stdout.
Run from repo root: python starter/quorum_audit.py
Without the bundle it prints the missing-data message and exits 0.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from starter.baseline import HOME, NOBODY, smooth, team_of_each_track  # noqa: E402
from starter.possession import (D1_MAX, DISP_TAU, MARGIN, Q_MIN, hold_fill,  # noqa: E402
                                interpolate, prune, scan)

# Pre-registered (w3-p1quorum §8): drop >=25% of anchors -> fall back to the
# -inf guard alone and bench the quorum predicate. Never tuned on GT.
COLLAPSE_FRAC = 0.25
FLIP_MIN = 25  # frames; "wrong-team block" reporting threshold (w2-gapfix H4)


def graded_mask(n):
    mask = np.zeros(n, dtype=bool)
    for _, a, b in half_video_blocks(GAME):
        mask[a:b + 1] = True
    return mask


def pct(q, x):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def main():
    try:
        meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME, None)
    except (IOError, KeyError) as error:
        print("quorum_audit: {}\nRerun with the bundle unpacked "
              "(see DATA.md).".format(error))
        return 0
    n = source_frames("tactical", GAME)
    codes, d1, margin, disp, qh, qa = scan(raw_path,
                                           team_of_each_track(meta_path), n,
                                           GAMES[GAME]["pitch"])[:6]
    vis = codes != NOBODY
    dual = (qh >= Q_MIN) & (qa >= Q_MIN)
    graded = graded_mask(n)

    # Legacy (C0) margin, reconstructed exactly: dual frames carry the same
    # finite margin PR1 computed (same bh/ba); single-team voteless-side
    # frames carried +inf. No code path differs besides the P1 guard.
    legacy_margin = np.where(vis & ~dual, np.inf, margin)
    base_new = (margin > MARGIN) & (d1 < D1_MAX) & (disp < DISP_TAU)
    base_leg = (legacy_margin > MARGIN) & (d1 < D1_MAX) & (disp < DISP_TAU)
    c0 = np.flatnonzero(vis & base_leg)
    t1 = np.flatnonzero(vis & base_new)
    t2 = np.flatnonzero(vis & dual & base_leg)
    t3 = np.flatnonzero(vis & dual & base_new)

    print("== A1 visibility + margin audit (n={}) ==".format(n))
    for name, m in (("all", np.ones(n, dtype=bool)), ("graded", graded)):
        v, d = vis & m, dual & m
        qh_m, qa_m = qh, qa  # full-length: vis_m already carries the mask;
        vis_m = v            # qh[m] compression broke broadcast on "graded"
        print(" [{}] frames={} vis={} dual={} single-H={} single-A={} none={}"
              .format(name, int(np.sum(m)), int(np.sum(vis_m)),
                      int(np.sum(d)),
                      int(np.sum(vis_m & (qh_m > 0) & (qa_m == 0))),
                      int(np.sum(vis_m & (qa_m > 0) & (qh_m == 0))),
                      int(np.sum(m & ~vis))))
    fin = margin[np.isfinite(margin)]
    print(" margin: finite={} (-inf mass={} = single-team {:.3f} of vis)"
          .format(len(fin), int(np.sum(np.isneginf(margin))),
                  np.sum(np.isneginf(margin)) / max(int(np.sum(vis)), 1)))
    print(" margin|finite: p10={:.2f} med={:.2f} p90={:.2f}"
          .format(pct(10, fin), pct(50, fin), pct(90, fin)))
    print(" margin|dual:   p10={:.2f} med={:.2f} p90={:.2f}"
          .format(pct(10, margin[vis & dual]), pct(50, margin[vis & dual]),
                  pct(90, margin[vis & dual])))

    print("== A2 C0 vs T3 ==")
    print(" anchors: C0={} T3={} dropped={} ({:.3f})"
          .format(len(c0), len(t3), len(c0) - len(t3),
                  (len(c0) - len(t3)) / max(len(c0), 1)))
    print(" T1==T3: {}  T2==T3: {}".format(np.array_equal(t1, t3),
                                           np.array_equal(t2, t3)))
    dropped = np.setdiff1d(c0, t3, assume_unique=True)
    print(" dropped-frame d1: p10={:.2f} med={:.2f} p90={:.2f}"
          .format(pct(10, d1[dropped]), pct(50, d1[dropped]),
                  pct(90, d1[dropped])) if len(dropped) else " dropped: none")

    letter = np.where(codes == HOME, "H", np.where(vis, "A", "X"))
    alive = np.isin(smooth(codes, 50), ("H", "A"))
    finals = {}
    for key, anchors in (("c0", c0), ("t3", t3)):
        if len(anchors) == 0:
            finals[key] = np.full(n, "X", dtype="<U1")
            continue
        team = prune(interpolate(n, anchors, letter[anchors]))
        final = np.where(alive, team, "D")
        for _, a, b in half_video_blocks(GAME):
            final[a:b + 1] = hold_fill(final[a:b + 1])
        finals[key] = final
    diff = finals["c0"] != finals["t3"]
    flips = []
    if diff.any():
        chg = np.flatnonzero(np.diff(diff.astype(np.int8)))
        bounds = np.concatenate(([0], chg + 1, [n]))
        for s, e in zip(bounds[:-1], bounds[1:]):
            if diff[s] and e - s >= FLIP_MIN:
                flips.append((int(s), int(e - s), str(finals["c0"][s]),
                              str(finals["t3"][s])))
    print(" label flips C0->T3: {} frames in {} runs >=25f"
          .format(int(np.sum(diff)), len(flips)))
    for s, ln, a, b in flips[:20]:
        print("   f{} len={} {}->{}".format(s, ln, a, b))

    drop_frac = (len(c0) - len(t3)) / max(len(c0), 1)
    if drop_frac >= COLLAPSE_FRAC:
        print("VERDICT: COLLAPSE (drop {:.3f} >= {:.2f}) -> fall back to "
              "-inf guard alone, bench quorum predicate.".format(drop_frac,
                                                                 COLLAPSE_FRAC))
    else:
        print("VERDICT: guard holds (drop {:.3f} < {:.2f}); T3 stands."
              .format(drop_frac, COLLAPSE_FRAC))
    return 0


if __name__ == "__main__":
    sys.exit(main())
