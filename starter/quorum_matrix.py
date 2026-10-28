"""P1 ATD matrix A4 (+T4/T5): variant plausibility on the full match.

Runs C0/T1/T2/T3/T4/T5 anchor variants through the shared
interpolate->prune->mask->hold_fill pipeline and prints per-variant
full-match plausibility: anchor count, pred-spell count, spell-duration
distribution, D/X shares. No GT contact: counts and shapes only, never
public-spellF1 ranking (w1-rules §2).

Integrity checks (pre-registered, w3-p1quorum §4): T1==T3, T2==T3, and
T4 == C0-T3 exactly (the falsifier must anchor exactly the dropped set).
T5 (Q_MIN=2) is diagnostic-only with no ship path.

No file writes; report goes to stdout.
Run from repo root: python starter/quorum_matrix.py
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


def spell_runs(labels):
    chg = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    bounds = np.concatenate(([0], chg, [len(labels)]))
    out = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        if labels[s] in ("H", "A"):
            out.append(e - s)
    return out


def main():
    try:
        meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME, None)
    except (IOError, KeyError) as error:
        print("quorum_matrix: {}\nRerun with the bundle unpacked "
              "(see DATA.md).".format(error))
        return 0
    n = source_frames("tactical", GAME)
    codes, d1, margin, disp, qh, qa = scan(raw_path,
                                           team_of_each_track(meta_path), n,
                                           GAMES[GAME]["pitch"])
    vis = codes != NOBODY
    dual = (qh >= Q_MIN) & (qa >= Q_MIN)
    strict = (qh >= 2) & (qa >= 2)
    legacy_margin = np.where(vis & ~dual, np.inf, margin)  # exact C0 (audit)
    base_new = (margin > MARGIN) & (d1 < D1_MAX) & (disp < DISP_TAU)
    base_leg = (legacy_margin > MARGIN) & (d1 < D1_MAX) & (disp < DISP_TAU)
    variants = {
        "c0": np.flatnonzero(vis & base_leg),
        "t1": np.flatnonzero(vis & base_new),
        "t2": np.flatnonzero(vis & dual & base_leg),
        "t3": np.flatnonzero(vis & dual & base_new),
        "t4": np.flatnonzero(vis & ~dual & base_leg),
        "t5": np.flatnonzero(vis & strict & base_new),
    }
    print("== integrity (pre-registered) ==")
    print(" T1==T3:", np.array_equal(variants["t1"], variants["t3"]),
          " T2==T3:", np.array_equal(variants["t2"], variants["t3"]))
    dropped = np.setdiff1d(variants["c0"], variants["t3"], assume_unique=True)
    print(" T4 == C0-T3:", np.array_equal(variants["t4"], dropped),
          "(t4 anchors on single-team frames only)")

    letter = np.where(codes == HOME, "H", np.where(vis, "A", "X"))
    alive = np.isin(smooth(codes, 50), ("H", "A"))  # shared: P1 cannot move it
    print("== full-match plausibility ==")
    print(" {:>3} {:>7} {:>7} {:>7} {:>7} {:>7} {:>6} {:>6}".format(
        "var", "anchors", "spells", "med_f", "p10_f", "p90_f", "Dshr",
        "Xshr"))
    for key, anchors in variants.items():
        if len(anchors) == 0:
            print(" {:>3} {:>7} {:>7} {:>7} {:>7} {:>7} {:>6} {:>6}   "
                  "(empty anchors -> all-X)".format(key, 0, 0, "-", "-",
                                                    "-", "-", "-"))
            continue
        team = prune(interpolate(n, anchors, letter[anchors]))
        final = np.where(alive, team, "D")
        for _, a, b in half_video_blocks(GAME):
            final[a:b + 1] = hold_fill(final[a:b + 1])
        durs = spell_runs(final)
        arr = np.array(durs, dtype=float)
        print(" {:>3} {:>7} {:>7} {:>7.0f} {:>7.0f} {:>7.0f} {:>6.3f} {:>6.3f}"
              .format(key, len(anchors), len(durs),
                      np.percentile(arr, 50), np.percentile(arr, 10),
                      np.percentile(arr, 90),
                      np.mean(final == "D"),
                      np.mean(prune(interpolate(
                          n, anchors, letter[anchors])) == "X")))
    print("D-share must be IDENTICAL across variants (shared mask); any drift "
          "is a pipeline bug, not a P1 effect.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
