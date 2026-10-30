"""Shared PR1-track runner for the W3-P2 ATD audits (not a script).

run_pr1_track() reproduces the possession.py pipeline with switches the audits
need (graded vs frozen prune, quorum mirror on/off, proxy touch). It calls
games.require(), so it raises IOError without the gated bundle — callers catch
that and exit cleanly. No GT contact here; A4 reads public GT for a
correctness check only.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from starter.baseline import HOME, NOBODY, smooth, team_of_each_track  # noqa: E402
from starter.possession import (D1_MAX, DISP_TAU, MARGIN, graded_prune,  # noqa: E402
                                interpolate_with_prov, hold_fill, prune,
                                rle_runs, scan)

# Proxy touch (audits only): ball jump >0.5 m/frame with a player within 1.5 m.
# P6 owns the real touch gate; this proxy exists so A1/E7 can run before P6
# lands, and its verdicts are labeled PROXY until it is replaced.
TOUCH_DISP, TOUCH_D1 = 0.5, 1.5


def final_log_records(records):
    """Deduplicate fixpoint-log records by (s, e), keeping first decision.

    ha_graded_pass appends every kept/dropped run on EVERY fixpoint pass,
    so multi-pass runs inflate raw log counts; audits must count runs.
    """
    seen = {}
    for rec in records:
        seen.setdefault((rec[0], rec[1]), rec)
    return list(seen.values())


def proxy_touch(disp, d1):
    # inf marks a missing ball coordinate (reappearance), not a jump.
    return np.isfinite(disp) & (disp > TOUCH_DISP) & (d1 < TOUCH_D1)


def quorum_mirror(margin, vis):
    """Two-sided visibility without new scanning: with best-player present,
    margin=inf holds exactly on single-team frames (w1-pr1code §3), so finite
    margin is the quorum. Mirror of P1 M2; replace with P1 counters when it
    lands (results labeled QUORUM-MIRROR until then)."""
    return vis & np.isfinite(margin)


def run_pr1_track(use_graded=True, use_quorum=False, want_touch=False):
    """Full PR1 pipeline; returns dict(final, team, prov, disp, d1, anchors,
    touch, log). use_graded=False runs the frozen prune() control."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME, None)
    n = source_frames("tactical", GAME)
    codes, d1, margin, disp = scan(raw_path, team_of_each_track(meta_path),
                                   n, GAMES[GAME]["pitch"])[:4]
    vis = codes != NOBODY
    letter = np.where(codes == HOME, "H", np.where(vis, "A", "X"))
    if use_quorum:
        # Merged scan returns qh/qa too (P1); the -inf margin already
        # excludes single-team frames, so no dual predicate is needed.
        gate = vis & (margin > MARGIN) & (d1 < D1_MAX) & (disp < DISP_TAU)
        gate = gate & quorum_mirror(margin, vis)
    else:
        # C0 control: single-team frames (-inf) approved one-sided again,
        # or the off arm is identical to the on arm and A5 is vacuous.
        one_sided = np.where(np.isneginf(margin), np.inf, margin)
        gate = vis & (one_sided > MARGIN) & (d1 < D1_MAX) & (disp < DISP_TAU)
    anchors = np.flatnonzero(gate)
    team, prov = interpolate_with_prov(n, anchors, letter[anchors])
    log = {}
    if use_graded:
        touch = proxy_touch(disp, d1) if want_touch else None
        team = graded_prune(team, prov=prov, touch=touch, log=log)
    else:
        team = prune(team)
        touch = None
    alive = np.isin(smooth(codes, 50), ("H", "A"))
    final = np.where(alive, team, "D")
    for _, a, b in half_video_blocks(GAME):
        final[a:b + 1] = hold_fill(final[a:b + 1])
    return {"final": final, "team": team, "prov": prov, "disp": disp,
            "d1": d1, "anchors": anchors, "touch": touch, "log": log}


def full_video_spells(labels):
    return sum(1 for _, _, lab in rle_runs(labels) if lab in ("H", "A"))
