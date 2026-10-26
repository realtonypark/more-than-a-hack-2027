"""Frozen winner: anchor-interpolate team (B-v2) + hold-last X fill.

B-v2 (anchors M=3m, gaps G=100, prune <12f) under a smooth50 alive mask, X
hold-filled inside graded halves, passthrough outside. C-v1 bridge/hysteresis
selected a no-op (277->277 spells), so fill alone reproduces the winner.

Run from repo root: .venv/bin/python starter/possession.py -> results CSV.
numpy-only; reuses starter/baseline.py team map + smoother.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from starter.baseline import HOME, NOBODY, smooth, team_of_each_track  # noqa: E402

# Frozen winners (M, G swept {1,2,3}x{25,50,100}; M3/G100 closest to ~300
# spells at -74% flicker). Short spells (C2 1/6) and swaps (2/12) are the
# residue.
# ponytail: swap recall 2/12 needs video/tracklet signal; no tracking gate
# separated flicker from events in three rescue rounds, so none is attempted.
MARGIN, GAP, PRUNE = 3.0, 100, 12
D1_MAX, DISP_TAU = 3.0, 0.2  # m; m/frame (ATD ball speed is 91% NaN)
# N=25 frozen: holdout linear-interp p90 error 0.36/1.36/4.57m at gaps 10/25/50
# (round-5 calibration); 25 is the largest gap with p90 < 2m team scale.
GAP_N = 25
OUT = os.path.join(ROOT, "evaluation", "results", "possession_anchors.csv")


def scan(raw_path, team_of_track, n, pitch):
    """Raw nearest-team codes + d1/margin/ball-displacement per frame.

    Ball track is pre-read so bracketed missing runs <=GAP_N get linear-interp
    x/y and full votes; 26..250f runs stay voteless (gap-fill hold/X); longer
    and unbracketed runs are untouched.
    """
    length, width = pitch
    nx = np.full(n, np.nan)
    ny = np.full(n, np.nan)
    with open(raw_path) as handle:  # light ball-only pass for the track
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3 or parts[2][0] == "N":
                continue
            frame = int(parts[0]) - 1
            if 0 <= frame < n:
                c1 = parts[2].index(",")
                c2 = parts[2].index(",", c1 + 1)
                nx[frame] = float(parts[2][:c1])
                ny[frame] = float(parts[2][c1 + 1:c2])
    tracked = ~np.isnan(nx)
    ix = np.full(n, np.nan)  # interp ball x/y (normalized), NaN = no fill
    iy = np.full(n, np.nan)
    edges = np.flatnonzero(np.diff(np.concatenate(([True], tracked, [True]))))
    for i in range(0, len(edges), 2):  # each missing run with both brackets
        s, e = edges[i], edges[i + 1]
        if e - s > GAP_N or s == 0 or e == n:
            continue
        t = np.arange(s, e)
        ix[s:e] = nx[s - 1] + (nx[e] - nx[s - 1]) * (t - s + 1) / (e - s + 1)
        iy[s:e] = ny[s - 1] + (ny[e] - ny[s - 1]) * (t - s + 1) / (e - s + 1)
    codes = np.full(n, NOBODY, dtype=np.int8)
    d1 = np.full(n, np.inf)
    margin = np.full(n, np.inf)
    bx = np.full(n, np.nan)
    by = np.full(n, np.nan)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1
            if parts[2][0] != "N":
                c1 = parts[2].index(",")
                c2 = parts[2].index(",", c1 + 1)
                px = float(parts[2][:c1]) * length
                py = float(parts[2][c1 + 1:c2]) * width
            elif not np.isnan(ix[frame]):
                px, py = ix[frame] * length, iy[frame] * width
            else:
                continue
            best, team, bh, ba = None, NOBODY, np.inf, np.inf
            for slot, group in enumerate(parts[1].split(";")):
                if group[0] == "N":
                    continue
                c = group.index(",")
                dx = float(group[:c]) * length - px
                dy = float(group[c + 1:group.index(",", c + 1)]) * width - py
                dist = dx * dx + dy * dy
                if team_of_track[slot] == HOME:
                    bh = min(bh, dist)
                else:
                    ba = min(ba, dist)
                if best is None or dist < best:
                    best, team = dist, team_of_track[slot]
            if best is None:
                continue
            codes[frame] = team
            d1[frame] = float(np.sqrt(best))
            margin[frame] = abs(float(np.sqrt(bh) - np.sqrt(ba)))
            bx[frame], by[frame] = px, py
    disp = np.full(n, np.inf)
    ok = ~np.isnan(bx[1:]) & ~np.isnan(bx[:-1])
    disp[1:][ok] = np.hypot(np.diff(bx)[ok], np.diff(by)[ok])
    return codes, d1, margin, disp


def interpolate(n, anchors, teams):
    """Gap fill between anchors; X core in long/differing gaps."""
    out = np.full(n, "X", dtype="<U1")
    out[anchors] = teams
    h = GAP // 2

    def paint(s, e, left, right):
        ln = e - s
        if ln <= 0:
            return
        if left is None or right is None:
            edge = right if left is None else left
            at = e - h if left is None else s
            out[s:e] = edge if ln <= GAP else "X"
            if ln > GAP:
                out[at:at + h] = edge
        elif ln <= GAP:
            out[s:e] = left if left == right else "X"
            if left != right:
                out[s:s + ln // 2] = left
                out[s + ln // 2:e] = right
        else:
            out[s:s + h] = left
            out[e - h:e] = right

    paint(0, anchors[0], None, teams[0])
    for (s, lt), (e, rt) in zip(zip(anchors, teams), zip(anchors[1:], teams[1:])):
        paint(s + 1, e, lt, rt)
    paint(anchors[-1] + 1, n, teams[-1], None)
    return out


def prune(out):
    """Reassign H/A runs <PRUNE by neighbor majority (tie -> left)."""
    chg = np.flatnonzero(out[1:] != out[:-1]) + 1
    b = np.concatenate(([0], chg, [len(out)]))
    runs = [(int(b[i]), int(b[i + 1]), out[b[i]]) for i in range(len(b) - 1)]
    for i, (s, e, lab) in enumerate(runs):
        if lab not in ("H", "A") or e - s >= PRUNE:
            continue
        prev = next((runs[j][2] for j in range(i - 1, -1, -1)
                     if runs[j][2] in ("H", "A")), None)
        nxt = next((runs[j][2] for j in range(i + 1, len(runs))
                    if runs[j][2] in ("H", "A")), None)
        if prev is None:
            rep = nxt
        elif nxt is None or prev == nxt:
            rep = prev
        else:
            lp = next(runs[j][1] - runs[j][0] for j in range(i - 1, -1, -1)
                      if runs[j][2] == prev)
            ln = next(runs[j][1] - runs[j][0] for j in range(i + 1, len(runs))
                      if runs[j][2] == nxt)
            rep = prev if lp >= ln else nxt
        if rep is not None:
            out[s:e] = rep
    return out


def hold_fill(w):
    """Hold-last X fill for one half; cf. possession_metrics.fill_abstentions
    (reimplemented: that import pulls pandas, this file stays numpy-only)."""
    idx = np.arange(len(w))
    carry = np.maximum.accumulate(np.where(w != "X", idx, -1))
    known = np.flatnonzero(w != "X")
    if not len(known):
        return w
    carry[carry < 0] = known[0]
    return w[carry]


def main():
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME, None)
    n = source_frames("tactical", GAME)
    codes, d1, margin, disp = scan(raw_path, team_of_each_track(meta_path), n,
                                   GAMES[GAME]["pitch"])
    vis = codes != NOBODY
    letter = np.where(codes == HOME, "H", np.where(vis, "A", "X"))
    anchors = np.flatnonzero(vis & (margin > MARGIN) & (d1 < D1_MAX)
                             & (disp < DISP_TAU))
    team = prune(interpolate(n, anchors, letter[anchors]))
    alive = np.isin(smooth(codes, 50), ("H", "A"))  # mask only, not teams
    final = np.where(alive, team, "D")
    for _, a, b in half_video_blocks(GAME):  # hold-fill halves; passthrough out
        final[a:b + 1] = hold_fill(final[a:b + 1])
    with open(OUT, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(final))
    print("wrote {} ({} rows)".format(os.path.relpath(OUT, ROOT), len(final)))


if __name__ == "__main__":
    main()
