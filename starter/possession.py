"""Frozen winner: anchor-interpolate team (B-v2) + hold-last X fill.

B-v2 (anchors M=3m, gaps G=100, prune <12f) under a smooth50 alive mask, X
hold-filled inside graded halves, passthrough outside. C-v1 bridge/hysteresis
selected a no-op (277->277 spells), so fill alone reproduces the winner.

W3-P2 (branch w3-p2-antiprune): prune() replaced by graded_prune() — a D-first
two-pass RLE prune with a GAP-sourced differ-flank exemption (short-spell
rescue), an alternation test, and a D-share floor. Bands pre-registered in the
phase-3a design before any scored run; see the *_THR constants. prune() is kept
as the frozen reference. main() runs the graded path.

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

# --- W3-P2 pre-registered bands (frozen in the phase-3a design; non-GT priors) ---
# 20f H/A flicker floor: PRUNE=12 precedent widened to the sub-second kick-cycle
# (touch-to-touch <0.8 s implausible as separate possessions). GT P10=19f was
# explicitly NOT used to set this; P10 stays unrescued by design.
HA_FLICKER_THR = 20
# 25f D flicker floor: ball-out-of-view blips <1 s are tracking loss; stoppage
# recognition needs >=1 s of stillness.
D_FLICKER_THR = 25
# 100f contest ceiling: reuses frozen GAP, the pipeline's own bridge horizon.
CONTEST_THR = 100
# 150f D protection: below the shortest plausible genuine stoppage (throw-in
# routine >=6 s incl. retrieval); shorter ball-missing episodes are tracking loss.
D_PROTECT_THR = 150
# D-share floor: football-literature stoppage share is 0.15-0.25; the floor takes
# the lower bound (NOT the public 0.196, which would be GT-derived).
D_SHARE_FLOOR = 0.15
# Alternation flank minimum: same kick-cycle prior as the H/A floor.
ALT_FLANK_THR = 20

# Provenance codes from interpolate_with_prov (PR1 track only).
PROV_ANCHOR, PROV_GAPFILL, PROV_EDGE, PROV_VOTE = 0, 1, 2, 3


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


def interpolate(n, anchors, teams, prov=None):
    """Gap fill between anchors; X core in long/differing gaps.

    When prov (int array, length n) is given, each painted frame is also tagged:
    PROV_GAPFILL for interior spans with ln <= GAP (same-team fill and midpoint
    splits alike), PROV_EDGE for edge spans and 50f blocks, PROV_ANCHOR for
    anchor frames. Labels are identical with or without prov.
    """
    out = np.full(n, "X", dtype="<U1")
    out[anchors] = teams
    h = GAP // 2
    if prov is not None:
        prov[:] = PROV_EDGE
        prov[anchors] = PROV_ANCHOR

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
            if prov is not None:
                prov[s:e] = PROV_GAPFILL
        else:
            out[s:s + h] = left
            out[e - h:e] = right

    paint(0, anchors[0], None, teams[0])
    for (s, lt), (e, rt) in zip(zip(anchors, teams), zip(anchors[1:], teams[1:])):
        paint(s + 1, e, lt, rt)
    paint(anchors[-1] + 1, n, teams[-1], None)
    return out


def interpolate_with_prov(n, anchors, teams):
    """interpolate() plus the per-frame provenance array for graded_prune."""
    prov = np.empty(n, dtype=np.int8)
    return interpolate(n, anchors, teams, prov=prov), prov


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


def rle_runs(out):
    """Run-length encode a label array into [(start, end, label)]."""
    chg = np.flatnonzero(out[1:] != out[:-1]) + 1
    b = np.concatenate(([0], chg, [len(out)]))
    return [(int(b[i]), int(b[i + 1]), out[b[i]]) for i in range(len(b) - 1)]


def _flanking_ha(runs, i):
    """Nearest H/A run indices left/right of run i, skipping X/D (PR1 P:153-6)."""
    prev = next((j for j in range(i - 1, -1, -1) if runs[j][2] in ("H", "A")),
                None)
    nxt = next((j for j in range(i + 1, len(runs)) if runs[j][2] in ("H", "A")),
               None)
    return prev, nxt


def _longer_side(runs, i):
    """Nearest non-X run each side; return (label or None) of the longer one.

    Tie goes left (PR1 P:166 convention); a lone existing side wins; no
    neighbor at all keeps the run (cannot prune the whole stream).
    """
    left = next((j for j in range(i - 1, -1, -1) if runs[j][2] != "X"), None)
    right = next((j for j in range(i + 1, len(runs)) if runs[j][2] != "X"),
                 None)
    if left is None:
        return runs[right][2] if right is not None else None
    if right is None:
        return runs[left][2]
    llen = runs[left][1] - runs[left][0]
    rlen = runs[right][1] - runs[right][0]
    return runs[left][2] if llen >= rlen else runs[right][2]


def _alive_prefer(runs, i):
    """Drop target preferring an H/A neighbor over D (w1-postproc §3).

    Both neighbors H/A, or neither H/A: longer wins, tie left. Lone side wins.
    """
    left = next((j for j in range(i - 1, -1, -1) if runs[j][2] != "X"), None)
    right = next((j for j in range(i + 1, len(runs)) if runs[j][2] != "X"),
                 None)
    if left is None:
        return runs[right][2] if right is not None else None
    if right is None:
        return runs[left][2]
    l_alive = runs[left][2] in ("H", "A")
    r_alive = runs[right][2] in ("H", "A")
    if l_alive != r_alive:
        return runs[left][2] if l_alive else runs[right][2]
    llen = runs[left][1] - runs[left][0]
    rlen = runs[right][1] - runs[right][0]
    return runs[left][2] if llen >= rlen else runs[right][2]


def _gap_sourced(prov, s, e):
    """True iff run [s:e] sits in inter-anchor gap(s) of length <= GAP.

    PROV_GAPFILL is painted only on interior spans with ln <= GAP, so every
    member gap qualifies by construction. Anchor frames are admitted (midpoint
    halves abut their anchor); X-core/edge/50f-block frames disqualify.
    """
    seg = prov[s:e]
    return bool(np.any(seg == PROV_GAPFILL)
                and np.all((seg == PROV_GAPFILL) | (seg == PROV_ANCHOR)))


def flat_prune(out, thr, policy="alive", classes=("H", "A", "D")):
    """Flat run prune on H/A/D label streams (w1-postproc control recipe).

    Runs shorter than thr merge into a neighbor: edge runs into the existing
    side; policy "alive" prefers an H/A neighbor over D, else the longer
    neighbor with ties left; policy "longer" always takes the longer side,
    ties left. Decisions read one pre-pass RLE snapshot.
    """
    work = out.copy()
    runs = rle_runs(work)
    for i, (s, e, lab) in enumerate(runs):
        if e - s >= thr or lab not in classes:
            continue
        left = runs[i - 1][2] if i > 0 else None
        right = runs[i + 1][2] if i + 1 < len(runs) else None
        if left is None:
            rep = right
        elif right is None:
            rep = left
        elif policy == "alive" and (left in ("H", "A")) != (right in ("H", "A")):
            rep = left if left in ("H", "A") else right
        else:
            llen = runs[i - 1][1] - runs[i - 1][0]
            rlen = runs[i + 1][1] - runs[i + 1][0]
            rep = left if llen >= rlen else right
        if rep is not None:
            work[s:e] = rep
    return work


def d_pass_prune(out, confirmed_d=None):
    """Graded D pass: D<25 dies into the longer side, D 25..150 bridges only
    same-team flanks, D>150 and ST/ball-out-confirmed runs survive. Returns
    (labels, dropped_d_runs) — the dropped list feeds the D-share floor."""
    work = out.copy()
    runs = rle_runs(work)
    dropped = []
    for i, (s, e, lab) in enumerate(runs):
        if lab != "D":
            continue
        length = e - s
        if length > D_PROTECT_THR:
            continue  # true stoppages are never touched
        if confirmed_d is not None and np.any(confirmed_d[s:e]):
            continue
        if length < D_FLICKER_THR:
            rep = _longer_side(runs, i)
            if rep is not None and rep != "D":
                work[s:e] = rep
                dropped.append((s, e))
        else:  # D 25..150: bridge only across same-team flanks
            left = runs[i - 1][2] if i > 0 else None
            right = runs[i + 1][2] if i + 1 < len(runs) else None
            if left in ("H", "A") and left == right:
                work[s:e] = left
                dropped.append((s, e))
    return work, dropped


def ha_graded_pass(out, prov=None, touch=None, keep=("differ-flank", "touch",
                   "alternation"), cap_source=True, floor_only=False, log=None):
    """Graded H/A pass: H/A<20 dies into the longer side; the 20..100 contest
    band defaults to DROP and keeps only on the enabled evidence rules.

    keep: subset of {"differ-flank", "touch", "alternation"}.
    cap_source: the differ-flank rule additionally requires GAP provenance
      (without prov, i.e. on label streams, it cannot fire when capped).
    floor_only: skip the contest band entirely (E2 ablation: keep all >= 20).
    Iterated to a fixpoint (cap 10); see graded_prune for why.
    """
    if log is None:
        log = {}
    work = out.copy()
    if prov is None:
        prov = np.full(len(work), PROV_VOTE, dtype=np.int8)
    contest_keep, contest_drop, ha_iters = [], [], 0
    for _ in range(10):
        runs = rle_runs(work)
        changed = False
        for i, (s, e, lab) in enumerate(runs):
            if lab not in ("H", "A"):
                continue
            length = e - s
            if length > CONTEST_THR or (floor_only and length >= HA_FLICKER_THR):
                continue
            if length < HA_FLICKER_THR:
                rep = _longer_side(runs, i)
                if rep is not None and rep != lab:
                    work[s:e] = rep
                    changed = True
                continue
            # Contest band 20..100: default DROP, keep only on evidence.
            prev, nxt = _flanking_ha(runs, i)
            reason = None
            if ("differ-flank" in keep and prev is not None and nxt is not None
                    and runs[prev][2] != runs[nxt][2]
                    and (not cap_source or _gap_sourced(prov, s, e))):
                reason = "differ-flank"  # g-H1 exemption, GAP-source capped
            elif "touch" in keep and touch is not None and np.any(touch[s:e]):
                reason = "touch"
            elif ("alternation" in keep and prev is not None and nxt is not None
                    and runs[prev][2] == runs[nxt][2]
                    and runs[prev][2] != lab
                    and runs[prev][1] - runs[prev][0] >= ALT_FLANK_THR
                    and runs[nxt][1] - runs[nxt][0] >= ALT_FLANK_THR):
                reason = "alternation"  # genuine A-H-A, not a flicker chain
            if reason is not None:
                contest_keep.append((s, e, lab, reason))
                continue
            rep = _alive_prefer(runs, i)
            if rep is not None and rep != lab:
                work[s:e] = rep
                changed = True
            contest_drop.append((s, e, lab))
        ha_iters += 1
        if not changed:
            break
    log["contest_keep"] = contest_keep
    log["contest_drop"] = contest_drop
    log["ha_iters"] = ha_iters
    return work


def graded_prune(out, prov=None, touch=None, confirmed_d=None, log=None,
                 apply_floor=True, keep=("differ-flank", "touch",
                 "alternation"), cap_source=True):
    """D-first two-pass RLE prune with differ-flank rescue (W3-P2).

    out: <U1 label array (H/A/X on the PR1 track; H/A/D on label streams).
    prov: per-frame provenance (interpolate_with_prov); None means a pure
      label stream, where the GAP-source exemption cannot fire.
    touch: optional boolean array; a contest run with any touch frame inside
      is kept (P6 owns the detector; this is just the interface).
    confirmed_d: optional boolean array; D runs with any confirmed frame
      (ST/ball-out evidence) survive the D pass.
    log: optional dict collecting dropped_d_runs, contest_keeps/drops and
      d_restored for the ATD audits.
    apply_floor: False skips the D-share restore (ablation arms E1-E5 run
      floorless; the floor belongs to the full E6 module).
    keep: subset of {"differ-flank", "touch", "alternation"} enabled as
      contest keep rules (ablation arms narrow this; full module enables all).
    cap_source: the differ-flank rule additionally requires GAP provenance.

    All decisions in a pass read a pre-pass RLE snapshot and apply at once.
    The H/A pass iterates to a fixpoint (cap 10): one-shot simultaneous apply
    lets adjacent shorts swap teams instead of dying (A8-H8 -> H8-A8), while
    iteration merges them away. Relabeling whole runs can only merge spans,
    never split them, so the run count falls monotonically and the loop ends.
    """
    if log is None:
        log = {}
    # Pass 1: D runs, length-dependent. Vacuous on the PR1 track (no D runs).
    work, dropped_d = d_pass_prune(out, confirmed_d)
    log["dropped_d_runs"] = dropped_d
    # Pass 2: H/A runs on the recomputed RLE, iterated to a fixpoint.
    work = ha_graded_pass(work, prov, touch, keep, cap_source,
                          floor_only=False, log=log)

    # D-share floor: restore longest dropped D runs until share >= floor.
    restored = []
    if apply_floor and dropped_d and np.mean(work == "D") < D_SHARE_FLOOR:
        for s, e in sorted(dropped_d, key=lambda se: se[0] - se[1]):
            work[s:e] = "D"
            restored.append((s, e))
            if np.mean(work == "D") >= D_SHARE_FLOOR:
                break
    log["d_restored"] = restored
    return work


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
    team, prov = interpolate_with_prov(n, anchors, letter[anchors])
    team = graded_prune(team, prov=prov)  # W3-P2: graded prune, GAP exemption
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
