"""Frozen winner: anchor-interpolate team (B-v2) + hold-last X fill.

B-v2 (anchors M=3m, gaps G=100, prune <12f) under a smooth50 alive mask, X
hold-filled inside graded halves, passthrough outside. C-v1 bridge/hysteresis
selected a no-op (277->277 spells), so fill alone reproduces the winner.

Run from repo root: .venv/bin/python starter/possession.py -> results CSV.
numpy-only; reuses starter/baseline.py team map + smoother.

P4 restart (w3-p4-restart): optional --trap (kickoff formation+stillness forces
D until first touch, fires-twice gated) and --restart (PZ-winner re-seed anchors
after D gaps, K=5 confirmed). Both default off; flags off == PR1 bit-identical.
"""

import argparse
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
# P4 restart, phase-3a pre-registered and frozen (Laws 9.15m circle + T1/T8
# literature scales; never tuned on GT; see /tmp/w3-p4restart.md section 2).
K_RESEED = 5  # confirmed PZ votes to mint a re-seed anchor
RPZ = 1.0  # possession-zone radius (m)
RESTART_WINDOW = 75  # post-gap search window (3 s at 25 fps)
FALLBACK_RADIUS, FALLBACK_QUORUM = 5.0, 0.6
MIN_FB_SLOTS = 2  # in-radius slots needed for a fallback decision (anti-churn)
ABSTAIN_LEN = 25  # dead-heat X window (1 s)
RESTART_LOOKBACK = 1500  # restart-point search-back bound (60 s)
TOUCH_SNAP = 3.0  # snap near-boundary restart points to the line (m)
R_CENTER, RING_MED = 10.0, 8.0  # ball-inside / median-slot-outside radii (m)
V_LO, V_COVER, STILL_HALF, STILL_QUORUM = 1.0, 1.5, 12, 0.6
SUSTAIN, KICK_SEP, KICK_CAP = 50, 25, 3750  # cand length / onset sep / touch cap
D_MIN = 12  # D runs shorter than this are flicker, not stoppages
OUT = os.path.join(ROOT, "evaluation", "results", "possession_anchors.csv")


def scan(raw_path, team_of_track, n, pitch):
    """Raw nearest-team codes + d1/margin/ball-displacement per frame.

    Ball track is pre-read so bracketed missing runs <=GAP_N get linear-interp
    x/y and full votes; 26..250f runs stay voteless (gap-fill hold/X); longer
    and unbracketed runs are untouched.

    Also records per-frame P4 aggregates: ball-tracked flag, per-team visible
    slot counts (two-sided quorum), slot-speed sum/count/slow-count (stillness),
    and median slot distance to the center spot (kickoff ring, NaN = no slots).
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
    nh = np.zeros(n, dtype=np.int16)
    na = np.zeros(n, dtype=np.int16)
    spd = np.zeros(n, dtype=np.float32)
    spdn = np.zeros(n, dtype=np.int16)
    slow = np.zeros(n, dtype=np.int16)
    ring = np.full(n, np.nan)
    cx, cy = length / 2, width / 2
    near_c = tracked & (np.hypot(nx * length - cx, ny * width - cy) < R_CENTER)
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
            h_f, a_f, s_sum, s_n, s_slow, rds = 0, 0, 0.0, 0, 0, []
            want_ring = near_c[frame]
            for slot, group in enumerate(parts[1].split(";")):
                if group[0] == "N":
                    continue
                c = group.index(",")
                c2 = group.index(",", c + 1)
                dx = float(group[:c]) * length - px
                dy = float(group[c + 1:c2]) * width - py
                dist = dx * dx + dy * dy
                if team_of_track[slot] == HOME:
                    bh = min(bh, dist)
                    h_f += 1
                else:
                    ba = min(ba, dist)
                    a_f += 1
                sp = float(group[c2 + 1:])
                if sp == sp:  # NaN speed: visible for quorum/ring, out of stillness
                    s_sum += sp
                    s_n += 1
                    s_slow += sp < V_COVER
                if want_ring:
                    rds.append((dx + px - cx) ** 2 + (dy + py - cy) ** 2)
                if best is None or dist < best:
                    best, team = dist, team_of_track[slot]
            if best is None:
                continue
            codes[frame] = team
            d1[frame] = float(np.sqrt(best))
            margin[frame] = abs(float(np.sqrt(bh) - np.sqrt(ba)))
            bx[frame], by[frame] = px, py
            nh[frame], na[frame] = h_f, a_f
            spd[frame], spdn[frame], slow[frame] = s_sum, s_n, s_slow
            if rds:
                ring[frame] = float(np.sqrt(np.median(rds)))
    disp = np.full(n, np.inf)
    ok = ~np.isnan(bx[1:]) & ~np.isnan(bx[:-1])
    disp[1:][ok] = np.hypot(np.diff(bx)[ok], np.diff(by)[ok])
    return codes, d1, margin, disp, tracked, nh, na, spd, spdn, slow, ring, bx, by


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


def hold_fill(w, back_cap=None):
    """Hold-last X fill for one half; cf. possession_metrics.fill_abstentions
    (reimplemented: that import pulls pandas, this file stays numpy-only).
    back_cap bounds the leading back-fill to that many frames (C1 control)."""
    idx = np.arange(len(w))
    carry = np.maximum.accumulate(np.where(w != "X", idx, -1))
    known = np.flatnonzero(w != "X")
    if not len(known):
        return w
    carry[carry < 0] = known[0]
    out = w[carry]
    if back_cap is not None:
        out[:max(0, known[0] - back_cap)] = "X"
    return out


def stillness(spd_sum, slow_cnt, vis_cnt):
    """Collective stillness per frame over a +-STILL_HALF window (d-H2 values)."""
    n = len(spd_sum)
    idx = np.arange(n)
    lo = np.maximum(0, idx - STILL_HALF)
    hi = np.minimum(n, idx + STILL_HALF + 1)
    c_sum = np.concatenate(([0.0], np.cumsum(spd_sum, dtype=np.float64)))
    c_slow = np.concatenate(([0], np.cumsum(slow_cnt)))
    c_vis = np.concatenate(([0], np.cumsum(vis_cnt)))
    tot = c_vis[hi] - c_vis[lo]
    ok = tot > 0
    mean = np.full(n, np.inf)
    frac = np.zeros(n)
    mean[ok] = (c_sum[hi[ok]] - c_sum[lo[ok]]) / tot[ok]
    frac[ok] = (c_slow[hi[ok]] - c_slow[lo[ok]]) / tot[ok]
    return ok & (mean < V_LO) & (frac >= STILL_QUORUM)


def _center_dist(tracked, bx, by, pitch):
    dist_c = np.full(len(bx), np.inf)
    dist_c[tracked] = np.hypot(bx[tracked] - pitch[0] / 2, by[tracked] - pitch[1] / 2)
    return dist_c


def kickoff_events(tracked, bx, by, disp, still, ring, halves, pitch):
    """Ungated kickoff spans [(k0, k1)] plus voided onsets [k0]. Pure ATD function.

    Candidate = ball visible inside R_CENTER and stationary (Laws: the ball is
    stationary at kickoff, which also rejects center transits) + collective
    stillness + median slot ring outside RING_MED, sustained SUSTAIN frames.
    Touch = finite disp > DISP_TAU x3 (motion onset) or ball exits R_CENTER;
    no touch within KICK_CAP voids the onset (fail-safe: no forcing).
    """
    n = len(bx)
    dist_c = _center_dist(tracked, bx, by, pitch)
    cand = (tracked & (dist_c < R_CENTER) & (disp < DISP_TAU) & still
            & (ring >= RING_MED))
    chg = np.flatnonzero(np.diff(np.concatenate(([False], cand, [False]))))
    onsets = []
    for i in range(0, len(chg), 2):
        s, e = int(chg[i]), int(chg[i + 1])
        if e - s >= SUSTAIN and (not onsets or s - onsets[-1] >= KICK_SEP):
            onsets.append(s)
    spans, voids = [], []
    for k0 in onsets:
        k1 = None
        end = min(k0 + 1 + KICK_CAP, n - 2)
        f = k0 + 1
        while f < end:
            if tracked[f] and dist_c[f] >= R_CENTER:
                k1 = f
                break
            d3 = disp[f:f + 3]  # finite motion only: inf means unknown, not fast
            if np.isfinite(d3).all() and (d3 > DISP_TAU).all():
                k1 = f
                break
            f += 1
        if k1 is None:
            voids.append(k0)
            continue
        for _, a, b in halves:  # spans never cross a half boundary
            if a <= k0 <= b:
                k1 = min(k1, b + 1)
        spans.append((k0, k1))
    return spans, voids


def kickoff_spans(tracked, bx, by, disp, still, ring, halves, pitch):
    """Fires-twice gated spans: exactly 2 events, one per half, else []."""
    spans, _ = kickoff_events(tracked, bx, by, disp, still, ring, halves, pitch)
    if len(spans) != 2:
        return []
    hs = [next((h for h, a, b in halves if a <= k0 <= b), None) for k0, _ in spans]
    if None in hs or hs[0] == hs[1]:
        return []
    return spans


def d_spans(alive, extra=()):
    """(gap_start, offset) of each D run >= D_MIN; trap spans join as (k1, k1)."""
    chg = np.flatnonzero(np.diff(np.concatenate(
        ([False], ~np.asarray(alive), [False]))))
    return ([(int(chg[i]), int(chg[i + 1])) for i in range(0, len(chg), 2)
             if chg[i + 1] - chg[i] >= D_MIN]
            + [(k1, k1) for _, k1 in extra])


def slot_geometry(raw_path, wanted, team_of_track, pitch):
    """{frame: (xy metres, team codes)} for wanted frames only; one pass."""
    length, width = pitch
    out = {}
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1
            if frame not in wanted:
                continue
            xy, tm = [], []
            for slot, group in enumerate(parts[1].split(";")):
                if group[0] == "N":
                    continue
                c = group.index(",")
                xy.append([float(group[:c]) * length,
                           float(group[c + 1:group.index(",", c + 1)]) * width])
                tm.append(team_of_track[slot])
            out[frame] = (np.array(xy), np.array(tm, dtype=np.int8))
    return out


def reseed_anchors(d1, letter, quorum, bx, by, tracked, gaps, pitch,
                   raw_path=None, team_of_track=None):
    """Re-seed anchors after D gaps: PZ-confirmed K-runs, else cluster fallback.

    gaps = [(gap_start, offset)]. Returns (frames, teams, abstains); abstains
    are [offset, offset+ABSTAIN_LEN) X windows for dead heats. raw_path None
    (unit path) skips the fallback geometry pass, so no-K-run gaps abstain.
    """
    n = len(d1)
    live = (letter == "H") | (letter == "A")
    pz = live & quorum & (d1 < RPZ)
    frames, teams, abstains, need_fb = [], [], [], []
    for gs, o in gaps:
        e = min(o + RESTART_WINDOW, n)
        run = -1
        for i in range(o, e - K_RESEED + 1):
            if (pz[i:i + K_RESEED].all()
                    and (letter[i:i + K_RESEED] == letter[i]).all()):
                run = i
                break
        if run >= 0:
            frames.append(run)
            teams.append(letter[run])
        else:
            need_fb.append((gs, o))
    if need_fb and raw_path is not None:
        wanted = set()
        for _, o in need_fb:
            wanted.update(range(o, min(o + RESTART_WINDOW, n)))
        geo = slot_geometry(raw_path, wanted, team_of_track, pitch)
        for gs, o in need_fb:
            frames, teams, abstains = _fallback(
                gs, o, bx, by, tracked, geo, pitch, frames, teams, abstains)
    else:
        abstains += [(o, min(o + ABSTAIN_LEN, n)) for _, o in need_fb]
    return frames, teams, abstains


def _fallback(gs, o, bx, by, tracked, geo, pitch, frames, teams, abstains):
    """Cluster fallback for one gap: append anchor or abstain, return all three."""
    n = len(bx)
    lo = max(0, gs - RESTART_LOOKBACK)
    cand = np.flatnonzero(tracked[lo:o + 1])
    if not len(cand):
        return frames, teams, abstains + [(o, min(o + ABSTAIN_LEN, n))]
    rx, ry = bx[lo + cand[-1]], by[lo + cand[-1]]
    length, width = pitch
    if min(ry, width - ry) < TOUCH_SNAP:  # near-boundary restarts snap to the
        ry = 0.0 if ry < width / 2 else width  # line: throw/corner/goal geometry
    if min(rx, length - rx) < TOUCH_SNAP:
        rx = 0.0 if rx < length / 2 else length
    pick = next((f for f in range(o, min(o + RESTART_WINDOW, n))
                 if f in geo and len(geo[f][0]) >= 2), None)
    if pick is None:
        return frames, teams, abstains + [(o, min(o + ABSTAIN_LEN, n))]
    xy, tm = geo[pick]
    near = tm[np.hypot(xy[:, 0] - rx, xy[:, 1] - ry) < FALLBACK_RADIUS]
    if len(near) < MIN_FB_SLOTS:
        return frames, teams, abstains + [(o, min(o + ABSTAIN_LEN, n))]
    share_h = float(np.sum(near == HOME)) / len(near)
    team = ("H" if share_h >= FALLBACK_QUORUM
            else "A" if 1 - share_h >= FALLBACK_QUORUM else None)
    if team is None:
        return frames, teams, abstains + [(o, min(o + ABSTAIN_LEN, n))]
    return frames + [pick], teams + [team], abstains


def pipeline(scan_out, pitch, halves, raw_path=None, team_of_track=None,
             use_trap=False, use_restart=False, back_cap=None):
    """PR1 stages with optional P4 trap/restart. Flags off == PR1 bit-identical."""
    (codes, d1, margin, disp, tracked, nh, na, spd, spdn, slow, ring, bx, by) = scan_out
    n = len(codes)
    vis = codes != NOBODY
    letter = np.where(codes == HOME, "H", np.where(vis, "A", "X"))
    anchors = np.flatnonzero(vis & (margin > MARGIN) & (d1 < D1_MAX)
                             & (disp < DISP_TAU))
    alive = np.isin(smooth(codes, 50), ("H", "A"))  # mask only, not teams
    spans = []
    if use_trap:
        spans = kickoff_spans(tracked, bx, by, disp,
                              stillness(spd, slow, spdn), ring, halves, pitch)
    abstains, n_reseed = [], 0
    if use_restart:
        quorum = (nh > 0) & (na > 0)
        rfr, rtm, abstains = reseed_anchors(
            d1, letter, quorum, bx, by, tracked, d_spans(alive, spans), pitch,
            raw_path, team_of_track)
        if len(rfr):
            base = {int(a): t for a, t in zip(anchors, letter[anchors])}
            for f, t in zip(rfr, rtm):
                base.setdefault(int(f), t)  # PR1 anchors win collisions
            anchors = np.array(sorted(base))
            teams = np.array([base[a] for a in anchors])
        else:
            teams = letter[anchors]
        n_reseed = len(rfr)
    else:
        teams = letter[anchors]
    team = prune(interpolate(n, anchors, teams))
    for s, e in abstains:
        team[s:e] = "X"
    final = np.where(alive, team, "D")
    for _, a, b in halves:  # hold-fill halves; passthrough out
        final[a:b + 1] = hold_fill(final[a:b + 1], back_cap=back_cap)
    for s, e in spans:
        final[s:e] = "D"
    return final, {"spans": spans, "n_reseed": n_reseed, "n_abstain": len(abstains)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trap", action="store_true", help="P4 kickoff trap")
    ap.add_argument("--restart", action="store_true", help="P4 restart re-seed")
    args = ap.parse_args(argv)
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME, None)
    n = source_frames("tactical", GAME)
    team_of_track = team_of_each_track(meta_path)
    scan_out = scan(raw_path, team_of_track, n, GAMES[GAME]["pitch"])
    halves = half_video_blocks(GAME)
    final, info = pipeline(scan_out, GAMES[GAME]["pitch"], halves, raw_path,
                           team_of_track, args.trap, args.restart)
    with open(OUT, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(final))
    extra = (" trap={} reseed={} abstain={}".format(info["spans"], info["n_reseed"],
                                                   info["n_abstain"])
             if args.trap or args.restart else "")
    print("wrote {} ({} rows){}".format(os.path.relpath(OUT, ROOT), len(final), extra))


if __name__ == "__main__":
    main()
