"""P5 episode-first possession: D walls from dual triggers + guarded smoothing.

w3-p5-episode on PR1 base 64244c2. Design: /tmp/w3-p5episode.md (phase 3a).
Inverts PR1: detect dead-ball spans FIRST (ball-out AND stillness, per half),
freeze them as walls, classify H/A and smooth ONLY inside in-play episodes.
PR1 anchor predicate and scan() are untouched; what changes is scope (walls),
the alive mask (guarded vote, ties keep-center), and hold_fill (capped head).

W3-P2 (branch w3-p2-antiprune): prune() replaced by graded_prune() — a D-first
two-pass RLE prune with a GAP-sourced differ-flank exemption (short-spell
rescue), an alternation test, and a D-share floor. Bands pre-registered in the
phase-3a design before any scored run; see the *_THR constants. prune() is kept
as the frozen reference. main() runs the graded path.

w3-integrate: P1 quorum + P2 E6 + P5 structure on this base. Default walls are
PR1-morphology (K1 fallback); --walls enables the HELD P5 trigger detectors
pending the T-ATD-1 gate in starter/p5_atd_check.py.

P4 restart (w3-p4-restart): optional --trap (kickoff formation+stillness forces
D until first touch, fires-twice gated) and --restart (PZ-winner re-seed anchors
after D gaps, K=5 confirmed). Both default off. E1-E4 gates decide integration
into run_pipeline; until then --trap/--restart run P4's standalone pipeline().

Run from repo root: python starter/possession.py -> results CSV.
numpy-only (+ stdlib + games.py + starter helpers); no pandas, no kloppy.
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
from starter.load_smart_tagging import load as st_load  # noqa: E402
from starter.touch_chain import (MIN_ATTRIBUTED, V_TOUCH,  # noqa: E402
                                 V_TOUCH_FB, touch_chain)

# --- PR1 frozen constants (unchanged) -------------------------------------
# (M, G swept {1,2,3}x{25,50,100}; M3/G100 closest to ~300 spells.)
MARGIN, GAP, PRUNE = 3.0, 100, 12
D1_MAX, DISP_TAU = 3.0, 0.2  # m; m/frame (ATD ball speed is 91% NaN)
# P1 two-sided quorum (frozen pre-bundle, /tmp/w3-p1quorum.md §2): an anchor
# needs >=Q_MIN visible slots on EACH side, else the opponent distance is
# unknown and the margin gate must not pass. Q_MIN=1 is the minimum
# cardinality for "measured", not a tuned threshold; all values above are
# PR1-frozen and unchanged.
Q_MIN = 1
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
# --- P5 pre-registered constants (frozen in phase-3a design; see §2 there) --
# Justifications are non-GT by construction: pitch physics, locomotion bands,
# frozen PR1 geometry, and ATD-stats derivation rules. Any change needs a
# re-registration note in the competition report, never a silent edit.
M_LINE = 1.5       # m; ball within this of touch/goal line counts as "at line"
K_LINE = 5         # visible frames of line-proximity history before a NaN run
N_ONSET = 10       # ball-NaN run length (0.4s) for a ball-out onset
N_REAPPEAR = 5     # consecutive visible frames that confirm ball reappearance
W_STILL = 25       # stillness window (1s of slot speeds)
W_STILL_MIN = 12   # valid frames needed inside a stillness window
S_Q = 1.5          # m/s; slot counts as "slow" below this (stand/jog band)
F_STILL = 0.60     # supermajority of visible slots slow for stillness onset
N_WAKE = 12        # consecutive above-v_hi frames that confirm stillness offset
T_CONJ = 50        # frames (2s); conjuncts within this pair into one D onset
D_MAX_SPAN = 1500  # frames (60s); span cap adopted from d-H3 (no-offset case)
BLINK_MERGE = 6    # frames; merge triggered spans closer than this (tracker blink)
D_MIN_KEEP = 12    # frames; reuse frozen PRUNE: spans below this flag shortD, kept
D_CLOSE = 12       # frames; H/A flicker inside firm D filled by morphological close
PRUNE_INPLAY = 20  # frames (0.8s); literature sub-second chain filter, below the
# ... ~100f erasure floor so P10/P12-class (19f/33f) short spells can survive.
W_G = 50           # guarded vote half-window; conservative half of the 90-125
# ... plateau (w1-postproc); guarded removes the erasure that forced wide windows.
HEAD_CAP = GAP // 2  # 50f; g-H5(iii) back-fill cap, frozen PR1 geometry
ST_WIN_BACK = 75   # ST candidate window [start-3s, end+1s]; ST ~1s accuracy,
ST_WIN_FWD = 25    # asymmetric because a restart marks the END of dead ball.
FAST_RESTART_S = 50  # ST SET PIECES span <2s with no trigger stays in-play.
# Stillness scale start points. Provider speed scale is unknown until the bundle
# lands (w1-methods U1); T-ATD-1 derives v_lo/v_hi from ATD stats via the frozen
# rule (p15 ball-NaN / p40 graded, fallback p10/p50) and re-registers these two
# numbers BEFORE the first scored run. Nothing below is tuned against public GT.
V_LO_INIT, V_HI_INIT = 1.0, 2.0


def scan(raw_path, team_of_track, n, pitch):
    """Raw nearest-team codes + d1/margin/ball-displacement per frame.

    Ball track is pre-read so bracketed missing runs <=GAP_N get linear-interp
    x/y and full votes; 26..250f runs stay voteless (gap-fill hold/X); longer
    and unbracketed runs are untouched.

    Returns (codes, d1, margin, disp, qh, qa, tracked, nh, na, spd, spdn,
    slow, ring, bx, by). margin is finite only where both teams have >=1
    visible slot (P1 quorum); single-team frames get -inf (opponent
    distance unknown), no-vote frames keep +inf with vis False. P4 tail:
    ball-tracked flag, per-team visible counts (nh/na == qh/qa by
    construction), slot-speed sum/count/slow-count, median center-ring
    distance (NaN = no slots). Callers slice: ours [:6], P4 [:4]+[6:].
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
    qh = np.zeros(n, dtype=np.int16)  # visible HOME slots (0 = none / no vote)
    qa = np.zeros(n, dtype=np.int16)  # visible AWAY slots
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
            ch, ca = 0, 0
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
                    ch += 1
                    h_f += 1
                else:
                    ba = min(ba, dist)
                    ca += 1
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
            qh[frame], qa[frame] = ch, ca
            if ch > 0 and ca > 0:
                margin[frame] = abs(float(np.sqrt(bh) - np.sqrt(ba)))
            else:
                margin[frame] = -np.inf  # one side unseen: gap UNKNOWN, not far
            bx[frame], by[frame] = px, py
            nh[frame], na[frame] = h_f, a_f
            spd[frame], spdn[frame], slow[frame] = s_sum, s_n, s_slow
            if rds:
                ring[frame] = float(np.sqrt(np.median(rds)))
    disp = np.full(n, np.inf)
    ok = ~np.isnan(bx[1:]) & ~np.isnan(bx[:-1])
    disp[1:][ok] = np.hypot(np.diff(bx)[ok], np.diff(by)[ok])
    # Union return (w3-int + P4): ours-6 then P4's 9. Callers slice (see
    # docstring); nh/na duplicate qh/qa for caller compatibility.
    return (codes, d1, margin, disp, qh, qa, tracked, nh, na, spd, spdn,
            slow, ring, bx, by)


def ball_track(raw_path, n):
    """Normalized ball x/y per video frame; NaN where the ball is missing."""
    nx = np.full(n, np.nan)
    ny = np.full(n, np.nan)
    with open(raw_path) as handle:
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
    return nx, ny


def slot_speeds(raw_path, n):
    """Per-frame (mean visible-slot speed, fraction slow, n visible).

    Slot group layout is assumed `x,y,s[,...]` (s in m/s per DATA.md); a group
    without a parseable third field contributes position but no speed. Frames
    with no visible slot yield NaN mean / NaN fraction / 0 count, which the
    stillness detector treats as "no evidence", never as stillness.
    """
    tot = np.zeros(n)
    slow = np.zeros(n)
    cnt = np.zeros(n, dtype=np.int64)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1
            if frame < 0 or frame >= n:
                continue
            for group in parts[1].split(";"):
                if group[0] == "N":
                    continue
                fields = group.split(",")
                if len(fields) < 3:
                    continue
                try:
                    spd = float(fields[2])
                except ValueError:
                    continue
                if np.isnan(spd):
                    continue
                tot[frame] += spd
                cnt[frame] += 1
                if spd < S_Q:
                    slow[frame] += 1
    mean = np.full(n, np.nan)
    frac = np.full(n, np.nan)
    vis = cnt > 0
    mean[vis] = tot[vis] / cnt[vis]
    frac[vis] = slow[vis] / cnt[vis]
    return mean, frac, cnt


def setpiece_windows(game=GAME, data_dir=None):
    """ST SET PIECES candidate windows [start-3s, end+1s) in video frames.

    Windows CONFIRM or VETO marginal triggers only; ST never writes a D label
    (DATA.md:179-180). Missing/unparseable ST asset -> [] with a note: triggers
    run standalone (firm conjuncts only), never suppressed by ST absence.
    """
    try:
        events = st_load(game, data_dir)
    except Exception as err:  # IOError (gated bundle) or XML parse failure
        print("setpiece_windows: no Smart Tagging ({}) -> "
              "triggers standalone".format(err).split("\n")[0])
        return []
    wins = []
    for ev in events:
        if (ev["code"] or "").strip().lower() != "set pieces":
            continue
        s = max(0, ev["start_frame"] - ST_WIN_BACK)
        e = ev["end_frame"] + ST_WIN_FWD
        wins.append((s, e, ev["end_frame"] - ev["start_frame"]))
    return wins


def _nan_runs(mask):
    """[start, end) half-open runs where mask is True."""
    edges = np.flatnonzero(np.diff(np.concatenate(([False], mask, [False]))))
    return [(int(edges[i]), int(edges[i + 1])) for i in range(0, len(edges), 2)]


def _windowed_mean(vals, width, min_valid):
    """Trailing windowed mean; NaN where fewer than min_valid frames valid."""
    ok = ~np.isnan(vals)
    cs = np.concatenate(([0.0], np.cumsum(np.where(ok, vals, 0.0))))
    cc = np.concatenate(([0], np.cumsum(ok)))
    n = len(vals)
    idx = np.arange(n)
    lo = np.maximum(0, idx - width + 1)
    tot = cs[idx + 1] - cs[lo]
    cnt = cc[idx + 1] - cc[lo]
    out = np.full(n, np.nan)
    good = cnt >= min_valid
    out[good] = tot[good] / cnt[good]
    return out


def detect_dead_spans(nx, ny, mean_spd, frac_slow, halves, pitch,
                      st_windows=(), v_lo=V_LO_INIT, v_hi=V_HI_INIT):
    """D-wall spans [(start, end, flag)] from the dual trigger, per half.

    Ball-out onset (at-line then NaN>=N_ONSET) AND stillness onset (windowed
    mean < v_lo with supermajority slow) within T_CONJ pair into one firm span.
    Offset = first confirmed of {ball reappear, stillness wake}. A lone
    conjunct inside an ST window becomes a marginalD span (P4 resolves its
    tail); a lone conjunct outside ST is dropped (precision-first). Fast
    restarts (ST span <2s, no trigger) stay in-play: no forced D. Flags:
    firm | marginalD | shortD (kept, <D_MIN_KEEP) | noOffset (capped).
    """
    length, width = pitch
    walls = []
    wins = [(s, e) for s, e, _ in st_windows]

    def in_st(t):
        return any(s <= t < e for s, e in wins)

    for _, ha, hb in halves:
        s0, s1 = ha, hb + 1  # half-open slice; windows never span halftime
        bx, by = nx[s0:s1], ny[s0:s1]
        nan = np.isnan(bx)
        vis = ~nan
        # Line proximity on visible frames only.
        edge = np.zeros(len(bx), dtype=bool)
        edge[vis] = (
            (np.minimum(bx[vis], 1.0 - bx[vis]) * length < M_LINE)
            | (np.minimum(by[vis], 1.0 - by[vis]) * width < M_LINE))
        # Ball-out onsets: NaN run >= N_ONSET with line-contact in the last
        # K_LINE visible frames before it.
        vis_idx = np.flatnonzero(vis)
        ball_on = []
        for rs, re in _nan_runs(nan):
            if re - rs < N_ONSET:
                continue
            back = vis_idx[vis_idx < rs][-K_LINE:]
            if back.size and edge[back].any():
                ball_on.append(rs)
        # Reappearances: visible runs >= N_REAPPEAR (offset = run start).
        reappear = [rs for rs, re in _nan_runs(vis) if re - rs >= N_REAPPEAR]
        # Stillness: windowed series on the slice, NaN-aware.
        wm = _windowed_mean(mean_spd[s0:s1], W_STILL, W_STILL_MIN)
        wf = _windowed_mean(frac_slow[s0:s1], W_STILL, W_STILL_MIN)
        still = ~np.isnan(wm) & (wm < v_lo) & (wf >= F_STILL)
        still_on = [rs for rs, _ in _nan_runs(still)]
        wake = ~np.isnan(wm) & (wm > v_hi)
        wake_off = [rs for rs, re in _nan_runs(wake) if re - rs >= N_WAKE]

        def first_ge(sorted_list, t):
            for v in sorted_list:
                if v >= t:
                    return v
            return None

        used_still = set()
        spans = []
        for b in ball_on:
            mate = next((s for s in still_on
                         if abs(s - b) <= T_CONJ and s not in used_still), None)
            if mate is None:
                if in_st(s0 + b):  # lone conjunct in ST window -> marginal
                    off = first_ge(reappear, b)
                    flag = "marginalD"
                    if off is None:
                        off, flag = min(b + D_MAX_SPAN, s1 - s0), "marginalD"
                    spans.append((b, off, flag))
                continue
            used_still.add(mate)
            on = min(b, mate)
            cands = [v for v in (first_ge(reappear, on),
                                 first_ge(wake_off, on)) if v is not None]
            if cands:
                spans.append((on, min(cands), "firm"))
            else:
                spans.append((on, min(on + D_MAX_SPAN, s1 - s0), "noOffset"))
        for s in still_on:  # lone stillness conjuncts: marginal iff in ST
            if s in used_still:
                continue
            if in_st(s0 + s):
                off = first_ge(wake_off, s)
                if off is None:
                    off = first_ge(reappear, s)
                spans.append((s, off if off is not None
                              else min(s + D_MAX_SPAN, s1 - s0), "marginalD"))
        # Merge spans split by tracker blink; clip to the half slice.
        spans.sort()
        merged = []
        for s, e, flag in spans:
            e = min(max(e, s + 1), s1 - s0)
            if merged and s - merged[-1][1] < BLINK_MERGE:
                ps, _, pflag = merged.pop()
                keep = ("firm" if pflag == flag == "firm"
                        else ("marginalD" if "marginalD" in (pflag, flag)
                              else pflag))
                merged.append((ps, e, keep))
            else:
                merged.append((s, e, flag))
        for s, e, flag in merged:
            if e - s < D_MIN_KEEP and flag == "firm":
                flag = "shortD"
            walls.append((s0 + s, s0 + e, flag))
    walls.sort()
    return walls


def episodes_from_walls(n, walls, halves):
    """In-play episodes: complement of walls inside halves, half-open [s,e)."""
    blocked = []
    for _, ha, hb in halves:
        cur = ha
        for ws, we, _ in sorted(w for w in walls if w[0] < hb + 1 and w[1] > ha):
            s, e = max(ws, ha), min(we, hb + 1)
            if s > cur:
                blocked.append((cur, s))
            cur = max(cur, e)
        if cur < hb + 1:
            blocked.append((cur, hb + 1))
    return blocked


def interpolate(n, anchors, teams, prov=None, walls=None, halves=None):
    """Gap fill between anchors; X core in long/differing gaps.

    prov (P2): when given, each painted frame is tagged PROV_GAPFILL /
    PROV_EDGE / PROV_ANCHOR; labels are identical with or without it.
    walls (P5): None -> PR1 path bit-identical (+ empty-anchor guard);
    with walls, paint ranges intersect episodes and wall frames stay X
    (guarded_smooth stamps them D).
    """
    out = np.full(n, "X", dtype="<U1")
    if prov is not None:
        prov[:] = PROV_EDGE  # before the empty guard: unpainted prov must
    if len(anchors) == 0:    # read EDGE, never uninitialized garbage
        return out
    out[anchors] = teams
    h = GAP // 2
    if prov is not None:
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

    if walls is None:
        paint(0, anchors[0], None, teams[0])
        for (s, lt), (e, rt) in zip(zip(anchors, teams),
                                    zip(anchors[1:], teams[1:])):
            paint(s + 1, e, lt, rt)
        paint(anchors[-1] + 1, n, teams[-1], None)
        return out
    for es, ee in episodes_from_walls(n, walls, halves):
        inside = (anchors >= es) & (anchors < ee)
        ea, et = anchors[inside], teams[inside]
        if len(ea) == 0:
            continue
        paint(es, ea[0], None, et[0])
        for (s, lt), (e, rt) in zip(zip(ea, et), zip(ea[1:], et[1:])):
            paint(s + 1, e, lt, rt)
        paint(ea[-1] + 1, ee, et[-1], None)
    return out


def interpolate_with_prov(n, anchors, teams, walls=None, halves=None):
    """interpolate() plus the per-frame provenance array for graded_prune."""
    prov = np.empty(n, dtype=np.int8)
    return interpolate(n, anchors, teams, prov=prov, walls=walls,
                       halves=halves), prov


def _prune_once(out, thresh):
    """One reassignment sweep; returns number of runs changed."""
    chg = np.flatnonzero(out[1:] != out[:-1]) + 1
    b = np.concatenate(([0], chg, [len(out)]))
    runs = [(int(b[i]), int(b[i + 1]), out[b[i]]) for i in range(len(b) - 1)]
    changed = 0
    for i, (s, e, lab) in enumerate(runs):
        if lab not in ("H", "A") or e - s >= thresh:
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
        if rep is not None and rep != lab:
            out[s:e] = rep
            changed += 1
    return changed


def prune(out):
    """PR1 entry: reassign H/A runs <PRUNE by neighbor majority (tie->left).

    Single sweep, PR1-identical (stale-list effect preserved on purpose: a
    fixpoint iteration cascades back-to-back shorts and can oscillate, shown on
    synthetic streams during phase 3b — dropped, see competition report)."""
    _prune_once(out, PRUNE)
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


def prune_episodes(out, walls, halves, thresh=PRUNE_INPLAY):
    """Separated prune: each episode pruned independently, walls never crossed.
    In-play H/A runs < thresh merge to the EPISODE majority. (The P2/M1
    differing-flank exemption plugs in here at stack time; P5 ships the shape.)
    """
    for es, ee in episodes_from_walls(len(out), walls, halves):
        _prune_once(out[es:ee], thresh)
    return out


def guarded_smooth(codes, walls, halves, window=W_G):
    """Guarded vote: D stamped on walls; H/A majority inside episodes only.

    Window vote counts H/A codes only (NOBODY excluded, never counted as D);
    ties keep the CENTER label (kills ties-go-home bias); windows with zero
    H/A votes yield X. Outside halves -> X (ungraded passthrough). ST touches
    nothing here: inputs are ATD labels + walls (d-H5 audit point).
    """
    n = len(codes)
    out = np.full(n, "X", dtype="<U1")
    for ws, we, _ in walls:
        out[ws:we] = "D"
    for es, ee in episodes_from_walls(n, walls, halves):
        seg = codes[es:ee]
        home = np.concatenate(([0], np.cumsum(seg == HOME)))
        away = np.concatenate(([0], np.cumsum((seg != HOME) & (seg != NOBODY))))
        idx = np.arange(ee - es)
        lo = np.maximum(0, idx - window)
        hi = np.minimum(ee - es, idx + window + 1)
        h = home[hi] - home[lo]
        a = away[hi] - away[lo]
        lab = np.full(ee - es, "X", dtype="<U1")
        center_h = seg == HOME
        center_a = (seg != HOME) & (seg != NOBODY)
        win_h = (h > a) | ((h == a) & (h + a > 0) & center_h)
        win_a = (a > h) | ((h == a) & (h + a > 0) & center_a)
        # h==a ties with a NOBODY center stay X (no evidence either way).
        lab[win_h] = "H"
        lab[win_a] = "A"
        out[es:ee] = lab
    return out


def close_d_flicker(labels, walls, width=D_CLOSE):
    """Fill H/A runs <= width inside walls with D; walls never move.

    Idempotent safeguard in P5-alone (walls stamp D wholesale); at stack time
    it also repairs any H/A P4's restart retype bleeds into firm spans.
    """
    for ws, we, _ in walls:
        seg = labels[ws:we]
        for rs, re in _nan_runs((seg == "H") | (seg == "A")):
            if re - rs <= width:
                seg[rs:re] = "D"
    return labels


def assert_wall_invariants(final, walls, d_pre_halves, halves):
    """INV-1/INV-2 hard asserts (bug = stop); INV-3 shrink hard, growth soft.

    d_pre_halves = wall-frame count over graded halves (pre-smoothing D).
    Design §1.8 wants ±25%; growth via D-stop fill is BY DESIGN (X after D
    fills D), so only the erasure direction crashes. Returns D counts.
    """
    for ws, we, flag in walls:  # INV-1: walls immutable by vote
        if not np.all(final[ws:we] == "D"):
            bad = ws + int(np.flatnonzero(final[ws:we] != "D")[0])
            raise AssertionError(
                "INV-1 wall leak: {} wall [{},{}) broken at frame {}".format(
                    flag, ws, we, bad))
    # INV-2: no H/A run spans a wall (explicit scan; implied by INV-1).
    alive = (final == "H") | (final == "A")
    for rs, re in _nan_runs(alive):
        for ws, we, _ in walls:
            if ws > rs and we < re:
                raise AssertionError(
                    "INV-2 wall leak: H/A run [{},{}) spans wall [{},{})".format(
                        rs, re, ws, we))
    d_post = 0
    for _, ha, hb in halves:
        d_post += int(np.sum(final[ha:hb + 1] == "D"))
    if d_pre_halves and d_post < 0.75 * d_pre_halves:
        raise AssertionError(
            "INV-3 D erasure: {} D frames post vs {} pre".format(
                d_post, d_pre_halves))
    if d_pre_halves and d_post > 1.25 * d_pre_halves:
        print("INV-3 note: D grew {} -> {} (+{:.0f}% via D-stop fill)".format(
            d_pre_halves, d_post, 100.0 * (d_post - d_pre_halves)
            / d_pre_halves))
    return d_pre_halves, d_post


def hold_fill(w, head_cap=None):
    """Hold-last X fill for one window; cf. possession_metrics.fill_abstentions
    (reimplemented: that import pulls pandas, this file stays numpy-only).
    head_cap caps leading back-fill (None = PR1 behavior). P4's pipeline
    passes its back_cap through under this name."""
    idx = np.arange(len(w))
    carry = np.maximum.accumulate(np.where(w != "X", idx, -1))
    known = np.flatnonzero(w != "X")
    if not len(known):
        return w
    carry[carry < 0] = known[0]
    out = w[carry]
    if head_cap is not None:
        out[:max(0, known[0] - head_cap)] = "X"
    return out


def morphology_walls(codes, halves):
    """D-wall spans from the PR1 alive mask (K1 fallback, always available).

    Integration choice (w3-integrate, pre-data, non-GT): D runs >= D_MIN_KEEP
    of the smooth50 alive mask become "morphD" walls, so default D behavior
    matches PR1 structurally while the guarded vote (which never votes across
    walls) stays active. Trigger walls (--walls) replace these once T-ATD-1
    passes; see starter/p5_atd_check.py.
    """
    mask = smooth(codes, 50)
    walls = []
    for _, ha, hb in halves:
        for rs, re in _nan_runs(mask[ha:hb + 1] == "D"):
            if re - rs >= D_MIN_KEEP:
                walls.append((ha + rs, ha + re, "morphD"))
    return walls


def touch_team(raw_path, team_of_track, n, pitch, halves):
    """Last-touch ownership team stage (--touch). Pre-registered; see
    PREREGISTRATION.md. Structural fallback to V_TOUCH_FB on starvation."""
    nx, ny = ball_track(raw_path, n)
    team, info = touch_chain(raw_path, team_of_track, n, pitch, halves,
                             nx * pitch[0], ny * pitch[1], V_TOUCH)
    if info["attributed"] < MIN_ATTRIBUTED:
        team, info = touch_chain(raw_path, team_of_track, n, pitch, halves,
                                 nx * pitch[0], ny * pitch[1], V_TOUCH_FB)
        info["fallback"] = True
    print("touch: v={} candidates={} attributed={} vetoed={}{}".format(
        info["v_touch"], info["candidates"], info["attributed"],
        info["vetoed"], " FALLBACK" if info.get("fallback") else ""))
    return team


def run_pipeline(raw_path, meta_path, halves, n, pitch, walls=None,
                 use_touch=False):
    """Full integrated stack -> (final labels, walls). No I/O.

    walls=None derives K1 morphology walls from the scan's own codes.
    Team stage: P1 quorum anchors -> interpolate (prov + walls) -> PR1
    prune12 (FIX 2), or the last-touch ownership chain when use_touch
    (--touch, pre-registered in PREREGISTRATION.md). Walls stamp D;
    hold-fill per half.
    """
    codes, d1, margin, disp, qh, qa = scan(
        raw_path, team_of_each_track(meta_path), n, pitch)[:6]
    if walls is None:
        walls = morphology_walls(codes, halves)
    inplay = np.zeros(n, dtype=bool)
    for es, ee in episodes_from_walls(n, walls, halves):
        inplay[es:ee] = True
    vis = codes != NOBODY
    dual = (qh >= Q_MIN) & (qa >= Q_MIN)  # P1: both sides observed
    letter = np.where(codes == HOME, "H", np.where(vis, "A", "X"))
    gate = inplay & vis & dual & (margin > MARGIN) & (d1 < D1_MAX) \
        & (disp < DISP_TAU)
    anchors = np.flatnonzero(gate)
    team, prov = interpolate_with_prov(n, anchors, letter[anchors], walls,
                                       halves)
    if use_touch:
        team = touch_team(raw_path, team_of_each_track(meta_path), n,
                          pitch, halves)
    # FIX 2 (bundle-day): PR1 prune12 + hold-fill. E1-flat100 and voted-fill
    # regressed to 0.3415/24 (PR1: 0.5103/12): flat100 was measured on label
    # streams (wrong population: smoothing artifacts), and voted-fill inserts
    # vote-majority runs into hold-filled X cores (exactly the uncertain
    # regions). Team + walls->D + hold-fill; E1 and voted-fill stay out.
    team = prune(team)
    final = team
    for ws, we, _ in walls:
        final[ws:we] = "D"
    d_pre = sum(min(we, hb + 1) - max(ws, ha)
                for _, ha, hb in halves for ws, we, _ in walls
                if ws < hb + 1 and we > ha)
    assert_wall_invariants(final, walls, d_pre, halves)
    for _, a, b in halves:
        # Uncapped (PR1): the scorer back-fills leading X from first-known
        # regardless, so HEAD_CAP only spends macroF1 for identical spells.
        final[a:b + 1] = hold_fill(final[a:b + 1])
    assert_wall_invariants(final, walls, d_pre, halves)  # shipped artifact
    return final, walls


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
    (codes, d1, margin, disp, tracked, nh, na, spd, spdn, slow, ring, bx,
     by) = scan_out[:4] + scan_out[6:]
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
        final[a:b + 1] = hold_fill(final[a:b + 1], head_cap=back_cap)
    for s, e in spans:
        final[s:e] = "D"
    return final, {"spans": spans, "n_reseed": n_reseed, "n_abstain": len(abstains)}


def main(argv=None):
    """Default: K1 morphology stack. --walls: HELD P5 triggers (T-ATD-1).
    --trap/--restart: P4 standalone pipeline() (E1-E4 gates decide; fusion
    into run_pipeline only on PASS)."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trap", action="store_true", help="P4 kickoff trap")
    ap.add_argument("--restart", action="store_true",
                    help="P4 restart re-seed")
    ap.add_argument("--walls", action="store_true",
                    help="P5 trigger detectors (HELD pending T-ATD-1)")
    ap.add_argument("--touch", action="store_true",
                    help="last-touch ownership team stage (pre-registered)")
    args = ap.parse_args(argv)
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME, None)
    n = source_frames("tactical", GAME)
    pitch = GAMES[GAME]["pitch"]
    halves = half_video_blocks(GAME)
    team_of_track = team_of_each_track(meta_path)
    if args.trap or args.restart:
        scan_out = scan(raw_path, team_of_track, n, pitch)
        final, info = pipeline(scan_out, pitch, halves, raw_path,
                               team_of_track, args.trap, args.restart)
    else:
        walls = None
        if args.walls:
            # v_lo/v_hi: T-ATD-1 re-registered these BEFORE the first scored
            # run; see PREREGISTRATION.md. Triggers stay HELD regardless.
            nx, ny = ball_track(raw_path, n)
            mean_spd, frac_slow, _ = slot_speeds(raw_path, n)
            st_windows = setpiece_windows()
            walls = detect_dead_spans(nx, ny, mean_spd, frac_slow, halves,
                                      pitch, st_windows)
            print("ST windows: {}".format(len(st_windows)))
        final, walls = run_pipeline(raw_path, meta_path, halves, n, pitch,
                                    walls, use_touch=args.touch)
        n_d = sum(we - ws for ws, we, _ in walls)
        flags = {}
        for _, _, flag in walls:
            flags[flag] = flags.get(flag, 0) + 1
        print("walls: {} spans, {} frames ({:.1%} of video); flags {}".format(
            len(walls), n_d, n_d / n, flags))
    with open(OUT, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(final))
    extra = (" trap={} reseed={} abstain={}".format(info["spans"], info["n_reseed"],
                                                   info["n_abstain"])
             if args.trap or args.restart else "")
    print("wrote {} ({} rows){}".format(os.path.relpath(OUT, ROOT), len(final), extra))


if __name__ == "__main__":
    main()
