"""Last-touch ownership chain (bundle-day design, pre-registered 2026-10-30).

Motivation: the anchor autopsy showed ZERO opposite-team anchors across
entire miss windows (e.g., 127H/0A over the 641f P07 monster spanning two
true A spells): nearest-proximity lies for seconds in duels, scrambles,
and shields. Football possession follows the last touch.

v2 (frozen before the first v2-score): v1 fired on in-flight noise
(fast-ball jitter far from players, which persistence cannot reject
because fast motion persists). v2 gates on PROXIMITY FIRST (ball must be
at feet), then a loose event: |dv| > V_TOUCH=2.0 m/s (walking-pace kick;
above the p90 0.89 jitter band) OR heading turn > V2_TURN=30deg with
min-speed V2_MINV=3.0 m/s (clear deflection at jog pace; the min-speed
gates low-speed heading noise). Persistence (V_PERSIST=0.5), slow-endpoint
attribution, R_TOUCH=2.0m / DZ_R=2.0m (body scale) kept. Structural
fallback (GT-blind): V_TOUCH_FB=1.0 iff attributed < MIN_ATTRIBUTED=200.

numpy-only. No GT contact: distributions + physics + body scale only.
"""

import numpy as np

from starter.baseline import HOME  # noqa: E402

FPS = 25.0
V_TOUCH = 2.0
V_TOUCH_FB = 1.0
V_PERSIST = 0.5
V2_TURN = 30.0
V2_MINV = 3.0
R_TOUCH = 2.0
DZ_R = 2.0
MIN_ATTRIBUTED = 200


def _frame_dv(bx, by):
    """Per-frame ball speed and |dv| from meter positions (NaN-safe)."""
    v = np.full_like(bx, np.nan, dtype=float)
    ok = (~np.isnan(bx[:-1]) & ~np.isnan(bx[1:])
          & ~np.isnan(by[:-1]) & ~np.isnan(by[1:]))
    v[1:][ok] = np.hypot(np.diff(bx)[ok], np.diff(by)[ok]) * FPS
    d = np.full_like(bx, np.nan, dtype=float)
    ok2 = ~np.isnan(v[:-1]) & ~np.isnan(v[1:])
    d[1:][ok2] = np.abs(np.diff(v)[ok2])
    return v, d


def _heading_turn(bx, by):
    """Turn angle (deg) at each frame between incoming/outgoing segments."""
    dx, dy = np.diff(bx), np.diff(by)
    turn = np.full_like(bx, np.nan, dtype=float)
    h0 = np.arctan2(dy[:-1], dx[:-1])
    h1 = np.arctan2(dy[1:], dx[1:])
    turn[1:-1] = (np.abs((h1 - h0 + np.pi) % (2 * np.pi) - np.pi)
                  * 180.0 / np.pi)
    return turn


def slot_proximity(raw_path, team_of_track, n, pitch):
    """Per-frame (minH, minA) ball-slot meters. One raw pass."""
    pw, ph = pitch
    minH = np.full(n, np.inf)
    minA = np.full(n, np.inf)
    with open(raw_path, "r") as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1
            if not (0 <= frame < n) or parts[2][0] == "N" or not parts[1]:
                continue
            c1 = parts[2].index(",")
            px = float(parts[2][:c1]) * pw
            py = float(parts[2][c1 + 1:parts[2].index(",", c1 + 1)]) * ph
            bh, ba = np.inf, np.inf
            for slot, group in enumerate(parts[1].split(";")):
                if not group or group[0] == "N":
                    continue
                f = group.split(",")
                if len(f) < 2:
                    continue
                dist = np.hypot(float(f[0]) * pw - px,
                                float(f[1]) * ph - py)
                if team_of_track[slot] == HOME:
                    bh = min(bh, dist)
                else:
                    ba = min(ba, dist)
            minH[frame], minA[frame] = bh, ba
    return minH, minA


def touch_chain(raw_path, team_of_track, n, pitch, halves, bx, by,
                v_touch=V_TOUCH):
    """Ownership H/A/X per frame from the last attributed touch.

    bx/by: ball meters. Event: |dv|>v_touch OR (turn>V2_TURN at speed).
    Persistence confirmation, proximity-first, slow-endpoint attribution,
    DZ duel veto. Chain resets to X at each half start. Returns
    (ownership, info dict).
    """
    v, d = _frame_dv(bx, by)
    turn = _heading_turn(bx, by)
    turning = np.zeros(n, dtype=bool)
    turning[1:-1] = ((turn[1:-1] > V2_TURN)
                     & (np.minimum(v[1:-1], v[2:]) > V2_MINV))
    event = (d > v_touch) | turning
    confirmed = np.ones(n, dtype=bool)
    both = ~np.isnan(v[:-1]) & ~np.isnan(v[1:])
    mx = np.maximum(v[:-1], v[1:])
    okp = both & (mx > 0)
    ratio = np.zeros(n - 1, dtype=bool)
    ratio[okp] = ((np.minimum(v[:-1][okp], v[1:][okp]) / mx[okp])
                  > V_PERSIST)
    confirmed[:-1] = ratio | ~both
    minH, minA = slot_proximity(raw_path, team_of_track, n, pitch)
    touches = {}
    vetoed = 0
    n_prox = 0
    for f in np.flatnonzero(event & confirmed):
        fs = f if np.isnan(v[f - 1]) else (f - 1 if v[f - 1] <= v[f] else f)
        if min(minH[fs], minA[fs]) >= R_TOUCH:
            continue
        n_prox += 1
        if minH[fs] < DZ_R and minA[fs] < DZ_R:
            vetoed += 1
            continue
        touches[f] = "H" if minH[fs] < minA[fs] else "A"
    own = np.full(n, "X", dtype="<U1")
    for _, ha, hb in halves:
        owner = "X"
        for f in range(ha, hb + 1):
            if f in touches:
                owner = touches[f]
            own[f] = owner
    return own, {"events": int(np.sum(event & confirmed)),
                 "proximate": n_prox, "attributed": len(touches),
                 "vetoed": vetoed, "v_touch": v_touch}
