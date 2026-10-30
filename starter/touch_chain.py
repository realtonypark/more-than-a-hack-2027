"""Last-touch ownership chain (bundle-day design, pre-registered 2026-10-30).

Motivation: the anchor autopsy showed ZERO opposite-team anchors across
entire miss windows (e.g., 127H/0A over the 641f P07 monster spanning two
true A spells): nearest-proximity lies for seconds in duels, scrambles,
and shields. Football possession follows the last touch; this module
builds ownership from attributed ball touches instead of proximity.

Thresholds (frozen before the first touch-score; see PREREGISTRATION.md):
V_TOUCH=5.0 m/s frame-delta (physics: a clear kick impulse; provider
p995=4.81, p999=6.85; yields ~485 candidates/match, precision-first),
R_TOUCH=2.0 m (body scale: ball within reach to attribute),
DZ_R=2.0 m (duel veto: both teams inside -> contested, hold last owner).
Structural fallback (GT-blind): V_TOUCH_FB=4.0 iff attributed touches
fall below MIN_ATTRIBUTED=200 (~1/1000f starvation floor).

numpy-only. No GT contact: distributions + physics + body scale only.
"""

import numpy as np

from starter.baseline import HOME  # noqa: E402

FPS = 25.0
V_TOUCH = 5.0
V_TOUCH_FB = 4.0
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


def touch_candidates(bx, by, v_touch=V_TOUCH):
    """Frames whose ball |dv| exceeds v_touch (bx/by in meters)."""
    _, d = _frame_dv(bx, by)
    return np.flatnonzero(d > v_touch)


def touch_chain(raw_path, team_of_track, n, pitch, halves, bx, by,
                v_touch=V_TOUCH):
    """Ownership H/A/X per frame from the last attributed touch.

    bx/by: ball meters (NaN where missing). Attribution: nearest slot
    team within R_TOUCH at the candidate frame; DZ veto (both teams
    within DZ_R) holds the last owner. Chain resets to X at each half
    start. Returns (ownership, info dict). One raw pass over candidates.
    """
    pw, ph = pitch
    cand = set(touch_candidates(bx, by, v_touch).tolist())
    touches = {}
    vetoed = 0
    with open(raw_path, "r") as handle:
        for line in handle:
            frame = int(line.split(":", 1)[0]) - 1
            if frame not in cand:
                continue
            parts = line.rstrip("\n").split(":")
            if len(parts) < 3 or not parts[1]:
                continue
            b = parts[2].split(",")
            if len(b) < 2 or not b[0] or b[0] == "NaN":
                continue
            px, py = float(b[0]) * pw, float(b[1]) * ph
            bh, ba = np.inf, np.inf
            for slot, group in enumerate(parts[1].split(";")):
                if not group or group[0] == "N":
                    continue
                f = group.split(",")
                if len(f) < 2:
                    continue
                dist = np.hypot(float(f[0]) * pw - px, float(f[1]) * ph - py)
                if team_of_track[slot] == HOME:
                    bh = min(bh, dist)
                else:
                    ba = min(ba, dist)
            if min(bh, ba) > R_TOUCH:
                continue  # unattributed: nobody within reach
            if bh < DZ_R and ba < DZ_R:
                vetoed += 1
                continue  # duel: hold last owner
            touches[frame] = "H" if bh < ba else "A"
    own = np.full(n, "X", dtype="<U1")
    for _, ha, hb in halves:
        owner = "X"
        for f in range(ha, hb + 1):
            if f in touches:
                owner = touches[f]
            own[f] = owner
    return own, {"candidates": len(cand), "attributed": len(touches),
                 "vetoed": vetoed, "v_touch": v_touch}
