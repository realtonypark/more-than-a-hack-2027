"""P5 ATD experiments (T-ATD-1..6, phase-3a design §6). Bundle-gated.

Run from repo root: python starter/p5_atd_check.py
Without the gated bundle it prints the missing-data message and exits 0:
that is the normal first-run state, not a bug. With the bundle it derives the
stillness scale (frozen rule, ATD stats only), runs the dual trigger
standalone, and reports the falsifier band + diagnostics. Prints only; writes
no files. Public GT is never read here (correctness runs live in exp scripts
under /tmp, diagnostic-only, never selectors).
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from starter import possession as P  # noqa: E402


def derive_stillness_scale(mean_spd, nan_mask, halves):
    """Frozen §2.1 rule: v_lo=p15 windowed means on ball-NaN frames (graded
    halves), v_hi=p40 over all graded-half frames; fallback p10/p50 if the
    hysteresis gap collapses. Pure ATD stats, zero GT contact."""
    wm = P._windowed_mean(mean_spd, P.W_STILL, P.W_STILL_MIN)
    graded = np.zeros(len(mean_spd), dtype=bool)
    for _, ha, hb in halves:
        graded[ha:hb + 1] = True
    ok = graded & ~np.isnan(wm)
    pool_lo = wm[ok & nan_mask]
    pool_hi = wm[ok]
    if pool_lo.size < 100 or pool_hi.size < 100:
        return None, None, "pool too small ({} / {})".format(
            pool_lo.size, pool_hi.size)
    v_lo = float(np.percentile(pool_lo, 15))
    v_hi = float(np.percentile(pool_hi, 40))
    note = "p15-nan/p40-all"
    if not v_lo < v_hi:
        v_lo = float(np.percentile(pool_lo, 10))
        v_hi = float(np.percentile(pool_hi, 50))
        note = "FALLBACK p10/p50 (gap collapsed)"
    return v_lo, v_hi, note


def main():
    try:
        meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME, None)
    except IOError as err:
        print("p5_atd_check: nothing to run yet.\n{}".format(err))
        return 0
    game = GAMES[GAME]
    n = source_frames("tactical", GAME)
    halves = half_video_blocks(GAME)
    pitch = game["pitch"]

    nx, ny = P.ball_track(raw_path, n)
    mean_spd, frac_slow, cnt = P.slot_speeds(raw_path, n)
    nan_mask = np.isnan(nx)
    st_windows = P.setpiece_windows()

    print("== T-ATD-0 bundle inventory ==")
    print("frames: {} graded-halves blocks: {}".format(
        n, [(a, b) for _, a, b in halves]))
    print("ball-NaN share (video): {:.3f}; frames with slot speeds: {}".format(
        nan_mask.mean(), int(np.sum(cnt > 0))))

    print("== T-ATD-1 stillness derivation (§2.1 rule) ==")
    v_lo, v_hi, note = derive_stillness_scale(mean_spd, nan_mask, halves)
    print("v_lo={} v_hi={} ({})".format(v_lo, v_hi, note))
    if v_lo is None:
        print("ABORT: cannot derive scale; re-register before any scored run.")
        return 0

    print("== T-ATD-2 line-margin audit (M_LINE={}m frozen) ==".format(P.M_LINE))
    vis = ~nan_mask
    Lawn, W = pitch
    d_edge = np.full(n, np.nan)
    d_edge[vis] = np.minimum(
        np.minimum(nx[vis], 1.0 - nx[vis]) * Lawn,
        np.minimum(ny[vis], 1.0 - ny[vis]) * W)
    ok = vis & ~np.isnan(d_edge)
    print("ball visible: {}; within 1.5m of a line: {} ({:.3f}); "
          "within 3m: {:.3f}".format(
              int(vis.sum()), int(np.sum(d_edge[ok] < P.M_LINE)),
              float(np.mean(d_edge[ok] < P.M_LINE)),
              float(np.mean(d_edge[ok] < 3.0))))

    print("== T-ATD-3 trigger standalone (E2) ==")
    for tag, vlo, vhi in (("init", P.V_LO_INIT, P.V_HI_INIT),
                          ("derived", v_lo, v_hi)):
        walls = P.detect_dead_spans(nx, ny, mean_spd, frac_slow, halves,
                                    pitch, st_windows, vlo, vhi)
        spans = np.array([e - s for s, e, _ in walls], dtype=float)
        flags = {}
        for _, _, f in walls:
            flags[f] = flags.get(f, 0) + 1
        d_frames = int(spans.sum()) if spans.size else 0
        graded_n = sum(b - a + 1 for _, a, b in halves)
        print("[{}] spans={} d_share={:.3f} flags={}".format(
            tag, len(walls), d_frames / graded_n, flags))
        if spans.size:
            print("  span frames: min={:.0f} p50={:.0f} mean={:.0f} "
                  "max={:.0f}".format(spans.min(),
                                      float(np.median(spans)), spans.mean(),
                                      spans.max()))
        band = ("IN 55-75 BAND" if 55 <= len(walls) <= 75
                else ("CALIBRATE 30-120" if 30 <= len(walls) <= 120
                      else "FALSIFIED (<30 or >120) -> K1"))
        print("  falsifier: {}".format(band))
    walls = P.detect_dead_spans(nx, ny, mean_spd, frac_slow, halves, pitch,
                                st_windows, v_lo, v_hi)

    print("== T-ATD-4 H2-head check (E5; published boundary, not GT) ==")
    final, _ = P.run_pipeline(raw_path, meta_path, halves, n, pitch,
                              st_windows, v_lo, v_hi)
    head = final[110808:113742]
    print("frames 110808-113741 D fraction: {:.3f} (ship iff >=0.80)".format(
        float(np.mean(head == "D"))))

    print("== T-ATD-5 long-D coverage (M8 un-bench gate) ==")
    long_st = [(s, e) for s, e, ln in st_windows if ln >= 10 * 25]
    hits = 0
    for s, e in long_st:
        ov = sum(max(0, min(we, e) - max(ws, s))
                 for ws, we, _ in walls)
        if ov >= 0.5 * (e - s):
            hits += 1
    print("long ST set pieces (>=10s): {}; overlapped>=50%: {} (recall {:.2f};"
          " <0.5 un-benches M8)".format(
              len(long_st), hits, hits / len(long_st) if long_st else 0.0))

    print("== T-ATD-6 anchor-jitter robustness ==")
    for shift in (-25, 25):
        shifted = [(s + shift, e + shift, ln) for s, e, ln in st_windows]
        w2 = P.detect_dead_spans(nx, ny, mean_spd, frac_slow, halves, pitch,
                                 shifted, v_lo, v_hi)
        a = np.zeros(n, dtype=bool)
        b = np.zeros(n, dtype=bool)
        for ws, we, _ in walls:
            a[ws:we] = True
        for ws, we, _ in w2:
            b[max(0, ws):max(0, we)] = True
        inter = float(np.sum(a & b))
        union = float(np.sum(a | b))
        print("shift {:+d}f: span IoU {:.3f} (pass iff change <0.1, "
              "i.e. IoU>0.9)".format(shift, inter / union if union else 1.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
