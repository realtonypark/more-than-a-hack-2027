"""W3-P2 ATD contest audits A1/A2/A5 (PR1 track; needs the gated bundle).

A1: kept/dropped x PROXY-touch confusion + E7 keep ratio (>=2:1 for kept).
A2: kept contest runs must be >=90% GAP-sourced (flank-rule overreach check).
A5: P1-quorum-mirror on/off; kept-short count stable within +-20%.

Without the bundle this prints a missing-data note and exits 0.
Run from repo root: python3 starter/p2_atd_contest_audit.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from starter.p2_atd_track import proxy_touch, run_pr1_track  # noqa: E402
from starter.possession import PROV_ANCHOR, PROV_GAPFILL  # noqa: E402


def audit_a1(track):
    log, touch = track["log"], track["touch"]
    assert touch is not None
    kept_c = sum(1 for s, e, _, _ in log["contest_keep"] if touch[s:e].any())
    kept_u = len(log["contest_keep"]) - kept_c
    drop_c = sum(1 for s, e, _ in log["contest_drop"] if touch[s:e].any())
    drop_u = len(log["contest_drop"]) - drop_c
    print("A1 kept(conf/unconf)=%d/%d dropped(conf/unconf)=%d/%d"
          % (kept_c, kept_u, drop_c, drop_u))
    ratio = kept_c / float(kept_u) if kept_u else float("inf")
    ok = ratio >= 2.0 and kept_c > 0
    print(("PASS " if ok else "FAIL ")
          + "A1/E7 PROXY-touch keep ratio %.2f:1 (>=2:1)" % ratio)
    return ok


def audit_a2(track):
    log, prov = track["log"], track["prov"]
    keeps = log["contest_keep"]
    src = [np.all((prov[s:e] == PROV_GAPFILL)
                  | (prov[s:e] == PROV_ANCHOR)) for s, e, _, _ in keeps]
    frac = float(np.mean(src)) if keeps else 1.0
    print("A2 gap-sourced keeps %d/%d (%.2f)"
          % (sum(src), len(keeps), frac))
    ok = frac >= 0.90
    print(("PASS " if ok else "FAIL ")
          + "A2 >=90%% GAP-sourced (else tighten flank rule)")
    return ok


def audit_a5():
    outs = {}
    for quorum in (False, True):
        track = run_pr1_track(use_graded=True, use_quorum=quorum)
        n_anchor = len(track["anchors"])
        n_keep = len(track["log"]["contest_keep"])
        outs[quorum] = (n_anchor, n_keep)
        print("A5 quorum=%s anchors=%d contest-keeps=%d"
              % (quorum, n_anchor, n_keep))
    _, k0 = outs[False]
    _, k1 = outs[True]
    drift = abs(k1 - k0) / float(k0) if k0 else (0.0 if k1 == 0 else 1.0)
    ok = drift <= 0.20
    print(("PASS " if ok else "FAIL ")
          + "A5 QUORUM-MIRROR keep drift %.1f%% (<=20%%)" % (100 * drift))
    return ok


def main():
    try:
        track = run_pr1_track(use_graded=True, want_touch=True)
    except OSError as err:
        print("P2 ATD contest audit: missing gated data, nothing to run.\n%s"
              % err)
        return 0
    print("A0 anchors=%d (touch proxy frames=%d)"
          % (len(track["anchors"]), int(track["touch"].sum())))
    a1 = audit_a1(track)
    a2 = audit_a2(track)
    a5 = audit_a5()
    print("contest audits: A1=%s A2=%s A5=%s (PROXY/QUORUM-MIRROR labeled)"
          % (a1, a2, a5))
    return 0


if __name__ == "__main__":
    sys.exit(main())
