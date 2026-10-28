"""P4 ATD experiments A1/A2/A6 + gates E1/E2: kickoff detector on real tracking.

Run from repo root: python starter/p4_kickoff_exp.py
No data -> SKIP with the games.require message (exit 0). With data -> prints
visibility counters, the firing scan, void report, touch latency, trap-span
anchor share, and E1/E2 gate verdicts. Prints only; writes nothing.

Follow-up with data: score a trap-on CSV via
python evaluation/run_evaluation.py --results <dir-with-csv> (CI diagnostic only;
never a chooser -- see /tmp/w3-p4restart.md section 3).
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import (GAME, GAMES, anchor_video_frame, half_video_blocks, require,  # noqa: E402
                   source_frames)
from starter import possession as P  # noqa: E402
from starter.baseline import NOBODY, team_of_each_track  # noqa: E402


def graded_mask(n, halves):
    mask = np.zeros(n, dtype=bool)
    for _, a, b in halves:
        mask[a:b + 1] = True
    return mask


def main():
    try:
        meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME, None)
    except IOError as err:
        print("SKIP (no data): {}".format(err))
        return 0
    pitch = GAMES[GAME]["pitch"]
    n = source_frames("tactical", GAME)
    halves = half_video_blocks(GAME)
    out = P.scan(raw_path, team_of_each_track(meta_path), n, pitch)
    (codes, d1, margin, disp, tracked, nh, na, spd, spdn, slow, ring, bx, by) = out
    graded = graded_mask(n, halves)

    print("== A1 visibility (graded frames) ==")
    dual = int(np.sum(graded & (nh > 0) & (na > 0)))
    single = int(np.sum(graded & ((nh > 0) ^ (na > 0))))
    none = int(np.sum(graded & (nh == 0) & (na == 0)))
    print("dual/single/none: {} / {} / {} of {}".format(
        dual, single, none, int(np.sum(graded))))
    vis = codes != NOBODY
    anchors = np.flatnonzero(vis & (margin > P.MARGIN) & (d1 < P.D1_MAX)
                             & (disp < P.DISP_TAU) & graded)
    sq = anchors[(nh[anchors] == 0) | (na[anchors] == 0)]
    print("PR1 anchors (graded): {}; single-team share: {:.3f}".format(
        len(anchors), len(sq) / max(1, len(anchors))))

    print("== A2 firing scan ==")
    still = P.stillness(spd, slow, spdn)
    spans, voids = P.kickoff_events(tracked, bx, by, disp, still, ring, halves, pitch)
    gated = P.kickoff_spans(tracked, bx, by, disp, still, ring, halves, pitch)
    print("onsets: {} (spans {}, voids {})".format(len(spans) + len(voids), spans, voids))
    for h, a, b in halves:
        print("  half {}: {} span(s)".format(h, sum(a <= k0 <= b for k0, _ in spans)))
    if spans:
        lat = [k1 - k0 for k0, k1 in spans]
        print("touch latency (k1-k0): min {} med {} max {}".format(
            min(lat), int(np.median(lat)), max(lat)))
    e1 = len(gated) == 2
    print("E1 fires-twice gate: {} (gated {})".format("PASS" if e1 else "FAIL", gated))

    print("== E2 H2-head D-fraction ==")
    final, info = P.pipeline(out, pitch, halves, use_trap=True)
    h2 = next(b for b in halves if b[0] == 2)
    ko2 = anchor_video_frame(GAMES[GAME]["kickoff_2nd"])  # published anchor, not GT
    head = final[h2[1]:ko2 + 1]
    dfrac = float(np.sum(head == "D")) / max(1, len(head))
    print("H2 head [{}..{}]: D-fraction {:.3f} over {} frames".format(
        h2[1], ko2, dfrac, len(head)))
    print("E2 head-D gate (>=0.80): {}".format("PASS" if dfrac >= 0.80 else "FAIL"))

    print("== A6 trap-span anchor share ==")
    inset = sum(np.sum((anchors >= s) & (anchors < e)) for s, e in info["spans"])
    span_n = sum(e - s for s, e in info["spans"])
    print("PR1 anchors inside forced-D spans: {} / {} frames ({:.4f})".format(
        inset, span_n, inset / max(1, span_n)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
