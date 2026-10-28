"""W3-P2 ATD D audits A3/A4/A6 (PR1 track; needs the gated bundle).

A3: H2-kickoff region (video 110808-113742) must be >=80% D (published fact).
A4: best-IoU on the D1-adjacent GT spells P5/P6 must not drop vs the frozen
    prune() control (public GT correctness check only, never selection).
A6: kept-H/A length histogram + D share in [0.15,0.30] + spell counts.

Without the bundle this prints a missing-data note and exits 0.
Run from repo root: python3 starter/p2_atd_d_audit.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
from starter.p2_atd_track import full_video_spells, run_pr1_track  # noqa: E402
from starter.possession import rle_runs  # noqa: E402
import possession_metrics as pm  # noqa: E402

GT_CSV = os.path.join(ROOT, "evaluation", "gt",
                      "BAR-ATH_possession_gt_public.csv")
KICKOFF = (110808, 113742)  # published H2-kickoff all-D region (video frames)
CODE = {"H": 0, "A": 1, "D": 2}


def to_codes(labels):
    return np.array([CODE[v] for v in labels], dtype=np.int8)


def audit_a3(final):
    a, b = KICKOFF
    frac = float(np.mean(final[a:b + 1] == "D"))
    print("A3 kickoff D fraction %.3f (frames %d-%d)" % (frac, a, b))
    ok = frac >= 0.80
    print(("PASS " if ok else "FAIL ") + "A3 >=80%% D at H2 kickoff")
    return ok


def audit_a4(mod_final, ctl_final):
    frame = pm.read_label_csv(GT_CSV)
    gt = pm.label_codes(frame)
    assert pm.spell_scores(gt, gt, pm.valid_runs(gt))["n_gt"] == 17
    runs = pm.valid_runs(gt)
    gt_spells = pm.possession_spells(gt, runs)
    outs = {}
    for name, final in (("module", mod_final), ("control", ctl_final)):
        pred = pm.fill_abstentions(to_codes(final), runs)
        overlaps = pm.spell_overlaps(gt_spells,
                                     pm.possession_spells(pred, runs))
        best = [max([iou for iou, g, _ in overlaps if g == gi] or [0.0])
                for gi in (4, 5)]  # P5/P6: 5th/6th GT spells, D1-adjacent
        outs[name] = best
        print("A4 %s P5/P6 best-IoU %.3f/%.3f" % (name, best[0], best[1]))
    ok = all(m >= c for m, c in zip(outs["module"], outs["control"]))
    print(("PASS " if ok else "FAIL ")
          + "A4 module P5/P6 IoU not below control")
    return ok


def audit_a6(track):
    final, log = track["final"], track["log"]
    lens = sorted(e - s for s, e, _, _ in log["contest_keep"])
    bins = {"20-39": 0, "40-69": 0, "70-100": 0}
    for ln in lens:
        bins["20-39" if ln < 40 else "40-69" if ln < 70 else "70-100"] += 1
    print("A6 kept-H/A %d lens=%s bins=%s" % (len(lens), lens[:15], bins))
    d_full = float(np.mean(final == "D"))
    from games import half_video_blocks, GAME
    halves = np.zeros(len(final), dtype=bool)
    for _, a, b in half_video_blocks(GAME):
        halves[a:b + 1] = True
    d_half = float(np.mean(final[halves] == "D"))
    n_full = full_video_spells(final)
    print("A6 D share full=%.3f in-half=%.3f spells=%d" % (d_full, d_half,
                                                          n_full))
    ok = (0.15 <= d_half <= 0.30) and 300 <= n_full <= 900
    print(("PASS " if ok else "FAIL ")
          + "A6 in-half D in [0.15,0.30] and spells in [300,900]")
    return ok


def main():
    try:
        mod = run_pr1_track(use_graded=True)
        ctl = run_pr1_track(use_graded=False)
    except OSError as err:
        print("P2 ATD D audit: missing gated data, nothing to run.\n%s" % err)
        return 0
    a3 = audit_a3(mod["final"])
    a4 = audit_a4(mod["final"], ctl["final"])
    a6 = audit_a6(mod)
    print("D audits: A3=%s A4=%s A6=%s" % (a3, a4, a6))
    return 0


if __name__ == "__main__":
    sys.exit(main())
