"""W3-P2 label-stream experiments: T1-T10 proxy tests + E0-E6 arm matrix.

Reads committed baseline CSVs + public GT only (no gated data). Writes arm
CSVs under --out-dir (default /tmp/p2; must be under /tmp — never the repo).
Run from repo root: python3 starter/p2_label_experiments.py [--suite ...]

Mechanism tests (T1-T7, T9-score) assert and fail loudly on regression.
Experiment arms and falsifiers are measurements: printed verdicts, recorded
tables, judged in the competition report — never tuned against.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
from starter import possession as P  # noqa: E402
from starter.possession import (d_pass_prune, flat_prune, graded_prune,  # noqa: E402
                                ha_graded_pass, rle_runs)
import possession_metrics as pm  # noqa: E402

RAW_CSV = os.path.join(ROOT, "evaluation", "results", "baseline_raw.csv")
S50_CSV = os.path.join(ROOT, "evaluation", "results", "baseline_smooth50.csv")
GT_CSV = os.path.join(ROOT, "evaluation", "gt",
                      "BAR-ATH_possession_gt_public.csv")
W1_BEST_CSV = "/tmp/w1/csv/H_raw_maj100_pr100alive.csv"  # soft cross-check only

CODE = {"H": 0, "A": 1, "D": 2}


def read_labels(path):
    frame = pm.read_label_csv(path)
    err = pm.validate_submission(frame, len(frame))
    assert err is None, (path, err)
    return np.array([str(v).strip().upper() for v in frame["label"].values],
                    dtype="<U1")


def maj100(labels):
    """Centered 3-class majority +-100, ties keep center (w1-postproc §4)."""
    assert set(np.unique(labels)) <= {"H", "A", "D"}, set(np.unique(labels))
    n = len(labels)
    acc = np.zeros((3, n + 1), dtype=np.int64)
    for k, lab in enumerate("HAD"):
        acc[k, 1:] = np.cumsum(labels == lab)
    idx = np.arange(n)
    cnt = acc[:, np.minimum(n, idx + 101)] - acc[:, np.maximum(0, idx - 100)]
    top = cnt.max(axis=0)
    out = np.array(list("HAD"))[np.argmax(cnt, axis=0)].astype("<U1")
    out[(cnt == top).sum(axis=0) > 1] = labels[(cnt == top).sum(axis=0) > 1]
    return out


def write_csv(path, labels):
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab)
                          for i, lab in enumerate(labels))


def score_roundtrip(name, labels, gt_codes, out_dir, bootstrap_n=0):
    """Write CSV under out_dir, read back via the scorer, assert equality."""
    path = os.path.join(out_dir, name + ".csv")
    write_csv(path, labels)
    frame = pm.read_label_csv(path)
    err = pm.validate_submission(frame, len(labels))
    assert err is None, (name, err)
    pred = pm.label_codes(frame)
    expect = np.array([CODE[v] for v in labels], dtype=np.int8)
    assert (pred == expect).all(), name
    return pm.evaluate(gt_codes, pred, bootstrap_n=bootstrap_n)


def count_guard(n_pred, scope):
    """Abort tripwire: public n_pred<10 or full-match n_pred<300 (cap 4)."""
    return n_pred < (10 if scope == "public" else 300)


# -- arms: every arm reuses shipped possession.py logic, no forks -------------

def arm_e0(front, log):
    return flat_prune(front, 100, "alive")


def arm_e1(front, log):
    dpass, dropped = d_pass_prune(front)
    log["dropped_d_runs"] = dropped
    return flat_prune(dpass, 100, "alive", classes=("H", "A"))


def arm_e2(front, log):
    dflat = flat_prune(front, 100, "alive", classes=("D",))
    return ha_graded_pass(dflat, floor_only=True, log=log)


def arm_e3(front, log):
    dflat = flat_prune(front, 100, "alive", classes=("D",))
    return ha_graded_pass(dflat, keep=("differ-flank",), cap_source=False,
                          log=log)


def arm_e4(front, log):
    # Diagnostic: capped exemption with prov=None blocks ALL rescue on label
    # streams (the true E4, cap WITH provenance, is PR1-track-only -> A2).
    dflat = flat_prune(front, 100, "alive", classes=("D",))
    return ha_graded_pass(dflat, keep=("differ-flank",), cap_source=True,
                          log=log)


def arm_e5(front, log):
    dflat = flat_prune(front, 100, "alive", classes=("D",))
    return ha_graded_pass(dflat, keep=("differ-flank", "alternation"),
                          cap_source=False, log=log)


def arm_e6(front, log):
    return graded_prune(front, log=log)


ARMS = (("E0", arm_e0), ("E1", arm_e1), ("E2", arm_e2), ("E3", arm_e3),
        ("E4", arm_e4), ("E5", arm_e5), ("E6", arm_e6))


def stream(pieces):
    return np.concatenate([np.full(n, lab, dtype="<U1") for n, lab in pieces])


def run_lengths(out):
    return [(e - s, str(lab)) for s, e, lab in rle_runs(out)]


def test_t1_flicker_kill():
    s = stream([(500, "H")])
    for at, ln in ((100, 5), (250, 10), (400, 15)):
        s[at:at + ln] = "A"
    assert run_lengths(ha_graded_pass(s, floor_only=True)) == [(500, "H")]
    print("T1 flicker kill: 5/10/15f flips removed: PASS")


def test_t2_contest_rescue():
    # Uncapped differ-flank keep (E3 rule): H-run between A and H survives.
    for ln in (30, 60, 90):
        s = stream([(300, "A"), (ln, "H"), (50, "D"), (300, "H")])
        got = run_lengths(ha_graded_pass(s, keep=("differ-flank",),
                                         cap_source=False))
        assert (ln, "H") in got, (ln, got)
    print("T2 contest rescue: 30/60/90f differ-flank runs kept: PASS")


def test_t3_chain_death():
    s = stream([(200, "A"), (8, "H"), (8, "A"), (8, "H"), (8, "A"),
                (200, "H")])
    log = {}
    got = run_lengths(ha_graded_pass(s, log=log))
    assert got == [(216, "A"), (216, "H")], got
    assert log["ha_iters"] <= 3, log["ha_iters"]
    print("T3 flicker-chain death -> 2 runs in %d iters: PASS"
          % log["ha_iters"])


def test_t4_alternation():
    s = stream([(200, "A"), (40, "H"), (200, "A")])
    log = {}
    got = ha_graded_pass(s, keep=("differ-flank", "alternation"),
                         cap_source=False, log=log)
    assert run_lengths(got) == [(200, "A"), (40, "H"), (200, "A")]
    assert log["contest_keep"] and log["contest_keep"][0][3] == "alternation"
    # Solid flanks required: 8f flanks do not protect the sandwich.
    s = stream([(200, "A"), (8, "H"), (40, "A"), (8, "H"), (200, "A")])
    got = run_lengths(ha_graded_pass(s, keep=("differ-flank", "alternation"),
                                     cap_source=False))
    assert (40, "A") not in got, got
    print("T4 genuine alternation kept, weak-flank sandwich dropped: PASS")


def test_t5_d_bridge():
    w, dropped = d_pass_prune(stream([(300, "H"), (60, "D"), (300, "H")]))
    assert run_lengths(w) == [(660, "H")] and dropped == [(300, 360)]
    w, dropped = d_pass_prune(stream([(300, "H"), (60, "D"), (300, "A")]))
    assert run_lengths(w) == [(300, "H"), (60, "D"), (300, "A")] and not dropped
    w, dropped = d_pass_prune(stream([(300, "H"), (200, "D"), (300, "A")]))
    assert (200, "D") in run_lengths(w) and not dropped
    w, dropped = d_pass_prune(stream([(300, "H"), (10, "D"), (300, "A")]))
    assert (10, "D") not in run_lengths(w) and dropped == [(300, 310)]
    log = {}
    graded_prune(stream([(300, "H"), (60, "D"), (300, "H")]), log=log)
    assert log["dropped_d_runs"] and log["d_restored"]  # floor restores it
    print("T5 D-bridge/protect/flicker + floor restore: PASS")


def test_t6_edges():
    assert run_lengths(ha_graded_pass(stream([(15, "H"), (300, "A")]))) \
        == [(315, "A")]
    assert run_lengths(ha_graded_pass(stream([(300, "A"), (15, "H")]))) \
        == [(315, "A")]
    assert run_lengths(graded_prune(stream([(500, "H")]))) == [(500, "H")]
    print("T6 edge shorts absorbed, single-run stream kept: PASS")


def test_t7_fixpoint_idempotence():
    s = stream([(200, "A"), (15, "H"), (15, "A"), (200, "H")])
    once = graded_prune(s, apply_floor=False)
    assert run_lengths(once) == [(215, "A"), (215, "H")], run_lengths(once)
    m = stream([(400, "H"), (30, "D"), (500, "A"), (45, "H"), (60, "X"),
                (400, "A"), (200, "D"), (600, "H")])
    assert (graded_prune(graded_prune(m)) == graded_prune(m)).all()
    print("T7 back-to-back shorts + idempotence: PASS")


def run_synthetic():
    test_t1_flicker_kill()
    test_t2_contest_rescue()
    test_t3_chain_death()
    test_t4_alternation()
    test_t5_d_bridge()
    test_t6_edges()
    test_t7_fixpoint_idempotence()


def full_video_spells(labels):
    return sum(1 for _, _, lab in rle_runs(labels) if lab in ("H", "A"))


def row_of(name, res, labels, log):
    per = res["spell_f1"]
    return {
        "arm": name,
        "spellF1": res["spell_f1_mean"],
        "p03": per[0.3]["f1"], "p04": per[0.4]["f1"], "p05": per[0.5]["f1"],
        "p06": per[0.6]["f1"], "p07": per[0.7]["f1"],
        "m03": per[0.3]["n_matched"], "pr03": per[0.3]["precision"],
        "n_pred": res["n_pred_spells"], "n_gt": res["n_gt_spells"],
        "macroF1": res["macro_f1"],
        "D_f1": res["per_class"]["D"]["f1"],
        "D_pr": res["per_class"]["D"]["precision"],
        "D_re": res["per_class"]["D"]["recall"],
        "tr12": res["transitions"][12]["f1"],
        "tr25": res["transitions"][25]["f1"],
        "frag": res["fragmentation"],
        "full_n": full_video_spells(labels),
        "d_share": float(np.mean(labels == "D")),
        "kept": len(log.get("contest_keep", [])),
        "dropped": len(log.get("contest_drop", [])),
        "d_dropped": len(log.get("dropped_d_runs", [])),
        "d_restored": len(log.get("d_restored", [])),
        "iters": log.get("ha_iters", 0),
    }


def print_table(rows):
    head = ("arm spellF1 @0.3 @0.4 @0.5 @0.6 @0.7 m@.3 n_pr/n_gt macroF1 "
            "D-F1 tr12 tr25 frag full_n d% keep/drop d-/d+ it")
    print(head)
    for r in rows:
        print("{arm} {spellF1:.4f} {p03:.3f} {p04:.3f} {p05:.3f} {p06:.3f} "
              "{p07:.3f} {m03:>4d} {n_pred:>3d}/{n_gt:<2d} {macroF1:.4f} "
              "{D_f1:.3f} {tr12:.3f} {tr25:.3f} {frag:.2f} {full_n:>4d} "
              "{d_share:5.3f} {kept:>2d}/{dropped:<2d} "
              "{d_dropped:>2d}/{d_restored:<2d} {iters:>2d}".format(**r))


def run_matrix(base_name, base_labels, gt_codes, out_dir, bootstrap_n):
    print("--- base: %s (front = maj100) ---" % base_name)
    front = maj100(base_labels)
    rows, outs = [], {}
    for name, fn in ARMS:
        log = {}
        labels = fn(front, log)
        boot = bootstrap_n if (base_name == "raw" and name in ("E0", "E6")) \
            else 0
        res = score_roundtrip("%s_%s" % (base_name, name), labels, gt_codes,
                              out_dir, bootstrap_n=boot)
        if boot:
            print("%s CI: spellF1 [%.3f, %.3f] macroF1 [%.3f, %.3f]"
                  % (name, res["spell_bootstrap_low"],
                     res["spell_bootstrap_high"], res["bootstrap_low"],
                     res["bootstrap_high"]))
        rows.append(row_of(name, res, labels, log))
        outs[name] = (labels, log, res)
    print_table(rows)
    return rows, outs, front


def verdict(cond, text):
    print(("PASS " if cond else "FAIL ") + text)
    return bool(cond)


def judge_matrix(rows, base_name):
    print("--- falsifiers (%s base) ---" % base_name)
    by = {r["arm"]: r for r in rows}
    c = by["E0"]
    out = {}
    out["E1"] = verdict(by["E1"]["D_f1"] > c["D_f1"]
                        and 11 <= by["E1"]["n_pred"] <= 15,
                        "E1: D-F1 %.3f>%.3f and n_pred %d in 11-15"
                        % (by["E1"]["D_f1"], c["D_f1"], by["E1"]["n_pred"]))
    out["E2"] = verdict(by["E2"]["p03"] >= c["p03"] - 0.05,
                        "E2: @0.3 %.3f >= control %.3f-0.05"
                        % (by["E2"]["p03"], c["p03"]))
    out["E3"] = verdict(by["E3"]["m03"] >= by["E2"]["m03"] + 1,
                        "E3: @0.3 matches %d >= E2 %d+1"
                        % (by["E3"]["m03"], by["E2"]["m03"]))
    infl = (by["E4"]["n_pred"] - by["E0"]["n_pred"]) / by["E0"]["n_pred"]
    out["E4"] = verdict(abs(infl) <= 0.15,
                        "E4 diagnostic: n_pred inflation %+.1f%% within 15%%"
                        % (100 * infl))
    out["E6"] = verdict(by["E6"]["m03"] >= 11
                        and 13 <= by["E6"]["n_pred"] <= 18
                        and by["E6"]["p06"] >= c["p06"]
                        and by["E6"]["p07"] >= c["p07"]
                        and by["E6"]["d_share"] >= 0.15,
                        "E6: m@.3=%d(>=11) n=%d(13-18) @.6=%.3f @.7=%.3f "
                        "d%%=%.3f" % (by["E6"]["m03"], by["E6"]["n_pred"],
                                     by["E6"]["p06"], by["E6"]["p07"],
                                     by["E6"]["d_share"]))
    out["K1"] = verdict(by["E6"]["m03"] >= c["m03"],
                        "K1: E6 @0.3 matches %d >= control %d"
                        % (by["E6"]["m03"], c["m03"]))
    out["K2"] = verdict(by["E6"]["pr03"] >= c["pr03"] - 0.05,
                        "K2: E6 @0.3 precision %.3f >= control %.3f-0.05"
                        % (by["E6"]["pr03"], c["pr03"]))
    e6, e0 = by["E6"]["n_pred"], by["E0"]["n_pred"]
    out["K3"] = verdict((e6 - e0) / e0 <= 0.15 or by["E6"]["m03"] > c["m03"],
                        "K3: inflation %+.1f%% with %d new @0.3 matches"
                        % (100 * (e6 - e0) / e0, by["E6"]["m03"] - c["m03"]))
    out["guard_pub"] = verdict(not count_guard(by["E6"]["n_pred"], "public"),
                               "guard: public n_pred %d trips abort: %s"
                               % (by["E6"]["n_pred"],
                                  count_guard(by["E6"]["n_pred"], "public")))
    out["guard_full"] = verdict(not count_guard(by["E6"]["full_n"], "full"),
                                "guard/K5: full-video spells %d trips abort: "
                                "%s" % (by["E6"]["full_n"],
                                        count_guard(by["E6"]["full_n"],
                                                    "full")))
    return out


T8_VARIANTS = {"base": (20, 100, 150, 20), "ha15": (15, 100, 150, 15),
               "ha25": (25, 100, 150, 25), "c75": (20, 75, 150, 20),
               "c125": (20, 125, 150, 20), "d112": (20, 100, 112, 20),
               "d188": (20, 100, 188, 20)}


def run_t8(raw_labels, gt_codes, out_dir):
    """Plateau stability: move 20/100/150 +-25%, require sign-stability of
    the (E6-E0) spellF1 delta. Bands stay frozen whatever the outcome."""
    print("--- T8 plateau stability (raw base; bands NOT refit) ---")
    front = maj100(raw_labels)
    saved = (P.HA_FLICKER_THR, P.CONTEST_THR, P.D_PROTECT_THR, P.ALT_FLANK_THR)
    e0 = score_roundtrip("t8_E0", arm_e0(front, {}), gt_codes,
                         out_dir)["spell_f1_mean"]
    base_delta, bad = None, []
    try:
        for name, (ha, co, dp, alt) in sorted(T8_VARIANTS.items()):
            P.HA_FLICKER_THR, P.CONTEST_THR = ha, co
            P.D_PROTECT_THR, P.ALT_FLANK_THR = dp, alt
            got = score_roundtrip("t8_E6_%s" % name, arm_e6(front, {}),
                                  gt_codes, out_dir)["spell_f1_mean"]
            delta = got - e0
            if name == "base":
                base_delta = delta
            mark = ""
            if base_delta is not None and name != "base" and delta * base_delta < 0:
                mark, bad = "  RANK-FLIP", bad + [name]
            print("T8 %-4s E6=%.4f delta=%+.4f%s" % (name, got, delta, mark))
    finally:
        (P.HA_FLICKER_THR, P.CONTEST_THR, P.D_PROTECT_THR,
         P.ALT_FLANK_THR) = saved
    assert (P.HA_FLICKER_THR, P.CONTEST_THR, P.D_PROTECT_THR,
            P.ALT_FLANK_THR) == (20, 100, 150, 20)
    verdict(not bad, "T8: no rank flip under +-25%% band moves (bad=%s)" % bad)
    return bad


def run_t9(raw_labels, gt_codes, out_dir):
    """Control reproduction: E0 must score 0.5200 with 13 spells; bit-match
    against the w1 CSV is a soft cross-check (scratch may be absent)."""
    print("--- T9 control reproduction ---")
    e0 = arm_e0(maj100(raw_labels), {})
    res = score_roundtrip("t9_E0", e0, gt_codes, out_dir)
    assert abs(res["spell_f1_mean"] - 0.52) < 1e-9, res["spell_f1_mean"]
    assert res["n_pred_spells"] == 13, res["n_pred_spells"]
    print("T9 E0 = 0.5200 / 13 spells: PASS")
    if os.path.isfile(W1_BEST_CSV):
        w1 = read_labels(W1_BEST_CSV)
        assert (w1 == e0).all(), "E0 differs from w1 best CSV"
        print("T9 bit-match vs w1 CSV: PASS")
    else:
        print("T9 bit-match vs w1 CSV: SKIP (scratch absent)")


def run_t10(gt_codes, out_dir):
    """Guard predicate checks + raw-feed inflation measurement (reported)."""
    print("--- T10 guard + raw-feed ---")
    assert count_guard(9, "public") and not count_guard(10, "public")
    assert count_guard(299, "full") and not count_guard(300, "full")
    print("T10 guard trips at n<10 public / n<300 full: PASS")
    raw = read_labels(RAW_CSV)
    mod = graded_prune(raw, apply_floor=False)
    flat = flat_prune(raw, 100, "alive")
    rm = score_roundtrip("t10_mod_raw", mod, gt_codes, out_dir)
    rf = score_roundtrip("t10_flat_raw", flat, gt_codes, out_dir)
    infl = (rm["n_pred_spells"] - rf["n_pred_spells"]) / rf["n_pred_spells"]
    print("T10 raw-feed: module n=%d flat n=%d inflation=%+.1f%% guard=%s"
          % (rm["n_pred_spells"], rf["n_pred_spells"], 100 * infl,
             count_guard(rm["n_pred_spells"], "public")))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", default="all",
                    choices=("all", "synthetic", "matrix"))
    ap.add_argument("--out-dir", default="/tmp/p2")
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args(argv)
    out_dir = os.path.realpath(args.out_dir)
    roots = {os.path.realpath(os.environ.get("TMPDIR", "/tmp")),
             os.path.realpath("/tmp")}
    assert any(out_dir == r or out_dir.startswith(r + os.sep) for r in roots), \
        "out-dir must be under /tmp, got %s" % out_dir
    os.makedirs(out_dir, exist_ok=True)

    if args.suite in ("all", "synthetic"):
        run_synthetic()
    if args.suite in ("all", "matrix"):
        gt_codes = pm.label_codes(read_labels(GT_CSV))
        raw = read_labels(RAW_CSV)
        run_t9(raw, gt_codes, out_dir)
        rows_raw, _, _ = run_matrix("raw", raw, gt_codes, out_dir,
                                    args.bootstrap)
        judge_matrix(rows_raw, "raw")
        s50 = read_labels(S50_CSV)
        rows_s50, _, _ = run_matrix("s50", s50, gt_codes, out_dir, 0)
        judge_matrix(rows_s50, "s50")
        run_t8(raw, gt_codes, out_dir)
        run_t10(gt_codes, out_dir)
    print("DONE suite=%s out=%s" % (args.suite, out_dir))


if __name__ == "__main__":
    main()
