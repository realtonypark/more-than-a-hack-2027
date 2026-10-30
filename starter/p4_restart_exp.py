"""P4 ATD experiments A3/A4/A5 + gates E3-E8: restart inference on real data.

Run from repo root: python starter/p4_restart_exp.py
No data -> SKIP with the games.require message (exit 0). With data -> prints
ST-Team x PZ-winner agreement, throw-in decisiveness/stability, RPZ ablation
(diagnostic; ship value stays 1.0m), ST-jitter robustness, the trap x restart
2x2 plus C1/C2 controls, and E3-E8 gate verdicts. Prints only; writes nothing.
numpy-only (spell counts via local RLE, no pandas) so it runs in the starter env.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from starter import possession as P  # noqa: E402
from starter import load_smart_tagging as ST  # noqa: E402
from starter.baseline import HOME, NOBODY, smooth, team_of_each_track  # noqa: E402

MATCH_TOL = 75  # ST restart anchor <-> pipeline D-offset match window (frames)
STAB_LO, STAB_HI = 100, 150  # stability probe window after offset (frames)


def runs_of(labels):
    """[(label, start, end)] maximal runs."""
    chg = np.flatnonzero(np.asarray(labels)[1:] != np.asarray(labels)[:-1]) + 1
    b = np.concatenate(([0], chg, [len(labels)]))
    return [(labels[b[i]], int(b[i]), int(b[i + 1])) for i in range(len(b) - 1)]


def spell_stats(labels, halves):
    """(n_pred H/A spells, mean/median spell len, D-share, D-span count>=12)."""
    n_sp, lens, d_n, d_spans, tot = 0, [], 0, 0, 0
    for _, a, b in halves:
        w = labels[a:b + 1]
        tot += len(w)
        for lab, s, e in runs_of(w):
            if lab in ("H", "A"):
                n_sp += 1
                lens.append(e - s)
            elif lab == "D":
                d_n += e - s
                d_spans += e - s >= P.D_MIN
    return (n_sp, float(np.mean(lens)) if lens else 0.0,
            float(np.median(lens)) if lens else 0.0,
            d_n / max(1, tot), d_spans)


def st_team(tag):
    tag = (tag or "").strip().lower()
    if tag.startswith("barcelona"):
        return "H"
    if tag.startswith("atlet"):
        return "A"
    return None


def main():
    try:
        meta_path, raw_path, _ = require(
            ["atd_meta", "atd_raw", "smart_tagging"], GAME, None)
    except IOError as err:
        print("SKIP (no data): {}".format(err))
        return 0
    pitch = GAMES[GAME]["pitch"]
    n = source_frames("tactical", GAME)
    halves = half_video_blocks(GAME)
    team_of_track = team_of_each_track(meta_path)
    out = P.scan(raw_path, team_of_track, n, pitch)
    (codes, d1, margin, disp, tracked, nh, na, spd, spdn, slow, ring, bx,
     by) = out[:4] + out[6:]
    vis = codes != NOBODY
    letter = np.where(codes == HOME, "H", np.where(vis, "A", "X"))
    alive = np.isin(smooth(codes, 50), ("H", "A"))
    spans = P.kickoff_spans(tracked, bx, by, disp,
                            P.stillness(spd, slow, spdn), ring, halves, pitch)
    gaps = P.d_spans(alive, spans)
    offs = sorted(o for _, o in gaps)
    quorum = (nh > 0) & (na > 0)
    live = (letter == "H") | (letter == "A")
    pz = live & quorum & (d1 < P.RPZ)
    origin = []
    rfr, rtm, abst = P.reseed_anchors(d1, letter, quorum, bx, by, tracked, gaps,
                                     pitch, raw_path, team_of_track,
                                     out_origin=origin)
    # Per-gap attribution: abstains match offsets exactly; anchors carry
    # their originating gap offset (nearest-preceding misattributes when
    # D-spans sit <75f apart and restart windows overlap).
    team_of, kind_of = {}, {}
    for s, _ in abst:
        team_of[s], kind_of[s] = "X", "x"
    for f, t, o in zip(rfr, rtm, origin):
        is_pz = bool(pz[f:f + P.K_RESEED].all()
                     and (letter[f:f + P.K_RESEED] == t).all())
        team_of[o], kind_of[o] = t, ("pz" if is_pz else "fb")

    events = ST.load()
    setpieces = [e for e in events if (e["code"] or "").strip().lower() == "set pieces"]
    print("== ST inventory: {} instances, {} SET PIECES ==".format(
        len(events), len(setpieces)))

    def match(shift=0):
        """ST restart anchors (end + shift) -> (team_st, team_p4, kind, is_throw)."""
        rows = []
        for e in setpieces:
            tags = e["tags"]
            r = e["end_frame"] + shift
            near = [o for o in offs if abs(o - r) <= MATCH_TOL]
            if not near:
                continue
            o = min(near, key=lambda x: abs(x - r))
            is_throw = "throw" in (tags.get("Type") or "").lower()
            rows.append((st_team(tags.get("Team")), team_of.get(o), kind_of.get(o),
                         is_throw, o, r))
        return rows

    rows = match()
    agree = [r for r in rows if r[0] and r[1] in ("H", "A") and not r[3]]
    hits = sum(1 for r in agree if r[0] == r[1])
    print("== A3/E3 agreement (non-throw-in, matched) ==")
    print("matched {}/{} set pieces; agreement {}/{} = {:.3f}".format(
        len(rows), len(setpieces), hits, len(agree),
        hits / max(1, len(agree))))
    over = [(o, s, p, k) for s, p, k, t, o, _ in rows if s and p in ("H", "A") and s != p]
    print("override log (ST disagrees, PZ decides): {} cases{}".format(
        len(over), " e.g. {}".format(over[:5]) if over else ""))
    e3 = hits / max(1, len(agree))
    print("E3 agreement gate (>=0.75 pass, <0.60 kill): {:.3f} -> {}".format(
        e3, "PASS" if e3 >= 0.75 else ("KILL-RESTART" if e3 < 0.60 else "HOLD")))

    print("== E4 throw-ins ==")
    throws = [r for r in rows if r[3]]
    dec = [r for r in throws if r[1] in ("H", "A")]
    print("matched {}/26 throw-ins; decisiveness {}/{} = {:.3f}".format(
        len(throws), len(dec), max(1, len(throws)),
        len(dec) / max(1, len(throws))))

    def stability(rows):
        ok, tot, unk = 0, 0, 0
        for _, t, _, _, o, _ in rows:
            if t not in ("H", "A"):
                continue
            v = letter[o + STAB_LO:o + STAB_HI][
                pz[o + STAB_LO:o + STAB_HI]]
            if len(v) < 3:
                unk += 1
                continue
            tot += 1
            ok += (np.sum(v == t) * 2 >= len(v))
        return ok, tot, unk

    ok, tot, unk = stability(rows)
    okt, tott, unkt = stability(throws)
    print("stability@+5s all restarts: {}/{} ({} unknown) = {:.3f}".format(
        ok, tot, unk, ok / max(1, tot)))
    print("stability@+5s throw-ins: {}/{} ({} unknown) = {:.3f}".format(
        okt, tott, unkt, okt / max(1, tott)))
    kinds = [k for _, _, k, _, _, _ in rows]
    print("(a)/(b)/(c) split: pz={} fb={} x={}".format(
        kinds.count("pz"), kinds.count("fb"), kinds.count("x")))
    print("E4 gate (decisiveness>=0.70, stability>=0.80): {}".format(
        "PASS" if len(dec) / max(1, len(throws)) >= 0.70
        and okt / max(1, tott) >= 0.80 else "FAIL"))

    print("== A4 RPZ ablation (diagnostic; ship stays 1.0m) ==")
    for r in (0.8, 1.0, 1.5):
        P.RPZ = r
        try:
            fr, _, ab = P.reseed_anchors(d1, letter, quorum, bx, by, tracked,
                                        gaps, pitch)  # K-run only: fallback is RPZ-free
        finally:
            P.RPZ = 1.0
        pzz = live & quorum & (d1 < r)
        stab = []
        for f in fr:
            v = letter[f:f + 50][pzz[f:f + 50]]
            if len(v) >= 3:
                stab.append(np.sum(v == letter[f]) * 2 >= len(v))
        print("RPZ={}: K-run gaps {}/{}, stability {:.3f}".format(
            r, len(fr), len(gaps), float(np.mean(stab)) if stab else float("nan")))
    same = []
    for shift in (-25, 25):
        shifted = {(o, t) for _, t, _, _, o, _ in match(shift) if t in ("H", "A")}
        base = {(o, t) for _, t, _, _, o, _ in rows if t in ("H", "A")}
        same.append(len(shifted & base) / max(1, len(base)))
    print("ST-jitter winner-unchanged (-25f/+25f): {:.3f} / {:.3f} (want >0.90)".format(*same))

    print("== A5 2x2 + C1/C2 ==")
    variants = {
        "C0 base": P.pipeline(out, pitch, halves)[0],
        "trap": P.pipeline(out, pitch, halves, use_trap=True)[0],
        "restart": P.pipeline(out, pitch, halves, raw_path, team_of_track,
                              use_restart=True)[0],
        "both": P.pipeline(out, pitch, halves, raw_path, team_of_track, True, True)[0],
        "C1 backcap50": P.pipeline(out, pitch, halves, back_cap=50)[0],
    }
    base_n = None
    for name, lab in variants.items():
        filled = lab.copy()
        for _, a, b in halves:
            filled[a:b + 1] = P.hold_fill(filled[a:b + 1])  # scorer-equivalent fill
        n_sp, mean_l, med_l, d_share, d_sp = spell_stats(filled, halves)
        base_n = base_n if base_n is not None else n_sp
        print("{:<12} n_pred={:>4} ({:+}% vs C0) mean/med {:>6.0f}/{:>5.0f}f "
              "D-share {:.3f} D-spans {}".format(
                  name, n_sp, 100.0 * (n_sp - base_n) / max(1, base_n),
                  mean_l, med_l, d_share, d_sp))
    c2, c2n = 0, 0
    for s, p, _, _, o, _ in rows:
        if not s:
            continue
        v = letter[o:o + P.RESTART_WINDOW][live[o:o + P.RESTART_WINDOW]]
        if not len(v):
            continue
        h = np.sum(v == "H")
        c = "H" if h * 2 > len(v) else ("A" if h * 2 < len(v) else None)
        if c is None:
            continue
        c2n += 1
        c2 += c == s
    print("C2 smoother-majority agreement: {}/{} = {:.3f} (PZ must beat this)".format(
        c2, c2n, c2 / max(1, c2n)))
    decided = sum(1 for o in offs if team_of.get(o) in ("H", "A"))
    print("E7 starvation: decided {}/{} gaps = {:.3f} (want >0.50)".format(
        decided, len(offs), decided / max(1, len(offs))))
    print("E8 not-copying audit: {} overrides (want non-empty)".format(len(over)))
    both_n = spell_stats(variants["both"], halves)[0]
    print("E6 guards: count {:+}% (kill >+15%), D-spans {} (want 30..120)".format(
        round(100.0 * (both_n - base_n) / max(1, base_n)),
        spell_stats(variants["both"], halves)[4]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
