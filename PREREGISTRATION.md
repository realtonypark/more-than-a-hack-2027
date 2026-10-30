# Pre-registration: w3-integrate (frozen before the bundle lands)

Branch `w3-integrate` off PR1 base `64244c2`. Stack: P1 two-sided quorum
anchors → interpolate (provenance + walls) → P2 E6 graded prune → P5
guarded smooth + combination + wall asserts → capped hold-fill per half.
Default walls are K1 morphology; `--walls` selects the HELD trigger path.

Rule: no constant below changes after the first scored ATD run except
through its named gate. Public GT is diagnostic-only, never a selector
(EVALUATION.md:139-142). Anything still free after first scoring is
hill-climbing and fails review.

## Frozen constants (non-GT justifications)

P1 quorum (`starter/possession.py`): Q_MIN=1 (minimum cardinality for
"measured", not tuned). Single-team frames get margin -inf.

P2 graded prune: HA_FLICKER_THR=20 (sub-second kick cycle; GT P10=19f
explicitly NOT used), D_FLICKER_THR=25 (1s stoppage recognition),
CONTEST_THR=100 (frozen GAP), D_PROTECT_THR=150 (throw-in routine ≥6s),
D_SHARE_FLOOR=0.15 (literature 0.15-0.25 lower bound, NOT public 0.196),
ALT_FLANK_THR=20. Recorded sensitivity (not a dial): D-bridge 112-150f
same-team D exists and bridging it hurts (T8 d112 flip).

P5 structure: W_G=50, HEAD_CAP=50 (frozen GAP//2), D_MIN_KEEP=12,
D_CLOSE=12, PRUNE_INPLAY=20, BLINK_MERGE=6, M_LINE=1.5m, K_LINE=5,
N_ONSET=10, N_REAPPEAR=5, W_STILL=25, W_STILL_MIN=12, S_Q=1.5m/s,
F_STILL=0.60, N_WAKE=12, T_CONJ=50, D_MAX_SPAN=1500, ST_WIN_BACK=75,
ST_WIN_FWD=25, FAST_RESTART_S=50.

Stillness scale: V_LO_INIT=1.0, V_HI_INIT=2.0 are START points.
T-ATD-1 derives v_lo/v_hi from ATD stats via the frozen rule (p15 on
ball-NaN windowed means / p40 over graded halves, fallback p10/p50) and
re-registers the two numbers BEFORE the first scored run.

Integration (w3-integrate): morphD walls are smooth50-mask D runs
≥ D_MIN_KEEP (K1 fallback, always available). graded_prune runs GLOBAL
E6 (exactly as measured), not per-episode: X-skip semantics already
prevent wall crossing on the team stage, where wall frames are X.

## Gates (all staged as runnable scripts, all SKIP-clean without data)

- A2 collapse guard (P1): anchor drop ≥25% vs legacy → fall back to
  -inf-guard-only. T4-beats-C0 on plausibility → kill P1.
  (`starter/quorum_audit.py`, `starter/quorum_matrix.py`)
- D-span 55-75 band (P5 detectors): count<30 or >120 → K1 fallback
  stands, detectors stay HELD. H2-head D fraction ≥0.80 (E5).
  (`starter/p5_atd_check.py`, T-ATD-1..6)
- A2 GAP-source audit (P2 rescue): <90% GAP-sourced keeps → fall back
  to E1 (D-pass + flat H/A). (`starter/p2_atd_*.py`)
- E1 fires-twice / E2 head-D ≥80% (P4 trap), E3 agreement ≥75% / E4
  throw-ins (P4 restart): each flag ships only on its own PASS.
  Flags stay default-off. (P4 branch `w3-p4-restart`, not merged)
- F1/F2 (P6 core), F3/F4 (touch/pressure): ship/hold per audit.
  A3 stays benched (E5 once-test). (P6 branch `w3-p6-flipgate`, HOLD)
- P3 C1-form re-test on ATD (halo-on C4 DROPPED: provably spell-inert,
  pure macro cost). (P3 branch `w3-p3-xpolicy`, HOLD)

Dropped by judge, do not revive without new evidence: pitch-control
tiebreak, chain post-merge, truncated votes, standalone clustering,
droop force-D, fitted pressure weights.

## Measured (label streams / synthetic only; public CIs overlap throughout)

- P2 E6: spellF1 0.5548 (E0 control 0.5200), +1 @0.3 match; gain is the
  D-pass (+0.073 alone), H/A rescue untestable without provenance.
  Reproduced exactly on the merged tree. Full-video spells 291.
- P5 machinery: D-F1 +0.10 diag, width-invariant D; half-split has zero
  measured effect (kept as hygiene).
- P1 synthetic: guard 1.00 vs legacy 0.40 on dense-occlusion traps
  (mechanism-sized, NOT a gain claim; judge estimate +1-3).
- P4 synthetic: 13/13 shape, cost model confirmed (wrong restart = IoU 0).
- P6 synthetic: 31/31 gate checks; V2≡raw proof (re-vote, not X, matters).
- Integration: synthetic end-to-end passes (morphD walls, H/A 1.00 on
  scripted blocks, trigger path invariants hold, empty-anchor prov safe).

Full-match expectation if all gates pass: ~0.45-0.60 (judge, honest,
overlapping not additive). 0.6+ is reachable, not banked.

## Bundle-day log (2026-10-30, workbox)

- T-ATD-0: bundle intact (199319f, halves match, ball-NaN 0.431, speeds ok).
- T-ATD-1: v_lo=0.611, v_hi=1.314 RE-REGISTERED (p15-nan/p40-all). Derived
  triggers: 23 spans -> FALSIFIED (<30). K1 morphology stands; P5 detectors
  stay HELD. (INIT 1.0/2.0 gave 55 spans IN-band but INIT values are
  never-shipped start points.) T-ATD-4 trigger H2-head D 0.020, consistent.
- T-ATD-5: long-ST recall 0.13 -> M8 eligible but DEFERRED (its constants
  were judged ungrounded; revisit only on scored long-D misses).
- P1: A1 single-team 1069f (0.9% of vis); A2 C0=T3=26326 anchors, 0 flips
  -> SHIP (behavior-neutral correctness). Matrix: 328 spells in-band.
- P2: A1 PROXY-touch 0.42:1 inconclusive (proxy only, not a kill); A2
  38/88 gap-sourced <90% -> E1 FALLBACK (d_pass + flat H/A @100) wired
  into run_pipeline, contest rescue shelved. A3 kickoff D 0.241 confirms
  the kickoff miss (P4's target; P4 E1/E2 pending). d_audit X-code bug
  (KeyError on valid out-of-half X) fixed.
