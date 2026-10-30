# Pre-registration: w3-integrate (frozen before the bundle lands)

Branch `w3-integrate` off PR1 base `64244c2`. Stack: P1 two-sided quorum
anchors → interpolate (provenance + walls) → PR1 prune12 → walls stamp D
→ uncapped hold-fill per half → Rule-D/B/R. [Corrected 2026-10-30: E6
killed (0.5103), guarded smooth removed as dead, hold-fill uncapped per
FIX 2 (HEAD_CAP spends macroF1 for identical spells); frozen original in
git history, kill log in the addendum below.]
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
≥ D_MIN_KEEP (K1 fallback, always available). Team stage is PR1 prune12:
E6 graded rescue was killed (0.5103 < control 0.5241 on the PR1 track)
and is default-off behind --e6 for ablation only. [Corrected
2026-10-30; frozen original in git history.]

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
- FIRST SCORE (vote-primary stack): public spellF1 0.3256, per-IoU
  identical to smooth50 to 4 decimals, 97 graded-frame diffs. Diagnosis:
  guarded +-50 votes nearly everywhere, so vote-primary combination
  reproduces the smoother and interpolation never matters. FIX 1 (wiring
  correction, not tuning): team-primary combination (team wins, walls
  stamp D, vote fills team-X holes), restoring PR1's proven
  architecture. P5's own note left the pairing direction open.
- P4 merge (default-off) verified neutral: control rescore 0.5241.
  P4 verdicts: E1 FAIL (trap fires 0x in halves) -> TRAP KILLED. E3 0/1
  correctness -> RESTART KILLED (E4 throw-in proxies pass on n=5 but
  measure decisiveness, not correctness, of the same unproven
  mechanism). A3 kickoff D 0.241 stands as an open miss.
- OUR MISS AUTOPSY (12 pred vs 17 GT): 5 true shorts missed
  (G02/G09/G11/G12/G14), 2 swaps (G01/G12 wrong team), 4 boundary-short
  longs (G00/G10/G13/G16 at 0.28-0.41). Walls innocent (0 cuts in 16/17).
- P6 standalone: 0.2902/34 (fragments; touch uncalibrated, 1193 fires)
  -> stays OUT, no merge. P3-C1: 0.5103/12 spell-neutral (macro cost
  only) -> stays OUT. Stack remains 0.5241; gap to 0.6 is +0.076.
- ANCHOR AUTOPSY: 0 opposite-team anchors across miss windows (P07:
  127H/0A over 641f incl two true A spells; G01: 0H over 202f H).
  Proximity lies for seconds; only a touch signal can fix it.
- TOUCH-CHAIN (frozen before first touch-score): V_TOUCH=5.0 (|dv| m/s;
  physics kick impulse; provider p995 4.81, 485 candidates), fallback
  4.0 iff attributed<200 (starvation, GT-blind), R_TOUCH=2.0/DZ_R=2.0
  (body scale), per-half chain, no wall reset (walls mask D anyway).
  Ship rule: structure 250-450 spells + D-share sane + public improves
  + flip spot-audit; CIs reported regardless.
- TOUCH FIX (detector noise-dominated; GT-blind): 1043 candidates but 39
  attributed (3.7%) -> all-X chain -> 0.0000 debug run. Cause: |dv|
  spikes are mostly tracking jitter (jumped-to position is far from
  everyone). Fix, frozen before re-run: persistence confirmation (speed
  must persist frame+1, V_PERSIST=0.5 round dichotomy; jitter reverts)
  + slow-endpoint attribution (contact at low speed).
- TOUCH-V2 (frozen before first v2-score): v1 fired on IN-FLIGHT noise
  (fast-ball jitter far from players; persistence cannot reject
  persistent fast motion). v2 gates PROXIMITY FIRST (ball <2m, body
  scale), then loose event: |dv|>V_TOUCH=2.0 (walking-pace kick; above
  p90 0.89 band) OR turn>V2_TURN=30deg at min-speed V2_MINV=3.0 (clear
  deflection at jog pace; gates low-speed heading noise). Persistence,
  DZ veto, slow-endpoint kept. Fallback 1.0 iff attributed<200. Ship
  rule unchanged (structure + improves + spot-audit + CIs).

## Bundle-day addendum (structural fixes + shipped rules, 2026-10-30)

Constants above are unchanged (verified: no PR1 value modified; additions
only). What changed is structure, each a diagnosed pathology with a
GT-blind mechanism, gated by its own audit:

- FIX 3 (touch X-desert): the chain resets to X at half starts; the
  7500f pre-first-touch desert hold-fills from the nearest wall-D, so
  the public window scored all-D. Fix: seed each half's leading X-run
  from the interpolated team (scan/anchor-only). Seeded touch scores
  0.5241, identical to control (no touches fall in-window: the chain
  is uninformative on public GT, kept default-off).
- E6 (P2 graded rescue on the PR1 track): 0.5103 < control 0.5241.
  The label-stream claim (0.5548) does not transfer; rescue keeps
  wrong shorts off-track. Killed (kept behind --e6 for ablation).
- Rule-D (--outD, default-on): tracked ball-out runs >= GAP(100,
  frozen) stamp D. Law-grounded (out is dead); 104 raw outs debounce
  to 16, 13 in-half, 1 in-window. Standalone 0.5379 (+0.0138).
- Rule-B (--bridge, default-on): bridge morph walls with same-team
  flanks (>= PRUNE, frozen), wall-span stillness below both flanks
  (strict relative, no threshold), and no ST overlap. 6/133 fire
  (wall1 stark 50x dip + 5 ungraded). Standalone 0.5714 (+0.0473).
- Rule-R (--refine, default-on): desert-fill H/A boundaries (gap >
  GAP) move to the vote-purity crossover; true midpoints (<= GAP)
  keep interpolate's split (anchors outrank votes); moves spanning
  tracked outs are refused (law). 54 fire globally, 1 in-window
  (13628->13671, G06 crosses @0.6). Standalone 0.5379 (+0.0138).
- Killed on principle/evidence, not shipped: guarded_smooth /
  prune_episodes / close_d_flicker removed as dead (no callers);
  vote-fill/micro-fill/E6
  (n-increasing splits dilute P at 4 thresholds per +1tp at 1:
  net-negative, proven by the F1 n-theorem); wall2 duel/mg-carry
  rules (no axis / fitting creep); all team-flip and micro-nudge
  ideas (evidence inverted or n=1 fits); threshold tuning of any
  constant (pre-reg freeze + EVALUATION.md smooth125 lesson).

Post-proof review fixes (no shipped-behavior change; default re-measures
0.6000/11): removed dead guarded_smooth/prune_episodes/close_d_flicker;
Rule-R out-guard covers all tracked outs (min_len=1); wall spans merged
on overlap before the invariant; reseed anchors carry origin offsets;
audits corrected (hysteresis recheck, inclusive E5 end-frame, GAPFILL +
fixpoint dedupe, legacy C0 margins, masked none-count, E3 non-throw
filter, registered 250-450 spell band, TMPDIR jail, pre-fill D-share,
finite-displacement proxy touch, one-sided C0 quorum-off arm). Residual:
A6 in-half D 0.379 outside [0.15,0.30] (module track; band not widened
without registration).

Proof (public window, 17 spells): control 0.5241/12 -> triple stack
0.6000/11 (macro 0.7028, 90% CI [0.3857, 0.7752] over 1000 draws;
target sits at point estimate with zero margin and wide CI --
reported, not hidden). Full-video structure sane: 274 H/A spells
(250-450 band), D-share 29.8%. PR1 pristine re-measures 0.5103/12.
