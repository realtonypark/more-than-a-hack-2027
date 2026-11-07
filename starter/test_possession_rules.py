"""Focused checks for the PR3 wall/boundary rules in possession.py.

Synthetic arrays only (no match data). Run: .venv/bin/python
starter/test_possession_rules.py  (also pytest-compatible).
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from starter.possession import (bridge_candidates, kick_extend_walls,  # noqa: E402
                                shape_refine_boundaries, slot_spreads,
                                st_restart_walls)
from starter.baseline import HOME  # noqa: E402


def test_kick_extends_until_ball_moves():
    disp = np.zeros(200)
    disp[60:] = 1.0  # ball played at 60
    out = kick_extend_walls([(10, 30, "morphD")], disp)
    assert 60 <= out[0][1] <= 70, out  # extends past 30, stops at the kick
    out = kick_extend_walls([(10, 30, "morphD")], np.zeros(200))
    assert out[0][1] == 30 + 300 or out[0][1] == 200, out  # cap, never > n
    disp = np.full(200, np.nan)  # ball never visible: not stillness
    out = kick_extend_walls([(10, 30, "morphD")], disp)
    assert out[0][1] == 30, out


def test_st_restart_extends_to_setpiece_start():
    final = np.full(400, "D", dtype="<U1")
    final[250:] = "H"
    walls = [(100, 200, "morphD")]
    out = st_restart_walls(walls, final, [(230, "H"), (300, "A")])
    assert out[0][1] == 230, out
    out = st_restart_walls(walls, final, [(230, "A")])  # team mismatch
    assert out[0][1] == 200, out
    out = st_restart_walls(walls, final, [(230 + 301, "H")])  # beyond ST_GAP
    assert out[0][1] == 200, out


def test_shape_refine_needs_fresh_flip_and_lagging_votes():
    final = np.array(["A"] * 300 + ["H"] * 300, dtype="<U1")
    letter = np.array(["A"] * 290 + ["H"] * 310, dtype="<U1")
    sgn = np.array(["A"] * 200 + ["H"] * 400, dtype="<U1")  # shape flips at 200
    out, moves = shape_refine_boundaries(final, letter, sgn, [])
    assert moves == [(300, 200)], moves
    assert (out[200:300] == "H").all()
    old = np.array(["H"] * 600, dtype="<U1")  # no flip in window: no move
    out, moves = shape_refine_boundaries(final, letter, old, [])
    assert moves == [], moves


def test_bridge_absolute_stillness():
    walls = [(100, 200, "morphD")]
    final = np.array(["H"] * 100 + ["D"] * 100 + ["H"] * 100, dtype="<U1")
    frac = np.full(300, 0.4)   # flanks busy (relative dip fails: 0.5 > 0.4)
    frac[100:200] = 0.5        # wall span: players moving, below abs ceiling
    assert bridge_candidates(walls, final, frac, []) == []
    assert bridge_candidates(walls, final, frac, [], abs_thr=0.55) == [(100, 200)]


def test_slot_spreads_rms(tmp_path=None):
    # Three home slots spread out, three away slots together: spreadH > spreadA.
    import tempfile
    row = ("1:"
           "0.00,0.50,NaN;0.20,0.50,NaN;0.00,0.50,NaN;"
           "0.50,0.50,NaN;0.50,0.50,NaN;0.50,0.50,NaN:"
           "0.15,0.50,NaN\n")
    if tmp_path is not None:  # pytest fixture
        path = str(tmp_path / "atd.txt")
        with open(path, "w") as handle:
            handle.write(row * 3)
        _check_spreads(path)
    else:  # direct execution
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "atd.txt")
            with open(path, "w") as handle:
                handle.write(row * 3)
            _check_spreads(path)


def _check_spreads(path):
    team = np.array([HOME, HOME, HOME, 1, 1, 1], dtype=np.int8)
    sh, sa = slot_spreads(path, team, 3, (100.0, 100.0))
    assert sh[0] > 9.0 and sa[0] < 0.1, (sh, sa)


if __name__ == "__main__":
    test_kick_extends_until_ball_moves()
    test_st_restart_extends_to_setpiece_start()
    test_shape_refine_needs_fresh_flip_and_lagging_votes()
    test_bridge_absolute_stillness()
    test_slot_spreads_rms()
    print("all rule checks passed")
