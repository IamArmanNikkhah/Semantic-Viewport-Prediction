
import sys
from pathlib import Path

import numpy as np
import pytest

from common.interfaces import TILE_ROWS, TILE_COLS, DEFAULT_CLASSES
from semantics.detection_type import Detection
from semantics import build_priors


# semantic_data_loader
def test_semantic_data_loader_calls_json_parser(monkeypatch):
    called = {}

    def fake_parse_json_log(log_file_path):
        called["path"] = log_file_path
        return ["json_result"]

    monkeypatch.setattr(build_priors, "parse_json_log", fake_parse_json_log)

    result = build_priors.semantic_data_loader("some_log.json")

    assert result == ["json_result"]
    assert isinstance(called["path"], Path)
    assert called["path"].name == "some_log.json"


def test_semantic_data_loader_calls_csv_parser(monkeypatch):
    called = {}

    def fake_parse_csv_log(log_file_path):
        called["path"] = log_file_path
        return ["csv_result"]

    monkeypatch.setattr(build_priors, "parse_csv_log", fake_parse_csv_log)

    result = build_priors.semantic_data_loader("some_log.csv")

    assert result == ["csv_result"]
    assert isinstance(called["path"], Path)
    assert called["path"].name == "some_log.csv"

# implementation we have now works so we should have a result of none 
def test_semantic_data_loader_unknown_extension_returns_none():
    result = build_priors.semantic_data_loader("log.txt")
    assert result is None



# accumulate_grids
def test_accumulate_grids_basic_sum_same_second():
    classes = DEFAULT_CLASSES
    tile_rows = TILE_ROWS
    tile_cols = TILE_COLS

    class_name = classes[0]

    # all the detections in tile 0 is row 0, col 0
    d1 = Detection(
        timestamp=0.2,
        identified_semantic_class=class_name,
        confidence=0.25,
        tile_id=0,
    )
    d2 = Detection(
        timestamp=0.9,
        identified_semantic_class=class_name,
        confidence=0.81,
        tile_id=0,
    )

    S = build_priors.accumulate_grids([d1, d2], temperature=1.0)

    # second 0 should be the only result 
    assert set(S.keys()) == {0}
    grid = S[0]

    # Correct shape: (K, TILE_ROWS, TILE_COLS)
    K = len(classes)
    assert grid.shape == (K, tile_rows, tile_cols)

    class_idx = classes.index(class_name)
    assert np.isclose(grid[class_idx, 0, 0], 0.25 + 0.81)

    # other classes at that tile should be zero
    if K > 1:
        assert np.allclose(grid[:class_idx, 0, 0], 0.0)
        assert np.allclose(grid[class_idx + 1 :, 0, 0], 0.0)


def test_accumulate_grids_temperature_scaling():
    classes = DEFAULT_CLASSES
    class_name = classes[0]

    confidence = 0.81  # sqrt(0.81) = 0.9
    temperature = 2.0

    d = Detection(
        timestamp=1.9,  # int(1.9) = 1
        identified_semantic_class=class_name,
        confidence=confidence,
        tile_id=0,
    )

    S = build_priors.accumulate_grids([d], temperature=temperature)

    assert set(S.keys()) == {1}
    grid = S[1]
    class_idx = classes.index(class_name)

    expected = confidence ** (1.0 / temperature)  # 0.9
    assert np.isclose(grid[class_idx, 0, 0], expected)


def test_accumulate_grids_tile_indexing_nonzero_row_col():
    classes = DEFAULT_CLASSES
    class_name = classes[0]

    # Pick a tile that is not (0, 0) to verify row/col mapping
    tile_id = TILE_COLS + 1  # row = 1, col = 1 for 6x12
    expected_row = tile_id // TILE_COLS
    expected_col = tile_id % TILE_COLS

    d = Detection(
        timestamp=0.1,
        identified_semantic_class=class_name,
        confidence=1.0,
        tile_id=tile_id,
    )

    S = build_priors.accumulate_grids([d], temperature=1.0)
    grid = S[0]
    class_idx = classes.index(class_name)

    assert np.isclose(grid[class_idx, expected_row, expected_col], 1.0)
    # ensure (0, 0) is still zero
    assert np.isclose(grid[class_idx, 0, 0], 0.0)


def test_accumulate_grids_multiple_seconds_and_classes():
    classes = DEFAULT_CLASSES
    assert len(classes) >= 2
    class_a = classes[0]
    class_b = classes[1]

    d1 = Detection(
        timestamp=0.1,
        identified_semantic_class=class_a,
        confidence=0.5,
        tile_id=0,
    )
    d2 = Detection(
        timestamp=1.1,
        identified_semantic_class=class_b,
        confidence=0.7,
        tile_id=0,
    )

    S = build_priors.accumulate_grids([d1, d2])

    assert set(S.keys()) == {0, 1}

    grid0 = S[0]
    grid1 = S[1]

    idx_a = classes.index(class_a)
    idx_b = classes.index(class_b)

    # at t=0, only class_a is non-zero
    assert np.isclose(grid0[idx_a, 0, 0], 0.5)
    assert np.isclose(grid0[idx_b, 0, 0], 0.0)

    # at t=1, only class_b is non-zero
    assert np.isclose(grid1[idx_b, 0, 0], 0.7)
    assert np.isclose(grid1[idx_a, 0, 0], 0.0)


def test_accumulate_grids_empty_clip_returns_empty_dict():
    S = build_priors.accumulate_grids([])
    assert S == {}



# inserts smooth grids 
def test_smooth_grids_ema_applied_correctly():
    # uses small 1x1x1 grids 
    S = {
        0: np.array([[[1.0]]]),
        1: np.array([[[3.0]]]),
        2: np.array([[[5.0]]]),
    }
    alpha = 0.5

    P = build_priors.smooth_grids(S, alpha=alpha)

    assert list(P.keys()) == [0, 1, 2]

    # first second: no smoothing, just raw
    assert np.allclose(P[0], np.array([[[1.0]]]))

    # P[1] = 0.5*3 + 0.5*1 = 2
    assert np.allclose(P[1], np.array([[[2.0]]]))

    # P[2] = 0.5*5 + 0.5*2 = 3.5
    assert np.allclose(P[2], np.array([[[3.5]]]))


def test_smooth_grids_single_key_returns_same():
    S = {10: np.ones((2, 3, 4))}
    P = build_priors.smooth_grids(S, alpha=0.3)

    assert list(P.keys()) == [10]
    assert np.allclose(P[10], S[10])



# debugging 
def test_debugging_statements_prints_when_debug_true(capsys):
    msg = "hello debug"
    build_priors.debugging_statements(msg, debug=True)

    captured = capsys.readouterr()
    assert "[DEBUG] hello debug" in captured.out


def test_debugging_statements_silent_when_debug_false(capsys):
    msg = "should not appear"
    build_priors.debugging_statements(msg, debug=False)

    captured = capsys.readouterr()
    assert msg not in captured.out



def test_run_calls_loader_and_processing(monkeypatch):
    called = {
        "loader": None,
        "accumulate": None,
        "smooth": None,
        "debug_msgs": [],
    }

    def fake_loader(path):
        called["loader"] = path
        return ["fake_clip"]

    def fake_accumulate(clip, temperature):
        called["accumulate"] = (clip, temperature)
        return {0: np.zeros((1, 1, 1))}

    def fake_smooth(S, alpha):
        called["smooth"] = (S, alpha)
        return S

    def fake_debug(msg, debug=False):
        if debug:
            called["debug_msgs"].append(msg)

    monkeypatch.setattr(build_priors, "semantic_data_loader", fake_loader)
    monkeypatch.setattr(build_priors, "accumulate_grids", fake_accumulate)
    monkeypatch.setattr(build_priors, "smooth_grids", fake_smooth)
    monkeypatch.setattr(build_priors, "debugging_statements", fake_debug)

    log_path = Path("dummy.json")
    build_priors.run(
        log_file_path=log_path,
        debugging=True,
        temperature=2.0,
        alpha=0.4,
        rate_hz=0.5,
    )

    # loader called with Path
    assert called["loader"] == log_path

    # accumulate_grids called with clip and temperature
    assert called["accumulate"] == (["fake_clip"], 2.0)

    # smooth_grids called with S and alpha
    S_passed, alpha_passed = called["smooth"]
    assert isinstance(S_passed, dict)
    assert alpha_passed == 0.4

    # Three debug messages about S, P_sem, and rate_hz
    assert len(called["debug_msgs"]) == 3
    assert "Generated raw S[t] grids for seconds" in called["debug_msgs"][0]
    assert "Generated smoothed P_sem[t] grids for seconds" in called["debug_msgs"][1]
    assert "Cadence (rate_hz): 0.5" in called["debug_msgs"][2]


#parsing the arguments 

def test_parse_arguments_parses_cli(monkeypatch):
    # simulate CLI:
    #   python -m semantics.build_priors log.json --debugging --temperature 2.0 --alpha 0.7 --rate_hz 0.5
    argv = [
        "semantics.build_priors",
        "log.json",
        "--debugging",
        "--temperature",
        "2.0",
        "--alpha",
        "0.7",
        "--rate_hz",
        "0.5",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    args = build_priors.parse_arguments()

    assert args.log_file_path == Path("log.json")
    assert args.debugging is True
    assert args.temperature == pytest.approx(2.0)
    assert args.alpha == pytest.approx(0.7)
    assert args.rate_hz == pytest.approx(0.5)
