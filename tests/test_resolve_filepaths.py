from pathlib import Path

import pytest

from utils.dataset_utils import resolve_filepaths

NAS = Path("/mnt/FiftyOneSentry/Database")
THERMAL = ("TnFoV", "TwFoV", "_T1_")  # Sentry narrow/wide, Watchkeeper thermal


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return str(path)


def test_uses_16bit_png_when_it_exists(tmp_path):
    fp8 = _touch(tmp_path / "Thermal_Images_8Bit" / "trip" / "a_frame0.jpg")
    fp16 = _touch(tmp_path / "Thermal_Images_16Bit" / "trip" / "a_frame0.png")
    assert resolve_filepaths([fp8]) == ([fp16], 1, 0)


def test_keeps_original_when_no_16bit_file(tmp_path):
    fp8 = _touch(tmp_path / "Thermal_Images_8Bit" / "trip" / "a_frame0.jpg")
    assert resolve_filepaths([fp8]) == ([fp8], 1, 1)


def test_leaves_unchanged_paths_alone():
    wk = "/data/Watchkeeper_Images/Thermal_Images/trip/seq/v_frame0.png"
    assert resolve_filepaths([wk]) == ([wk], 0, 0)


def test_mixed_list_counts(tmp_path):
    with16 = _touch(tmp_path / "Thermal_Images_8Bit" / "t" / "a_frame0.jpg")
    out16 = _touch(tmp_path / "Thermal_Images_16Bit" / "t" / "a_frame0.png")
    without16 = _touch(tmp_path / "Thermal_Images_8Bit" / "t" / "b_frame0.jpg")
    wk = "/data/Watchkeeper_Images/Thermal_Images/t/s/v_frame0.png"
    assert resolve_filepaths([with16, without16, wk]) == ([out16, without16, wk], 2, 1)


def test_empty_list():
    assert resolve_filepaths([]) == ([], 0, 0)


def _write_frame0(video, dest):
    cv2 = pytest.importorskip("cv2")
    cap = cv2.VideoCapture(str(video))
    ok, img = cap.read()
    cap.release()
    assert ok, f"could not read {video}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), img)


# (video root, image layout, trip, sequence, expected)
RECORDINGS = [
    ("WatchKeeper_Videos", "watchkeeper", "Watchkeeper_2025_04_Cascais_Seatrial", "Watchkeeper_recordings_2025_04_08_15_45_49", "not rewritten"),
    ("Sentry_Videos", "sentry", "Sentry_2026_07_23_PLF_Seatrial", "Sentry_recordings_2026_07_24_00_12_00", "fallback"),
    ("Sentry_Videos", "sentry", "Sentry_2022_07_Dubai", "Sentry_recordings_2022_07_13_16_08_04", "16-bit"),
]


@pytest.mark.skipif(not NAS.exists(), reason="NAS not mounted")
@pytest.mark.parametrize("root, layout, trip, seq, expected", RECORDINGS, ids=["watchkeeper", "sentry-no-16bit", "sentry-16bit"])
def test_reference_recordings(tmp_path, root, layout, trip, seq, expected):
    """Cut frame 0 of each thermal video into a temporary image tree, then resolve its path."""
    v8 = NAS / root / "Videos_8Bit" / trip / seq
    v16 = NAS / root / "Videos_16Bit" / trip / seq
    stems = sorted(p.stem for p in v8.glob("*.mp4") if any(t in p.stem for t in THERMAL))
    assert stems, "no thermal .mp4 found"
    for stem in stems:
        if layout == "sentry":
            fp8 = tmp_path / "Thermal_Images_8Bit" / trip / seq / f"{stem}_frame0.jpg"
        else:  # Watchkeeper paths contain neither "8Bit" nor "jpg"
            fp8 = tmp_path / "Watchkeeper_Images" / "Thermal_Images" / trip / seq / f"{stem}_frame0.png"
        _write_frame0(v8 / f"{stem}.mp4", fp8)
        fp16 = tmp_path / "Thermal_Images_16Bit" / trip / seq / f"{stem}_frame0.png"
        if layout == "sentry" and (v16 / f"{stem}.avi").exists():
            _write_frame0(v16 / f"{stem}.avi", fp16)

        resolved, n_rewritten, n_fallback = resolve_filepaths([str(fp8)])
        if expected == "not rewritten":
            assert (resolved, n_rewritten, n_fallback) == ([str(fp8)], 0, 0)
        elif expected == "fallback":
            assert (resolved, n_rewritten, n_fallback) == ([str(fp8)], 1, 1)
        else:
            assert (resolved, n_rewritten, n_fallback) == ([str(fp16)], 1, 0)