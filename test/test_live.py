"""Tests for live feed configuration and timestamp handling."""

from pytest import approx

from fast_alpr.live import Camera, LiveFeedRunner, _MonotonicPts, build_stream_url


def test_rtsp_url_encodes_credentials() -> None:
    url = build_stream_url("cam04", "rtsp", "alice@example.com", "p@ss/word")

    assert url == ("rtsp://alice%40example.com:p%40ss%2Fword@103.250.160.189:8554/stream/cam04")


def test_hls_url_uses_camera_id() -> None:
    assert build_stream_url("cam04", "hls") == "https://cctv.corp8.cloud/cam04/index.m3u8"


def test_pts_remains_monotonic_when_source_restarts() -> None:
    pts = _MonotonicPts()

    assert pts.update(1000) == 1.0
    assert pts.update(1033) == 1.033
    assert pts.update(0) == 1.033
    assert pts.update(33) == approx(1.066)


def test_live_runner_requires_a_camera() -> None:
    try:
        LiveFeedRunner([], "hls", "detections.csv")
    except ValueError as error:
        assert str(error) == "At least one camera is required"
    else:
        raise AssertionError("Expected an empty camera list to be rejected")


def test_camera_is_an_identifier() -> None:
    assert Camera("cam01").camera_id == "cam01"
