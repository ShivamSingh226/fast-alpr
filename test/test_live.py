"""Tests for live feed configuration and timestamp handling."""

import csv
import io
import os

from pytest import approx, raises

from demo_server import DemoState
from db.redis import (
    SIGHTING_FIELDS,
    CameraLocation,
    RedisSightingsPublisher,
    create_sightings_publisher,
    env_switch,
    load_camera_locations,
)
from fast_alpr.live import (
    LIVE_CSV_FIELDS,
    Camera,
    LiveFeedRunner,
    _MonotonicPts,
    build_stream_url,
    cameras_for_range,
    choose_cameras,
    load_env_file,
    prompt_camera_range,
    resolve_cameras,
)


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


def test_explicit_camera_ids_do_not_require_catalogue(monkeypatch) -> None:
    def catalogue_must_not_be_loaded(*_args, **_kwargs):
        raise AssertionError("Explicit camera selection should skip catalogue loading")

    monkeypatch.setattr("fast_alpr.live.load_camera_catalogue", catalogue_must_not_be_loaded)

    assert resolve_cameras(["cam01", "cam02", "cam01"]) == [Camera("cam01"), Camera("cam02")]


def test_camera_range_generates_inclusive_padded_ids() -> None:
    assert cameras_for_range(1, 5) == [Camera(f"cam{number:02d}") for number in range(1, 6)]
    assert cameras_for_range(7, 11) == [Camera(f"cam{number:02d}") for number in range(7, 12)]
    assert cameras_for_range(49, 50) == [Camera("cam49"), Camera("cam50")]
    assert choose_cameras(None, 1, 5) == cameras_for_range(1, 5)


def test_camera_range_prompt_retries_invalid_indices() -> None:
    answers = iter(("0", "many", "1", "1", "5"))

    assert prompt_camera_range(lambda _prompt: next(answers)) == (1, 5)


def test_camera_range_rejects_invalid_bounds() -> None:
    with raises(ValueError, match="Start index must be between 1 and 50"):
        cameras_for_range(0, 5)
    with raises(ValueError, match="End index must be between 1 and 50"):
        cameras_for_range(1, 51)
    with raises(ValueError, match="greater than start index"):
        cameras_for_range(5, 5)
    with raises(ValueError, match="greater than start index"):
        cameras_for_range(6, 5)
    with raises(ValueError, match="Both start index and end index"):
        choose_cameras(None, 1, None)


def test_env_file_loads_credentials_without_overriding_environment(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# CCTV access\nexport CCTV_EMAIL=alice@example.com\n"
        'CCTV_PASSWORD="secret=with#symbols" # trailing comment\n'
        "REDIS_URL=redis://localhost:6379/0\n"
        "CAMERA_LOCATIONS_FILE=db/camera_locations.json\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("CCTV_EMAIL", raising=False)
    monkeypatch.setenv("CCTV_PASSWORD", "already-exported")
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("CAMERA_LOCATIONS_FILE", raising=False)

    load_env_file(env_file)

    assert os.environ["CCTV_EMAIL"] == "alice@example.com"
    assert os.environ["CCTV_PASSWORD"] == "already-exported"
    assert os.environ["REDIS_URL"] == "redis://localhost:6379/0"
    assert os.environ["CAMERA_LOCATIONS_FILE"] == "db/camera_locations.json"


def test_demo_state_stores_latest_camera_preview(monkeypatch) -> None:
    class EncodedImage:
        def tobytes(self) -> bytes:
            return b"jpeg-data"

    monkeypatch.setattr("demo_server.cv2.imencode", lambda *_args: (True, EncodedImage()))
    state = DemoState()
    state.set_cameras([Camera("cam01")])

    state.publish_frame("cam01", object(), 20, 1.25)

    assert state.camera_preview("cam01") == b"jpeg-data"
    summary = state.camera_summaries()[0]
    assert summary["camera_id"] == "cam01"
    assert summary["frames"] == 1
    assert summary["frame_number"] == 20
    assert summary["status"] == "LIVE"


def test_redis_publisher_appends_timestamped_location_event() -> None:
    class FakeRedis:
        entry: tuple[str, dict[str, str], dict[str, object]] | None = None

        def ping(self) -> bool:
            return True

        def xadd(self, name, fields, **options) -> str:
            self.entry = (name, dict(fields), options)
            return "1727520000000-0"

    redis_client = FakeRedis()
    publisher = RedisSightingsPublisher(
        {"cam01": CameraLocation(40.7128, -74.006)}, client=redis_client
    )
    row = {
        "camera_id": "cam01",
        "plate": "ab-c 123",
        "captured_at_utc": "2026-09-28T13:37:42.123456+00:00",
        "pts_seconds": 12.5,
        "frame": 125,
        "ocr_confidence_percent": 96.0,
        "detector_confidence_percent": 98.0,
    }

    prepared = publisher.prepare(row)
    entry_id = publisher.publish(prepared)

    assert entry_id == "1727520000000-0"
    assert redis_client.entry is not None
    stream, fields, options = redis_client.entry
    assert stream == "alpr:sightings"
    assert fields["captured_at_utc"] == row["captured_at_utc"]
    assert fields["normalized_plate"] == "ABC123"
    assert fields["latitude"] == "40.7128"
    assert fields["longitude"] == "-74.006"
    assert fields["location_known"] == "true"
    assert fields == prepared
    assert options["maxlen"] == 100_000


def test_camera_location_file_loads_coordinates_and_validates_coverage(tmp_path) -> None:
    location_file = tmp_path / "camera_locations.json"
    location_file.write_text(
        '{"cam01":{"latitude":40.7,"longitude":-74.0},'
        '"cam02":{"lat":34.0,"lon":-118.2}}',
        encoding="utf-8",
    )

    locations = load_camera_locations(location_file)
    publisher = RedisSightingsPublisher(locations, client=object())

    publisher.validate_cameras([Camera("cam01"), Camera("cam02")])
    with raises(ValueError, match=r"Missing coordinates for camera.*cam03"):
        publisher.validate_cameras([Camera("cam03")])


def test_location_switch_defaults_off_and_uses_zero_coordinates(monkeypatch) -> None:
    monkeypatch.delenv("CAMERA_LOCATIONS_ENABLED", raising=False)

    assert env_switch("CAMERA_LOCATIONS_ENABLED") is False

    class FakeRedis:
        fields: dict[str, str] | None = None

        def ping(self) -> bool:
            return True

        def xadd(self, _name, fields, **_options) -> str:
            self.fields = dict(fields)
            return "1-0"

    client = FakeRedis()
    publisher = create_sightings_publisher(
        [Camera("cam01")], "redis://localhost:6379/0", None, client=client
    )

    assert publisher is not None
    assert publisher.locations["cam01"] == CameraLocation(0.0, 0.0)
    assert publisher.location_known is False
    publisher.publish({"camera_id": "cam01", "plate": "ABC123"})
    assert client.fields is not None
    assert client.fields["latitude"] == "0.0"
    assert client.fields["longitude"] == "0.0"
    assert client.fields["location_known"] == "false"


def test_enabled_location_mode_requires_a_location_file() -> None:
    with raises(ValueError, match="CAMERA_LOCATIONS_FILE"):
        create_sightings_publisher(
            [Camera("cam01")], "redis://localhost:6379/0", None,
            locations_enabled=True,
        )


def test_unknown_location_mode_publishes_zero_coordinates() -> None:
    class FakeRedis:
        fields: dict[str, str] | None = None

        def xadd(self, _name, fields, **_options) -> str:
            self.fields = dict(fields)
            return "1-0"

    client = FakeRedis()
    publisher = RedisSightingsPublisher(
        {"cam01": CameraLocation(0.0, 0.0)}, client=client, location_known=False
    )
    publisher.publish({"camera_id": "cam01", "plate": "ABC123"})

    assert client.fields is not None
    assert client.fields["latitude"] == "0.0"
    assert client.fields["longitude"] == "0.0"
    assert client.fields["location_known"] == "false"


def test_live_runner_publishes_rows_after_writing_csv(monkeypatch) -> None:
    published: list[dict[str, object]] = []
    published_frames: list[tuple[str, object, int, float | None]] = []
    row = {
        "camera_id": "cam01",
        "frame": 0,
        "pts_seconds": 0.0,
        "plate": "ABC123",
        "ocr_confidence_percent": 95.0,
        "detector_confidence_percent": 98.0,
    }

    class FakeAlpr:
        def predict(self, frame):
            assert frame is not None
            return []

    runner = LiveFeedRunner(
        [Camera("cam01")],
        "hls",
        "detections.csv",
        alpr=FakeAlpr(),
        on_row=published.append,
    )
    runner.set_frame_callback(lambda *frame_data: published_frames.append(frame_data))

    class FakeCapture:
        def isOpened(self) -> bool:  # noqa: N802 - OpenCV capture API name
            return True

        def read(self) -> tuple[bool, object]:
            runner.stop()
            return True, object()

        def get(self, _property_id: int) -> float:
            return 0.0

        def release(self) -> None:
            pass

    monkeypatch.setattr("fast_alpr.live.cv2.VideoCapture", lambda *_args: FakeCapture())
    monkeypatch.setattr(runner, "_rows", lambda *_args: [row])
    csv_file = io.StringIO()
    writer = csv.DictWriter(csv_file, fieldnames=LIVE_CSV_FIELDS)
    writer.writeheader()

    runner._process_camera(Camera("cam01"), None, None, writer, csv_file)

    assert len(published) == 1
    csv_row = next(csv.DictReader(io.StringIO(csv_file.getvalue())))
    assert set(csv_row) == set(SIGHTING_FIELDS)
    assert published[0] == csv_row
    assert published[0]["location_known"] == "false"
    assert published[0]["latitude"] == published[0]["longitude"] == "0.0"
    assert published[0]["normalized_plate"] == row["plate"]
    assert published[0]["captured_at_utc"].endswith("+00:00")
    assert len(published_frames) == 1
    assert published_frames[0][0] == "cam01"
    assert published_frames[0][2:] == (0, 0.0)
    assert "ABC123" in csv_file.getvalue()
