"""Publish timestamped ALPR sightings and camera coordinates to a Redis Stream."""

from __future__ import annotations

import json
import math
import os
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from fast_alpr.live import Camera

DEFAULT_STREAM_KEY = "alpr:sightings"
DEFAULT_MAX_STREAM_LENGTH = 100_000
SIGHTING_FIELDS = (
    "event_id",
    "camera_id",
    "plate",
    "normalized_plate",
    "latitude",
    "longitude",
    "location_known",
    "captured_at_utc",
    "pts_seconds",
    "frame",
    "ocr_confidence_percent",
    "detector_confidence_percent",
)
TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off", ""}


def env_switch(name: str, default: bool = False) -> bool:
    """Read a boolean environment switch; unset values use the supplied default."""
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise ValueError(f"{name} must be true or false")


class RedisStreamClient(Protocol):
    """Subset of redis-py used by the publisher, injectable for tests."""

    def xadd(
        self,
        name: str,
        fields: Mapping[str, str],
        id: str = "*",
        maxlen: int | None = None,
        approximate: bool = True,
    ) -> str: ...

    def ping(self) -> bool: ...


@dataclass(frozen=True)
class CameraLocation:
    latitude: float
    longitude: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.latitude) or not -90 <= self.latitude <= 90:
            raise ValueError(f"Invalid latitude: {self.latitude}")
        if not math.isfinite(self.longitude) or not -180 <= self.longitude <= 180:
            raise ValueError(f"Invalid longitude: {self.longitude}")


def load_camera_locations(path: str | Path) -> dict[str, CameraLocation]:
    """Read a JSON object mapping camera IDs to latitude/longitude values."""
    location_path = Path(path)
    try:
        payload = json.loads(location_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"Cannot read camera location file: {location_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in camera location file: {location_path}") from error
    if not isinstance(payload, dict):
        raise ValueError("Camera location file must contain a JSON object")

    locations: dict[str, CameraLocation] = {}
    for camera_id, coordinates in payload.items():
        if not isinstance(camera_id, str) or not isinstance(coordinates, dict):
            raise ValueError("Each camera location must map an ID to a coordinate object")
        try:
            latitude = float(coordinates.get("latitude", coordinates.get("lat")))
            longitude = float(coordinates.get("longitude", coordinates.get("lon")))
        except (TypeError, ValueError) as error:
            raise ValueError(f"Missing or invalid coordinates for {camera_id}") from error
        locations[camera_id] = CameraLocation(latitude, longitude)
    return locations


def prepare_sighting(
    row: Mapping[str, object],
    location: CameraLocation,
    location_known: bool,
) -> dict[str, str]:
    """Build a normalized sighting record for both CSV output and Redis Streams."""
    camera_id = row.get("camera_id")
    plate = row.get("plate")
    if not isinstance(camera_id, str) or not isinstance(plate, str) or not plate.strip():
        raise ValueError("A camera ID and recognized plate are required to prepare a sighting")
    captured_at = row.get("captured_at_utc")
    if not isinstance(captured_at, str):
        captured_at = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    return {
        "event_id": str(row.get("event_id") or uuid.uuid4().hex),
        "camera_id": camera_id,
        "plate": plate,
        "normalized_plate": re.sub(r"[^A-Z0-9]", "", plate.upper()),
        "latitude": str(location.latitude),
        "longitude": str(location.longitude),
        "location_known": str(location_known).lower(),
        "captured_at_utc": captured_at,
        "pts_seconds": "" if row.get("pts_seconds") is None else str(row["pts_seconds"]),
        "frame": str(row.get("frame", "")),
        "ocr_confidence_percent": str(row.get("ocr_confidence_percent", "")),
        "detector_confidence_percent": str(row.get("detector_confidence_percent", "")),
    }


class RedisSightingsPublisher:
    """Append detected plates to a Redis Stream with camera and capture-time metadata."""

    def __init__(
        self,
        locations: Mapping[str, CameraLocation],
        redis_url: str | None = None,
        stream_key: str = DEFAULT_STREAM_KEY,
        max_stream_length: int = DEFAULT_MAX_STREAM_LENGTH,
        client: RedisStreamClient | None = None,
        location_known: bool = True,
    ) -> None:
        if not locations:
            raise ValueError("At least one camera location is required")
        if max_stream_length < 1:
            raise ValueError("max_stream_length must be positive")
        self.locations = dict(locations)
        self.stream_key = stream_key
        self.max_stream_length = max_stream_length
        self.location_known = location_known
        self.client = client or self._create_client(redis_url or os.getenv("REDIS_URL"))

    def check_connection(self) -> None:
        """Fail before camera startup if the configured Redis service is unavailable."""
        try:
            self.client.ping()
        except Exception as error:
            raise RuntimeError(
                "Unable to connect to Redis; check REDIS_URL and the Redis service"
            ) from error

    @staticmethod
    def _create_client(redis_url: str | None) -> RedisStreamClient:
        try:
            redis = import_module("redis")
        except ImportError as error:
            raise RuntimeError(
                "Install Redis support with `python -m pip install redis`"
            ) from error
        return redis.Redis.from_url(redis_url or "redis://localhost:6379/0", decode_responses=True)

    def validate_cameras(self, cameras: Sequence[Camera]) -> None:
        missing = sorted({camera.camera_id for camera in cameras} - self.locations.keys())
        if missing:
            raise ValueError(f"Missing coordinates for camera(s): {', '.join(missing)}")

    def prepare(self, row: Mapping[str, object]) -> dict[str, str]:
        """Enrich a detection into the canonical Redis and CSV sighting schema."""
        camera_id = row.get("camera_id")
        if not isinstance(camera_id, str) or camera_id not in self.locations:
            raise ValueError(f"No coordinates configured for camera: {camera_id}")
        return prepare_sighting(row, self.locations[camera_id], self.location_known)

    def publish(self, row: Mapping[str, object]) -> str:
        """Append a prepared sighting and return its Redis Stream entry ID."""
        fields = (
            {field: str(row[field]) for field in SIGHTING_FIELDS}
            if all(field in row for field in SIGHTING_FIELDS)
            else self.prepare(row)
        )
        entry_id = self.client.xadd(
            self.stream_key,
            fields,
            maxlen=self.max_stream_length,
            approximate=True,
        )
        return str(entry_id)


def create_sightings_publisher(
    cameras: Sequence[Camera],
    redis_url: str | None,
    locations_path: str | Path | None,
    locations_enabled: bool = False,
    client: RedisStreamClient | None = None,
) -> RedisSightingsPublisher | None:
    """Create a publisher; unknown locations use (0, 0) and are marked unknown."""
    if locations_enabled:
        if locations_path is None:
            raise ValueError("Set CAMERA_LOCATIONS_FILE when CAMERA_LOCATIONS_ENABLED is true")
        locations = load_camera_locations(locations_path)
    else:
        locations = {camera.camera_id: CameraLocation(0.0, 0.0) for camera in cameras}
    publisher = RedisSightingsPublisher(
        locations,
        redis_url=redis_url,
        client=client,
        location_known=locations_enabled,
    )
    publisher.validate_cameras(cameras)
    publisher.check_connection()
    return publisher
