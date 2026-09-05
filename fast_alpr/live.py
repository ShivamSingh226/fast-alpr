"""Live multi-camera inference for CCTV streams."""

# The coordinator owns feed lifecycle, shared inference, output, and shutdown state.
# pylint: disable=duplicate-code,too-many-instance-attributes,too-many-locals

import argparse
import csv
import json
import logging
import os
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO
from urllib.parse import quote
from urllib.request import urlopen

import cv2

from fast_alpr import ALPR

LOGGER = logging.getLogger(__name__)
DEFAULT_CATALOGUE_URL = "https://cctv.corp8.cloud/cameras.json"
LIVE_CSV_FIELDS = (
    "camera_id",
    "frame",
    "pts_seconds",
    "plate",
    "ocr_confidence_percent",
    "detector_confidence_percent",
    "x1",
    "y1",
    "x2",
    "y2",
)


@dataclass(frozen=True)
class Camera:
    """Camera entry discovered from the remote catalogue."""

    camera_id: str


@dataclass(frozen=True)
class LiveStats:
    """Summary of a live processing run."""

    cameras_started: int
    frames_processed: int
    detections: int


class _RunStats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.cameras_started = 0
        self.frames_processed = 0
        self.detections = 0

    def increment(self, cameras: int = 0, frames: int = 0, detections: int = 0) -> None:
        with self._lock:
            self.cameras_started += cameras
            self.frames_processed += frames
            self.detections += detections

    def snapshot(self) -> LiveStats:
        with self._lock:
            return LiveStats(self.cameras_started, self.frames_processed, self.detections)


def load_camera_catalogue(url: str = DEFAULT_CATALOGUE_URL, timeout: float = 15.0) -> list[Camera]:
    """Load camera IDs from the catalogue without hard-coding the camera set."""
    with urlopen(url, timeout=timeout) as response:
        payload = json.load(response)

    entries = payload.get("cameras", payload) if isinstance(payload, dict) else payload
    cameras: list[Camera] = []
    for entry in entries:
        camera_id = entry if isinstance(entry, str) else entry.get("id")
        if isinstance(camera_id, str) and camera_id:
            cameras.append(Camera(camera_id=camera_id))
    if not cameras:
        raise ValueError(f"No cameras found in catalogue: {url}")
    return cameras


def build_stream_url(
    camera_id: str,
    protocol: str,
    email: str | None = None,
    password: str | None = None,
) -> str:
    """Build a credential-safe stream URL for one camera."""
    if protocol == "hls":
        return f"https://cctv.corp8.cloud/{quote(camera_id, safe='')}/index.m3u8"
    if protocol != "rtsp":
        raise ValueError(f"Unsupported protocol: {protocol}")
    if not email or not password:
        raise ValueError("CCTV_EMAIL and CCTV_PASSWORD are required for RTSP")
    user = quote(email, safe="")
    secret = quote(password, safe="")
    return f"rtsp://{user}:{secret}@103.250.160.189:8554/stream/{quote(camera_id, safe='')}"


class _MonotonicPts:
    """Turn source PTS values into a monotonic per-connection timeline."""

    def __init__(self) -> None:
        self.last_raw: float | None = None
        self.offset = 0.0
        self.last_pts: float | None = None

    def update(self, raw_pts_ms: float) -> float | None:
        if raw_pts_ms < 0:
            return None
        raw_pts = raw_pts_ms / 1000.0
        if self.last_raw is not None and raw_pts < self.last_raw:
            self.offset = (self.last_pts or 0.0) - raw_pts
        self.last_raw = raw_pts
        self.last_pts = raw_pts + self.offset
        return self.last_pts


class LiveFeedRunner:
    """Process multiple live feeds with reconnect and backoff handling."""

    def __init__(
        self,
        cameras: Sequence[Camera],
        protocol: str,
        output_csv: str | Path,
        alpr: ALPR | None = None,
        frame_stride: int = 1,
        reconnect_initial: float = 2.0,
        reconnect_max: float = 30.0,
    ) -> None:
        if not cameras:
            raise ValueError("At least one camera is required")
        if frame_stride < 1:
            raise ValueError("frame_stride must be at least 1")
        if reconnect_initial <= 0 or reconnect_max < reconnect_initial:
            raise ValueError("Reconnect backoff values are invalid")
        self.cameras = cameras
        self.protocol = protocol
        self.output_csv = Path(output_csv)
        self.alpr = alpr or ALPR()
        self.frame_stride = frame_stride
        self.reconnect_initial = reconnect_initial
        self.reconnect_max = reconnect_max
        self.stop_event = threading.Event()
        self.inference_lock = threading.Lock()
        self.csv_lock = threading.Lock()
        self.stats = _RunStats()

    def stop(self) -> None:
        """Request that all feed workers stop after their current read."""
        self.stop_event.set()

    def run(self, duration: float | None = None) -> LiveStats:
        """Process all feeds until stopped or until duration seconds elapse."""
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        email = os.getenv("CCTV_EMAIL")
        password = os.getenv("CCTV_PASSWORD")
        if self.protocol == "rtsp":
            build_stream_url(self.cameras[0].camera_id, self.protocol, email, password)
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

        with self.output_csv.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=LIVE_CSV_FIELDS)
            writer.writeheader()
            threads = [
                threading.Thread(
                    target=self._process_camera,
                    args=(camera, email, password, writer, csv_file),
                    name=f"fast-alpr-{camera.camera_id}",
                    daemon=True,
                )
                for camera in self.cameras
            ]
            for thread in threads:
                thread.start()
            if duration is None:
                try:
                    while any(thread.is_alive() for thread in threads):
                        time.sleep(0.5)
                except KeyboardInterrupt:
                    LOGGER.info("Stopping live feed processing")
                    self.stop()
            else:
                self.stop_event.wait(duration)
                self.stop()
            for thread in threads:
                thread.join()
            csv_file.flush()
        return self.stats.snapshot()

    def _process_camera(
        self,
        camera: Camera,
        email: str | None,
        password: str | None,
        writer: csv.DictWriter,
        csv_file: TextIO,
    ) -> None:
        stream_url = build_stream_url(camera.camera_id, self.protocol, email, password)
        backoff = self.reconnect_initial
        frame_number = 0
        self.stats.increment(cameras=1)
        pts = _MonotonicPts()
        while not self.stop_event.is_set():
            capture = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
            if not capture.isOpened():
                capture.release()
                LOGGER.warning("Could not open %s; retrying in %.1fs", camera.camera_id, backoff)
                self.stop_event.wait(backoff)
                backoff = min(backoff * 2, self.reconnect_max)
                continue

            backoff = self.reconnect_initial
            LOGGER.info("Connected to %s", camera.camera_id)
            try:
                while not self.stop_event.is_set():
                    ok, frame = capture.read()
                    if not ok:
                        LOGGER.warning("Read failed for %s; reconnecting", camera.camera_id)
                        break
                    current_pts = pts.update(capture.get(cv2.CAP_PROP_POS_MSEC))
                    if frame_number % self.frame_stride == 0:
                        with self.inference_lock:
                            results = self.alpr.predict(frame)
                        rows = self._rows(camera.camera_id, frame_number, current_pts, results)
                        if rows:
                            with self.csv_lock:
                                writer.writerows(rows)
                                csv_file.flush()  # type: ignore[attr-defined]
                        self.stats.increment(frames=1, detections=len(rows))
                    frame_number += 1
            finally:
                capture.release()
            if not self.stop_event.is_set():
                self.stop_event.wait(backoff)
                backoff = min(backoff * 2, self.reconnect_max)

    @staticmethod
    def _rows(
        camera_id: str, frame: int, pts: float | None, results: list
    ) -> list[dict[str, object]]:
        rows = []
        for result in results:
            if result.ocr is None or not result.ocr.text:
                continue
            confidence = result.ocr.confidence
            ocr_confidence = (
                sum(confidence) / len(confidence) if isinstance(confidence, list) else confidence
            )
            bbox = result.detection.bounding_box
            rows.append(
                {
                    "camera_id": camera_id,
                    "frame": frame,
                    "pts_seconds": "" if pts is None else round(pts, 3),
                    "plate": result.ocr.text,
                    "ocr_confidence_percent": round(ocr_confidence * 100, 2),
                    "detector_confidence_percent": round(result.detection.confidence * 100, 2),
                    "x1": bbox.x1,
                    "y1": bbox.y1,
                    "x2": bbox.x2,
                    "y2": bbox.y2,
                }
            )
        return rows


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run FastALPR on live CCTV feeds.")
    parser.add_argument("--camera-id", action="append", help="Camera ID; repeat or omit for all")
    parser.add_argument("--protocol", choices=("rtsp", "hls"), default="rtsp")
    parser.add_argument("--catalogue-url", default=DEFAULT_CATALOGUE_URL)
    parser.add_argument("-o", "--output-csv", default="live_alpr_detections.csv")
    parser.add_argument("--duration", type=float, help="Stop after this many seconds")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--reconnect-initial", type=float, default=2.0)
    parser.add_argument("--reconnect-max", type=float, default=30.0)
    parser.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the live CCTV CLI."""
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    catalogue = load_camera_catalogue(args.catalogue_url)
    selected = set(args.camera_id) if args.camera_id else {camera.camera_id for camera in catalogue}
    cameras = [camera for camera in catalogue if camera.camera_id in selected]
    missing = selected - {camera.camera_id for camera in cameras}
    if missing:
        raise SystemExit(f"Camera IDs not found in catalogue: {', '.join(sorted(missing))}")
    if not cameras:
        raise SystemExit("No cameras selected")
    stats = LiveFeedRunner(
        cameras=cameras,
        protocol=args.protocol,
        output_csv=args.output_csv,
        frame_stride=args.frame_stride,
        reconnect_initial=args.reconnect_initial,
        reconnect_max=args.reconnect_max,
    ).run(duration=args.duration)
    print(
        f"Started {stats.cameras_started} camera(s), processed {stats.frames_processed} frames, "
        f"and recorded {stats.detections} detections."
    )
    print(f"Detection CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
