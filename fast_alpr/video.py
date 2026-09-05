"""Video inference utilities and command-line entry point."""

# The frame-processing loop intentionally keeps resource handling in one place.
# ruff: noqa: PLR0915
# pylint: disable=too-many-locals

import argparse
import csv
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2

from fast_alpr.alpr import ALPR, ALPRResult


@dataclass(frozen=True)
class VideoStats:
    """Summary of a video inference run."""

    frames_processed: int
    frames_written: int
    detections: int
    recognized_plates: dict[str, int]


CSV_FIELDS = (
    "frame",
    "timestamp_seconds",
    "plate",
    "ocr_confidence_percent",
    "detector_confidence_percent",
    "x1",
    "y1",
    "x2",
    "y2",
)


def _result_row(frame_number: int, fps: float, result: ALPRResult) -> dict[str, object] | None:
    """Convert a recognized ALPR result into a CSV row."""
    if result.ocr is None or not result.ocr.text:
        return None

    confidence = result.ocr.confidence
    ocr_confidence = (
        sum(confidence) / len(confidence) if isinstance(confidence, list) else confidence
    )
    bbox = result.detection.bounding_box
    return {
        "frame": frame_number,
        "timestamp_seconds": round(frame_number / fps, 3),
        "plate": result.ocr.text,
        "ocr_confidence_percent": round(ocr_confidence * 100, 2),
        "detector_confidence_percent": round(result.detection.confidence * 100, 2),
        "x1": bbox.x1,
        "y1": bbox.y1,
        "x2": bbox.x2,
        "y2": bbox.y2,
    }


def process_video(
    input_source: str,
    output_video: str | Path,
    output_csv: str | Path,
    alpr: ALPR | None = None,
    frame_stride: int = 1,
    max_frames: int | None = None,
) -> VideoStats:
    """Process a video file, webcam index, or RTSP source.

    One CSV row is written for every recognized plate on every processed frame.
    The annotated video contains the OCR text and confidence percentage.
    """
    if frame_stride < 1:
        raise ValueError("frame_stride must be at least 1")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be at least 1")

    source: str | int = int(input_source) if input_source.isdigit() else input_source
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise ValueError(f"Could not open video source: {input_source}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width < 1 or height < 1:
        capture.release()
        raise ValueError(f"Could not read video dimensions from: {input_source}")

    output_path = Path(output_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore[attr-defined]
        fps / frame_stride,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise ValueError(f"Could not open output video: {output_path}")

    recognitions: Counter[str] = Counter()
    detector = alpr or ALPR()
    frames_processed = 0
    frames_written = 0
    frame_number = 0

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        csv_writer.writeheader()
        try:
            while max_frames is None or frames_processed < max_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_number % frame_stride == 0:
                    drawn = detector.draw_predictions(frame)
                    writer.write(drawn.image)
                    frames_processed += 1
                    frames_written += 1
                    for result in drawn.results:
                        row = _result_row(frame_number, fps, result)
                        if row is not None:
                            csv_writer.writerow(row)
                            recognitions[str(row["plate"])] += 1
                frame_number += 1
        finally:
            capture.release()
            writer.release()

    return VideoStats(
        frames_processed=frames_processed,
        frames_written=frames_written,
        detections=sum(recognitions.values()),
        recognized_plates=dict(recognitions),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run FastALPR on a video or camera stream.")
    parser.add_argument("input", help="Video path, webcam index such as 0, or RTSP URL")
    parser.add_argument("-o", "--output-video", default="alpr_output.mp4")
    parser.add_argument("-c", "--output-csv", default="alpr_detections.csv")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the video CLI."""
    args = _build_parser().parse_args(argv)
    stats = process_video(
        input_source=args.input,
        output_video=args.output_video,
        output_csv=args.output_csv,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
    )
    print(f"Processed {stats.frames_processed} frames and {stats.detections} detections.")
    print(f"Annotated video: {args.output_video}")
    print(f"Detection CSV: {args.output_csv}")
    if stats.recognized_plates:
        print("Recognized plates:")
        for plate, count in sorted(stats.recognized_plates.items()):
            print(f"  {plate}: {count} frame(s)")


if __name__ == "__main__":
    main()
