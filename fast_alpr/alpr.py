"""
ALPR module.
"""

import os
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
import onnxruntime as ort
from fast_plate_ocr.inference.hub import OcrModel
from open_image_models.detection.core.hub import PlateDetectorModel

from fast_alpr.base import BaseDetector, BaseOCR, DetectionResult, OcrResult
from fast_alpr.default_detector import DefaultDetector
from fast_alpr.default_ocr import DefaultOCR

# pylint: disable=too-many-arguments, too-many-locals
# ruff: noqa: PLR0913


@dataclass(frozen=True)
class ALPRResult:
    """
    Detection and OCR output for one license plate.

    Attributes:
        detection: Detector output for the plate.
        ocr: OCR output for the plate, or None if OCR does not return a result.
    """

    detection: DetectionResult
    ocr: OcrResult | None


@dataclass(frozen=True, slots=True)
class DrawPredictionsResult:
    """
    Return value from draw_predictions.

    Attributes:
        image: The input image with boxes and text drawn on it.
        results: The ALPR results used to draw the annotations.
    """

    image: np.ndarray
    results: list[ALPRResult]


class ALPR:
    """
    Automatic License Plate Recognition (ALPR) system class.

    This class combines a detector and an OCR model to recognize license plates in images.
    """

    def __init__(
        self,
        detector: BaseDetector | None = None,
        ocr: BaseOCR | None = None,
        detector_model: PlateDetectorModel = "yolo-v9-t-384-license-plate-end2end",
        detector_conf_thresh: float = 0.4,
        detector_providers: Sequence[str | tuple[str, dict]] | None = None,
        detector_sess_options: ort.SessionOptions = None,
        ocr_model: OcrModel | None = "cct-xs-v2-global-model",
        ocr_device: Literal["cuda", "cpu", "auto"] = "auto",
        ocr_providers: Sequence[str | tuple[str, dict]] | None = None,
        ocr_sess_options: ort.SessionOptions | None = None,
        ocr_model_path: str | os.PathLike | None = None,
        ocr_config_path: str | os.PathLike | None = None,
        ocr_force_download: bool = False,
    ) -> None:
        """
        Initialize the ALPR system.

        Parameters:
            detector: An instance of BaseDetector. If None, the DefaultDetector is used.
            ocr: An instance of BaseOCR. If None, the DefaultOCR is used.
            detector_model: The name of the detector model or a PlateDetectorModel enum instance.
                Defaults to "yolo-v9-t-384-license-plate-end2end".
            detector_conf_thresh: Confidence threshold for the detector.
            detector_providers: Execution providers for the detector.
            detector_sess_options: Session options for the detector.
            ocr_model: The name of the OCR model from the model hub. This can be none and
                `ocr_model_path` and `ocr_config_path` parameters are expected to pass them to
                `fast-plate-ocr` library.
            ocr_device: The device to run the OCR model on ("cuda", "cpu", or "auto").
            ocr_providers: Execution providers for the OCR. If None, the default providers are used.
            ocr_sess_options: Session options for the OCR. If None, default session options are
                used.
            ocr_model_path: Custom model path for the OCR. If None, the model is downloaded from the
                hub or cache.
            ocr_config_path: Custom config path for the OCR. If None, the default configuration is
                used.
            ocr_force_download: Whether to force download the OCR model.
        """
        # Initialize the detector
        self.detector = detector or DefaultDetector(
            model_name=detector_model,
            conf_thresh=detector_conf_thresh,
            providers=detector_providers,
            sess_options=detector_sess_options,
        )

        # Initialize the OCR
        self.ocr = ocr or DefaultOCR(
            hub_ocr_model=ocr_model,
            device=ocr_device,
            providers=ocr_providers,
            sess_options=ocr_sess_options,
            model_path=ocr_model_path,
            config_path=ocr_config_path,
            force_download=ocr_force_download,
        )

    def predict(self, frame: np.ndarray | str) -> list[ALPRResult]:
        """
        Run plate detection and OCR on an image.

        Parameters:
            frame: Unprocessed frame (Colors in order: BGR) or image path.

        Returns:
            A list of ALPRResult objects, one for each detected plate.
        """
        if isinstance(frame, str):
            img_path = frame
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Failed to load image from path: {img_path}")
        else:
            img = frame

        plate_detections = self.detector.predict(img)
        alpr_results: list[ALPRResult] = []
        for detection in plate_detections:
            bbox = detection.bounding_box
            x1, y1 = max(bbox.x1, 0), max(bbox.y1, 0)
            x2, y2 = min(bbox.x2, img.shape[1]), min(bbox.y2, img.shape[0])
            cropped_plate = img[y1:y2, x1:x2]
            ocr_result = self.ocr.predict(cropped_plate)
            alpr_result = ALPRResult(detection=detection, ocr=ocr_result)
            alpr_results.append(alpr_result)
        return alpr_results

    def draw_predictions(self, frame: np.ndarray | str) -> DrawPredictionsResult:
        """
        Draw detections and OCR results on an image.

        Parameters:
            frame: The original frame or image path.

        Returns:
            A DrawPredictionsResult with the annotated image and the ALPR results.
        """
        # If frame is a string, assume it's an image path and load it
        if isinstance(frame, str):
            img_path = frame
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Failed to load image from path: {img_path}")
        else:
            img = frame

        # Get ALPR results using the ndarray
        alpr_results = self.predict(img)

        for result in alpr_results:
            detection = result.detection
            ocr_result = result.ocr
            bbox = detection.bounding_box
            x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
            # Draw the bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (36, 255, 12), 2)
            if ocr_result is None or not ocr_result.text or not ocr_result.confidence:
                continue
            confidence: float = (
                statistics.mean(ocr_result.confidence)
                if isinstance(ocr_result.confidence, list)
                else ocr_result.confidence
            )
            font_scale = min(1.25, max(0.4, img.shape[1] / 1000))
            text_thickness = 1 if font_scale < 0.75 else 2
            outline_thickness = text_thickness + max(3, round(font_scale * 3))
            display_lines = [f"{ocr_result.text} {confidence * 100:.0f}%"]

            _, text_height = cv2.getTextSize(
                display_lines[0], cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
            )[0]
            line_gap = max(14, round(text_height * 0.6))
            line_height = text_height + line_gap
            text_y = y1 - 10 - ((len(display_lines) - 1) * line_height)
            if text_y - text_height < 0:
                text_y = y2 + text_height + 10

            for idx, line in enumerate(display_lines):
                text_width, current_text_height = cv2.getTextSize(
                    line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
                )[0]
                text_x = min(max(x1, 5), max(5, img.shape[1] - text_width - 5))
                current_y = min(
                    max(text_y + (idx * line_height), current_text_height + 5),
                    img.shape[0] - 5,
                )
                # Draw black background for better readability
                cv2.putText(
                    img=img,
                    text=line,
                    org=(text_x, current_y),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=font_scale,
                    color=(0, 0, 0),
                    thickness=outline_thickness,
                    lineType=cv2.LINE_AA,
                )
                # Draw white text
                cv2.putText(
                    img=img,
                    text=line,
                    org=(text_x, current_y),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=font_scale,
                    color=(255, 255, 255),
                    thickness=text_thickness,
                    lineType=cv2.LINE_AA,
                )

        return DrawPredictionsResult(image=img, results=alpr_results)
