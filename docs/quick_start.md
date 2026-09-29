## 🚀 Quick Start

Here's how to get started with FastALPR:

### Predictions

```python
from fast_alpr import ALPR

# You can also initialize the ALPR with custom plate detection and OCR models.
alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="cct-xs-v2-global-model",
)

# The "assets/test_image.png" can be found in repo root dir
# You can also pass a NumPy array containing cropped plate image
alpr_results = alpr.predict("assets/test_image.png")
print(alpr_results)
```

???+ note

    See [reference](reference.md) for the available models.

Output:

<img alt="ALPR Result" height="350" src="https://raw.githubusercontent.com/ankandrew/fast-alpr/5063bd92fdd30f46b330d051468be267d4442c9b/assets/alpr_result.webp" width="700"/>

### Draw Results

You can also **draw** the predictions directly on the image:

```python
import cv2

from fast_alpr import ALPR

# Initialize the ALPR
alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="cct-xs-v2-global-model",
)

# Load the image
image_path = "assets/test_image.png"
frame = cv2.imread(image_path)

# Draw predictions on the image and get the ALPR results
drawn = alpr.draw_predictions(frame)
annotated_frame = drawn.image
results = drawn.results
```

Annotated frame:

<img alt="ALPR Draw Predictions" src="https://github.com/ankandrew/fast-alpr/releases/download/assets/alpr_draw_predictions.webp"/>

### Draw Results on Video

`draw_predictions()` can also be called for every video frame. This writes an annotated video and
prints each recognized plate with its OCR confidence percentage:

```python
import cv2

from fast_alpr import ALPR

alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="cct-xs-v2-global-model",
)

input_path = "assets/test_video.mp4"
output_path = "annotated_video.mp4"
capture = cv2.VideoCapture(input_path)

if not capture.isOpened():
    raise RuntimeError(f"Could not open {input_path}")

fps = capture.get(cv2.CAP_PROP_FPS) or 25
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)

frame_number = 0
while True:
    ok, frame = capture.read()
    if not ok:
        break

    drawn = alpr.draw_predictions(frame)
    writer.write(drawn.image)

    for result in drawn.results:
        if result.ocr is not None:
            confidence = result.ocr.confidence
            if isinstance(confidence, list):
                confidence = sum(confidence) / len(confidence)
            timestamp = frame_number / fps
            print(f"{timestamp:.2f}s {result.ocr.text} {confidence * 100:.2f}%")
    frame_number += 1

capture.release()
writer.release()
```

For a ready-to-use command that also writes a CSV containing plate text, timestamps, confidence
percentages, and bounding boxes, run:

```shell
uv run fast-alpr-video assets/test_video.mp4 \
    --output-video annotated_video.mp4 \
    --output-csv alpr_detections.csv
```

### Process Live CCTV Feeds

The live runner reads the camera IDs from the CCTV catalogue instead of hard-coding the camera
set. RTSP is the default and is forced over TCP. Keep credentials in environment variables; the
runner percent-encodes the email and password before constructing each RTSP URL.

```shell
export CCTV_EMAIL='you@example.com'
export CCTV_PASSWORD='your-access-password'
```

Start with one camera for a 60-second smoke test:

```shell
uv run fast-alpr-live \
    --camera-id cam04 \
    --duration 60 \
    --output-csv live-cam04-detections.csv
```

Process every camera currently returned by `cameras.json`:

```shell
uv run fast-alpr-live \
    --duration 300 \
    --output-csv live-detections.csv
```

If RTSP is blocked by the network, use the HLS endpoints:

```shell
uv run fast-alpr-live \
    --protocol hls \
    --camera-id cam04 \
    --duration 60 \
    --output-csv live-cam04-detections.csv
```

Each CSV row has the same fields as its Redis sighting event: event ID, camera ID, plate and
normalized plate, latitude/longitude, location-known flag, UTC capture time, source PTS, frame
number, and OCR/detector confidence. The runner reconnects failed feeds with exponential backoff
capped at 30 seconds and does not use arrival time or reported FPS for stream timing.
