# Multi-Camera Vehicle Tracking (ANPR) - Interview Prep

Project: real-time vehicle tracing across ~30 live CCTV feeds for the Gujarat Police Sentinel hackathon.
Stack: OpenCV + fast-alpr (ONNX Runtime, CoreML) -> Redis -> FastAPI -> browser map (Leaflet).

---

## 0. Read this first: honesty rules

Interviewers dig at the edge of what you actually did. Prepare for these three probes.

**1. Built vs designed.** Fill this table from your real repo before any interview. Only say "I built" for the left column.

| Built and run (verify in your repo) | Designed / would add (say "I would...") |
|---|---|
| Thread-per-camera RTSP ingestion (`live.py`) | Capture/inference decoupling with a latest-frame reader |
| Shared model behind a lock, frame stride | Batched inference, multiple workers |
| TCP-forced RTSP, exponential backoff | Backoff jitter, per-camera health scoring |
| Monotonic PTS handling | Multi-object tracker (ByteTrack) |
| CSV output, watchlist, exact normalized match | Fuzzy plate matching, multi-frame voting |
| Redis (as event bus / short-term store), FastAPI, map | Postgres/PostGIS durable store, Kafka at scale |

**2. Your contribution vs the open-source.** The detector and OCR are `fast-alpr` (ankandrew, open source). Say so first. Your work is the live multi-camera runner, the integration, and the dashboard. Owning that line makes you credible.

**3. Numbers.** Never quote a number you did not measure. Section 7 tells you how to measure each one in under an hour. If your real result is "10 cameras stable, 30 not", say that and explain why. A diagnosed limit is a stronger answer than a claimed win.

---

## 1. The 60-second pitch

> "The problem was to trace a given vehicle across about 30 live CCTV feeds, show its route on a map, and raise an alert the moment a watchlisted plate appears. I started from an open-source ANPR library that processed one video file, and extended it into a live pipeline: one thread per camera pulling RTSP over TCP, a shared model for plate detection and OCR, and a Redis stream that decouples detection from a FastAPI service. FastAPI pushes events to a browser over server-sent events, where each detection lights up a camera on a Leaflet map and builds the vehicle's route. On a laptop with no GPU, the hard part was throughput, so I sampled frames, forced TCP, added backoff reconnects, and measured where the pipeline saturated."

Data flow in one line:
`CCTV RTSP -> reader thread -> sampled frame -> ALPR (detect + OCR) -> normalized plate event -> Redis Stream -> FastAPI consumer -> SSE -> browser map + watchlist alert`

---

## 1A. Beginner primer: how the pieces fit (read this first)

Read this section first, then use section 3 for the interview-grade versions of the same answers.

Your system is several programs that need to hand data to each other. Each tool below answers a different "how?" question.

```
cameras -> detector -> Redis Stream -+-> FastAPI -> (SSE) -> browser map   (live path)
                                     |
                                     +-> DB writer -> PostgreSQL           (history path)
```

A detection is born in the detector, goes down the **live path** so you see it within moments, and is copied down the **history path** so it is still there tomorrow.

### Redis: the hand-off point

The detector and the FastAPI app are two separate programs. Two programs cannot share Python variables, so the detector cannot append to a list that FastAPI reads. They need a middleman both can reach.

Think of a restaurant. The kitchen (detector) puts finished plates on a pass, and the waiters (FastAPI) pick them up. The kitchen never walks to the tables, and the waiters never walk into the kitchen. **Redis is the pass.**

- Redis keeps data in RAM, so it is very fast.
- A **Stream** is an append-only log. Every detection gets an ID, and each reader remembers the last ID it saw. If the dashboard disconnects for a few seconds, it resumes from where it left off. Plain Pub/Sub would have lost those events.
- A CSV file could act as a middleman too, but every reader would keep re-reading it, and two writers could corrupt it.

### PostgreSQL: why you need it if you already have Redis

Redis is a whiteboard, and PostgreSQL is a filing cabinet.

| | Redis | PostgreSQL |
|---|---|---|
| Lives in | RAM (small) | Disk (large) |
| History | Trimmed to a maximum length, so old events fall off | Keeps everything |
| After a restart | Data may be lost unless persistence is configured | Survives crashes and restarts |
| Good at | "Give me the latest events" | "Show every sighting of this plate today, in order, with camera locations, within 2 km of the last one" |

For police evidence you want a permanent, auditable record, which points to a database.

**For the hackathon demo you do not need Postgres.** Redis alone (or even memory) is enough to show live tracking. In the interview, say Postgres is the durable-storage design. Only claim it if you built it.

**Where it sits:** behind Redis. A small "DB writer" reads the stream and inserts rows in batches. The alert path never waits on the database, so a slow disk cannot delay an alert.

### FastAPI: the front door

A browser only speaks HTTP: "GET this URL." It cannot talk to Redis or to your detector. A web framework turns URLs into Python functions. When the browser requests `/api/route?plate=GJ01AB1234`, FastAPI runs your function and returns the answer.

Why FastAPI:
- It handles many open connections well (see SSE below).
- It validates incoming data automatically, so a bad watchlist request gets a clear error.
- It generates a `/docs` page listing every endpoint, which is a nice thing to show judges.

**FastAPI does not touch video.** The detector does that in a separate process. If you decoded video inside a FastAPI route, one slow frame would freeze the whole API, including every open dashboard.

### SSE: how the server pushes live updates

The problem: a new detection appears on the server, and the browser must know now. There are three ways:

- **Polling:** the browser asks "anything new?" every second. It wastes requests and adds up to a second of delay.
- **WebSocket:** a two-way channel. It works, but it is more machinery than this needs.
- **SSE (Server-Sent Events):** the browser opens one ordinary HTTP connection, and the server keeps writing to it whenever something happens. It is like a radio broadcast: tune in once, and updates arrive as they occur.

The browser side is one line:

```javascript
const es = new EventSource('/events');
es.onmessage = e => showOnMap(JSON.parse(e.data));
```

SSE also **reconnects automatically** if the connection drops, and it can tell the server the last event ID it saw so nothing is missed. Your data flows one way (server to browser), so SSE is the right amount of machinery.

Trade-off: each open browser tab holds a connection open. That is fine for a demo, but at large scale you would put a gateway in front.

### Threads vs processes vs asyncio in the detector

You have 30 streams to read and a model to run. There are three ways to do many things at once:

- **Threads** are several workers in one kitchen sharing one pantry (memory). They are cheap and can share the single loaded model.
- **Processes** are separate kitchens, each with its own pantry. Each would load its own copy of the model, and frames would have to be copied between them.
- **asyncio** is one cook juggling many dishes. It switches dishes only at points where the code says `await` ("I am waiting, switch to something else"). It works well when your waiting is done through async-aware libraries.

Python has a rule called the **GIL**: only one thread can run Python code at a time. That sounds like it kills threads. But when a thread is inside a C library, such as OpenCV reading a frame or ONNX Runtime running the model, it releases the GIL and another thread can run. Your slow work happens almost entirely inside those C libraries, so threads do run in parallel.

That is why:
- **Not asyncio:** `cap.read()` and the model call are blocking C functions and cannot be `await`ed. Calling them in async code would freeze the whole event loop. You would end up running them in a thread pool anyway.
- **Not processes:** you would pay for extra model copies and for copying frames between processes, with no gain, because the heavy work already runs in parallel.

**Two caveats to admit:**
1. In `live.py` the model is shared behind a lock, so *reading* 30 streams happens in parallel but *inference* happens one at a time. That is exactly why you hit a throughput ceiling.
2. If Python-side code around the model grows heavy, the GIL comes back as a bottleneck, and processes or a worker pool become the right answer.

Rule of thumb: if the program mostly waits on the network or on C libraries, use threads or asyncio. If it does heavy pure-Python math, use processes.

### One-sentence answers

- **Redis:** "It is the fast hand-off between the detector and the API, and a Stream lets a reconnecting dashboard resume without losing events."
- **Postgres:** "Redis is memory and gets trimmed, so Postgres is the durable, queryable record for history and evidence. It sits behind Redis, off the alert path."
- **FastAPI:** "It is the HTTP front door, and it never touches video, so a decoder problem cannot take the API down."
- **SSE:** "It is a one-way live push over plain HTTP with automatic reconnect, which is all the dashboard needs."
- **Threads:** "The heavy calls are in C libraries that release the GIL, so threads run in parallel and share one model. asyncio cannot await those calls, and processes would duplicate the model."

### Glossary

| Term | Plain meaning |
|---|---|
| **GIL** | Python's rule that only one thread runs Python code at a time. C libraries release it while they work. |
| **Stream (Redis)** | An append-only log with IDs. Readers remember their position and can resume. |
| **Consumer group** | Several workers sharing one stream, each event handled by one of them, with acknowledgements. |
| **Pub/Sub** | Fire-and-forget broadcast. If you are not listening at that moment, you miss it. |
| **SSE** | Server keeps one HTTP connection open and pushes events to the browser. |
| **Blocking call** | A function that makes the caller wait until it finishes. Reading a frame is one. |
| **Event loop** | The single "juggler" inside asyncio that switches between waiting tasks. |
| **Back-pressure / overload** | What happens when work arrives faster than it can be processed. Queues grow. |
| **Idempotent** | Doing it twice has the same effect as doing it once, so a re-delivered event is harmless. |
| **Stride** | Process every Nth frame and skip the rest. |
| **PTS** | Presentation timestamp: the video's own clock for each frame. |

---

## 1B. The ML models in the pipeline

Your system uses **two trained models, run one after the other**. Both come from the open-source `fast-alpr` library and are stored in ONNX format.

```
frame -> [Detector: YOLOv9]  -> plate box -> crop -> [OCR: CCT] -> "GJ01AB1234" + confidence per character
          "where is the plate?"                       "what does it say?"
```

| Stage | Model family | Comes from | Input | Output | What you store in the CSV |
|---|---|---|---|---|---|
| **Detector** | YOLOv9 (a small "t" or "s" variant) | `open-image-models` | The full video frame | Box around each plate + a confidence | `detector_confidence_percent`, `x1,y1,x2,y2` |
| **OCR** | Compact Convolutional Transformer (CCT) | `fast-plate-ocr` | The cropped plate image | Plate text + a confidence per character | `plate`, `ocr_confidence_percent` |

**Fill in your exact names.** The default varies between library versions (for example detector `yolo-v9-t-384-license-plate-end2end` or `yolo-v9-s-608-license-plate-end2end`, and OCR `cct-xs-v2-global-model` or `cct-s-v2-global-model`). Open `fast_alpr/default_detector.py` and `fast_alpr/default_ocr.py` and copy the strings here:

- Detector: `[paste from default_detector.py]`
- OCR: `[paste from default_ocr.py]`
- Library version: `[uv pip show fast-alpr]`

### Plain-language concepts

- **Detection vs recognition.** Detection finds *where* something is. Recognition reads *what* it is. Doing them as two stages keeps each model small and lets you upgrade one without touching the other (the library lets you swap in your own OCR).
- **Confidence.** A number from 0 to 1 (you show it as a percent) for how sure the model is. It is not the same as accuracy: a model can be confidently wrong.
- **ONNX** is a portable file format for trained models. **ONNX Runtime** is the engine that runs them. Its **execution providers** pick the hardware: CPU, CoreML (Apple GPU/Neural Engine), CUDA (NVIDIA), OpenVINO, DirectML, QNN. Because the model is in ONNX, moving from your M4 to a GPU server is a provider change, not a rewrite.
- **Model size variants** (the "t", "s", "xs" and the numbers like 384 or 608 in the names). Smaller means faster but less accurate. The number is the image size the detector works at: 384 is fast, 608 sees smaller plates.

### Model questions and answers

**Is the detector YOLOv8?**
No. The library ships YOLOv9 variants. Say: "The detector is a YOLOv9 variant from open-image-models. It is chosen by a config string, so swapping detectors is a one-line change, and I picked the default because it is trained for plates and runs without a GPU."

**Why two stages instead of one model that reads the whole frame?**
A plate is tiny in a wide CCTV frame. The detector's job is to find it at full-frame scale. The OCR then runs on a tight crop at a small fixed size, which is much cheaper and more accurate than reading text from a huge frame. It also lets you upgrade each stage independently.

**Why a YOLO-style detector?**
YOLO predicts all boxes in a single forward pass, which makes it fast enough for real-time video on modest hardware. That is the reason it is the standard choice for this kind of task.

**What is the trade-off in the detector's input size?**
Smaller input (384) is faster, but on a wide scene a distant plate may shrink to a few dozen pixels or vanish. A larger input (608) or splitting the frame into tiles finds smaller plates at higher cost. This is a real lever for CCTV, so mention it as something you would test on your own footage.

**How does the OCR model read a plate?**
It takes the cropped plate at a fixed size and predicts the characters position by position, giving a confidence for each character. That per-character confidence is why your code has `sum(confidence)/len(confidence)`.

**What is a weakness in how confidence is used today?**
Your `_rows()` averages the per-character confidences. One badly read character among nine confident ones barely moves the average. For deciding whether a watchlist match is trustworthy, use the **minimum** character confidence instead. This is a concrete, small improvement to mention.

**How accurate is it?**
Do not quote numbers from the library's documentation as your own results. Say what you measured: label the plates in a short clip of your footage and report exact-match accuracy (whole plate correct) and character error rate. If you have not done this yet, say it is the next step. Section 7 explains how.

**Will it work on Indian plates?**
The default OCR is a multi-country ("global") model, so it is a reasonable starting point, but you must verify it on your footage. Indian plates vary in font, two-line layouts, non-standard plates, dirt and night/IR video. The honest answer is: "It works as a baseline, and the way I would improve it is to collect and label local plates and fine-tune, or validate against the plate format."

**Why not train your own model?**
In a 2-day hackathon there was no labeled data or time. A pre-trained ONNX model gets a working system. The design keeps the model swappable, so fine-tuning on local data is an isolated upgrade.

**Why not a cloud OCR API?**
Cost and latency per frame, a network dependency on an operational feed, and sending surveillance images to a third party. Running locally keeps data in your control.

**What are the limits of plate-only tracking?**
It cannot tell two vehicles apart if a plate is cloned or misread, and it cannot follow a vehicle when no plate is visible. In a policing context a false match can lead to a wrongful stop, so alerts should be treated as *candidates for human verification*, with the snapshot shown and an audit log kept.

---

## 1C. Two meanings of "Model" (do not mix them up)

| Term | Meaning | Where it appears |
|---|---|---|
| **ML model** | A trained neural network (the YOLOv9 detector, the CCT OCR) | Section 1B, `fast_alpr/default_detector.py`, `default_ocr.py` |
| **Hackathon "Model 1/2/3/4"** | The four integration approaches in the problem statement | The Sentinel problem statement |

The four hackathon models are: **Model 1** central camera registry with GIS map (metadata only, no video), **Model 2** unified viewing platform with ANPR, event tagging and alerts, **Model 3** VMS federation middleware, **Model 4** a full central VMS with recording and storage.

**How to answer "which model did you follow?"**
"A hybrid, mainly Model 2: I consume the live feeds directly, run ANPR, keep searchable movement records and raise watchlist alerts, without storing all the video. I added a thin Model 1 layer, a camera registry with coordinates, to plot routes on a map. I did not build Model 3 (no federation middleware) or Model 4 (no central recording or storage)."

**A likely follow-up:** "Model 2 says no middleware, but you use Redis." Answer: "The 'no middleware' line means no federation layer connecting other departments' VMS systems (that is Model 3). Redis is my own pipeline's internal message bus between the detector and the API."

Verify that the registry and map (Model 1's part) exist in your build before you claim them.

---

## 1D. The ingestion path: how FastAPI moves data from Redis into PostgreSQL

This section answers a question people often get wrong at first: **no browser and no client ever calls an endpoint to "send" a detection.** A detection is born inside the detector process and pushed onto the Redis Stream directly. FastAPI does not receive it as an HTTP request. Instead, FastAPI runs a **background consumer** that pulls from the stream continuously and writes to Postgres. The only REST endpoints involved are the ones for *reading* what has already been written, and for the watchlist, which is a much lower-frequency, human-driven write.

```
detector process --XADD--> Redis Stream "alpr:sightings"
                                  |
                                  |  (no HTTP here - this is a stream read, not a request)
                                  v
                    FastAPI's background consumer task
                    (started at app startup, runs forever)
                                  |
                    preprocess -> validate -> batch -> INSERT
                                  |
                                  v
                             PostgreSQL
                                  ^
                                  |  (normal HTTP GET, this part IS the REST API)
                                  |
                          GET /track/{plate}   <- browser / judge
```

### Why this has to be a background task, not a request handler

A REST endpoint only runs when someone calls it. A detection can happen at any moment, with nobody watching, so nothing is calling an endpoint to trigger the write. The consumer has to be something that starts once, when the app boots, and then loops forever on its own. In FastAPI that is done in the **lifespan** function, the same place you already start the ingestion of live camera threads if this is the single-process version of the demo.

### The five preprocessing steps

Redis hands you raw bytes. Postgres wants typed, validated, de-duplicated rows. This is the "preprocessing" the consumer does on every batch it reads:

| Step | What happens | Why |
|---|---|---|
| **1. Decode** | Redis Stream fields arrive as `bytes`; decode to `str` and parse numbers | Redis does not know your schema, it just stores bytes |
| **2. Validate** | Parse into a Pydantic model (`DetectionEvent`); reject anything that does not fit | A malformed event from a buggy detector should not crash the writer or corrupt the table |
| **3. Normalize** | Re-derive `plate_norm` from `plate` locally, rather than trusting the producer's `normalized_plate` blindly; log a mismatch if they disagree | Defense in depth — the producer already normalizes, but the consumer should not assume upstream code never changes |
| **4. De-duplicate** | The Redis event already carries an `event_id`; use it as the Postgres primary key with `ON CONFLICT (event_id) DO NOTHING` | Redis Streams give *at-least-once* delivery, so the same event can arrive twice after a crash/restart; a single-column key makes the retry trivially safe |
| **5. Batch** | Collect up to `N` events or wait up to `T` milliseconds, then issue **one** multi-row `INSERT` | One round trip for 50 rows is far cheaper than 50 round trips; this is the single biggest performance lever on the write side |

### The real Redis schema

The Redis Stream actually carries these fields per event (confirmed from the running system, not guessed):

```
event_id, camera_id, plate, normalized_plate, latitude, longitude, location_known,
captured_at_utc, pts_seconds, frame, ocr_confidence_percent, detector_confidence_percent
```

Two things this schema gives you for free, compared to an earlier draft of this pipeline:

- **`event_id` makes de-duplication a one-column primary key** instead of a composite `(camera_id, captured_at, plate_norm)` key.
- **`location_known` tells the consumer exactly when `latitude`/`longitude` are placeholder `0.0, 0.0` values** rather than real coordinates. The consumer stores `NULL` for both whenever `location_known` is false, so a map or route query never has to remember to filter out "null island" itself — a `NULL` simply cannot be plotted.

```sql
CREATE TABLE detections (
    event_id                    text PRIMARY KEY,
    camera_id                   text NOT NULL,
    plate_raw                   text NOT NULL,
    plate_norm                  text NOT NULL,
    latitude                    double precision,        -- NULL when location_known = false
    longitude                   double precision,        -- NULL when location_known = false
    location_known               boolean NOT NULL,
    captured_at                  timestamptz NOT NULL,
    pts_seconds                  double precision,
    frame                         integer,
    ocr_confidence_percent        real,
    detector_confidence_percent   real,
    ingested_at                   timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX ON detections (plate_norm, captured_at);
CREATE INDEX ON detections (camera_id, captured_at);
CREATE INDEX ON detections (captured_at) WHERE location_known;
```

### The consumer, as code

```python
# ingestion_consumer.py
import asyncio
from datetime import datetime, timezone

import asyncpg
import redis.asyncio as aioredis
from pydantic import BaseModel, ValidationError, field_validator

STREAM = "alpr:sightings"   # confirmed via: docker exec -it redis redis-cli KEYS '*'
GROUP = "pg-writer"
CONSUMER_NAME = "writer-1"
BATCH_SIZE = 50
BATCH_WINDOW_MS = 200


class DetectionEvent(BaseModel):
    event_id: str
    camera_id: str
    plate: str
    normalized_plate: str
    latitude: float
    longitude: float
    location_known: bool
    captured_at_utc: datetime
    pts_seconds: float | None = None
    frame: int | None = None
    ocr_confidence_percent: float
    detector_confidence_percent: float

    @field_validator("captured_at_utc", mode="before")
    @classmethod
    def parse_captured_at(cls, v):
        if isinstance(v, bytes):
            v = v.decode()
        if isinstance(v, (int, float)):
            return datetime.fromtimestamp(v, tz=timezone.utc)
        if isinstance(v, str):
            stripped = v.replace(".", "", 1)
            if stripped.replace("-", "", 1).isdigit() and "T" not in v:
                return datetime.fromtimestamp(float(v), tz=timezone.utc)   # epoch seconds as string
            return datetime.fromisoformat(v.replace("Z", "+00:00"))         # ISO 8601 string
        return v

    @field_validator("location_known", mode="before")
    @classmethod
    def parse_bool(cls, v):
        if isinstance(v, bytes):
            v = v.decode()
        if isinstance(v, str):
            return v.strip().lower() in ("1", "true", "yes")
        return bool(v)


def normalize_plate(raw: str) -> str:
    return "".join(ch for ch in raw.upper() if ch.isalnum())


async def consume_detections(redis: aioredis.Redis, pg_pool: asyncpg.Pool) -> None:
    try:
        await redis.xgroup_create(STREAM, GROUP, id="0", mkstream=True)
    except aioredis.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise  # group already exists on a restart - fine, not an error

    while True:
        entries = await redis.xreadgroup(
            GROUP, CONSUMER_NAME, {STREAM: ">"}, count=BATCH_SIZE, block=BATCH_WINDOW_MS
        )
        if not entries:
            continue

        rows, message_ids = [], []
        for _, messages in entries:
            for message_id, fields in messages:
                decoded = {k.decode(): v.decode() for k, v in fields.items()}
                try:
                    event = DetectionEvent(**decoded)
                except ValidationError:
                    message_ids.append(message_id)   # malformed - ack it so it doesn't block the stream forever
                    continue

                recomputed = normalize_plate(event.plate)   # defense in depth, don't trust upstream blindly
                lat = event.latitude if event.location_known else None
                lon = event.longitude if event.location_known else None

                rows.append((
                    event.event_id, event.camera_id, event.plate, recomputed,
                    lat, lon, event.location_known, event.captured_at_utc,
                    event.pts_seconds, event.frame,
                    event.ocr_confidence_percent, event.detector_confidence_percent,
                ))
                message_ids.append(message_id)

        if rows:
            async with pg_pool.acquire() as conn:
                await conn.executemany(
                    """
                    INSERT INTO detections
                        (event_id, camera_id, plate_raw, plate_norm,
                         latitude, longitude, location_known, captured_at,
                         pts_seconds, frame, ocr_confidence_percent, detector_confidence_percent)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
                    ON CONFLICT (event_id) DO NOTHING
                    """,
                    rows,
                )

        # Ack only after the insert succeeds -- if it throws, execution never reaches
        # here, the batch stays pending in the consumer group, and the next loop
        # (or another worker) retries it. ON CONFLICT (event_id) makes that retry
        # harmless even if some rows in the batch already landed.
        if message_ids:
            await redis.xack(STREAM, GROUP, *message_ids)
```

Wired into the FastAPI app's lifespan:

```python
# main.py
from contextlib import asynccontextmanager
import asyncio
import asyncpg
import redis.asyncio as aioredis
from fastapi import FastAPI
from ingestion_consumer import consume_detections

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.redis = aioredis.Redis(host="localhost", port=6379)
    app.state.pg_pool = await asyncpg.create_pool(dsn="postgresql://...")

    consumer_task = asyncio.create_task(consume_detections(app.state.redis, app.state.pg_pool))
    yield
    consumer_task.cancel()          # graceful shutdown
    await app.state.pg_pool.close()

app = FastAPI(lifespan=lifespan)
```

Because this uses `redis.asyncio` and `asyncpg` (both async libraries), the `await redis.xreadgroup(...)` call yields control back to the event loop while it waits, so the consumer task never blocks incoming HTTP requests. This is the same asyncio model as the rest of the README's SSE discussion, applied to a background task instead of a request.

### The REST endpoints around this pipeline

The consumer above has no HTTP surface of its own. These are the actual endpoints that expose what it produces:

| Method & path | Reads / writes | Purpose |
|---|---|---|
| `GET /track/{plate}` | Reads Postgres | The vehicle's route history — this is reading the *output* of the ingestion pipeline, never Redis directly, since Redis is trimmed and does not hold full history |
| `GET /api/ingestion/status` | Reads Redis (`XLEN`, `XPENDING`) | Exposes consumer health: stream length, how many events are unacknowledged, and how far behind the writer is — this is what makes the background task observable through the API instead of being invisible |
| `POST /watchlist` | Writes Postgres directly | A human action, not a detection event, so it goes straight to Postgres and never touches the Redis stream |
| `DELETE /watchlist/{plate}` | Writes Postgres directly | Same reasoning as above |
| `GET /api/detections/recent` | Reads Postgres | Optional debug endpoint: last N rows, useful to show a judge that data really is landing in the database, not just flying through Redis |

Example of the health endpoint, since "how do you know ingestion is keeping up" is a natural interview question:

```python
@app.get("/api/ingestion/status")
async def ingestion_status(request: Request):
    redis = request.app.state.redis
    length = await redis.xlen(STREAM)
    pending = await redis.xpending(STREAM, GROUP)
    return {
        "stream_length": length,
        "unacknowledged": pending["pending"] if pending else 0,
    }
```

A consistently large `unacknowledged` count is the same overload signal as in STAR 2, but on the write side instead of the inference side: the writer is falling behind the stream.

### One process or two? (the same choice as the detector, applied here)

| | In-process (background task inside FastAPI) | Separate process ("db-writer" service) |
|---|---|---|
| Simplicity | One thing to run — good for a hackathon | Two things to run and coordinate |
| Failure isolation | A crash in the consumer loop is caught by `asyncio`, but a bug that hangs it could, in principle, still compete for the event loop | A stuck writer cannot affect the API's ability to serve `GET /track/{plate}` at all |
| Scaling | Cannot add a second writer without adding a second FastAPI process | Add more writer processes independently of API replicas, using Redis consumer groups to split the stream between them |
| What to say if asked | "For the demo I ran it as a background task in the same FastAPI process, started in lifespan. At real scale I would run it as its own process, the same separation I used for the detector, so a slow database can never affect the API." | |

**Update section 0's built-vs-designed table** once you have actually implemented this: mark whether the consumer, the batching, the `ON CONFLICT` idempotency, and the `/api/ingestion/status` endpoint are things you built and ran, or things you are describing as the design.

### A real debugging note (worth keeping as a STAR example)

Before trusting the consumer code above against production data, I verified the real stream directly with `docker exec -it redis redis-cli XRANGE alpr:sightings - + COUNT 10`. Two findings from that one command:

1. **The stream was actually named `alpr:sightings`, not `detections`.** I'd been writing consumer code against an assumed name. A one-line fix (`STREAM = "alpr:sightings"`), but it would have silently produced zero rows in Postgres and looked like a working, tested pipeline right up until the demo.
2. **One sampled entry had `plate: "1"`** — clearly not a real plate — while a later entry read `GJ32NG041` correctly. Comparing several entries showed it was an isolated OCR misread, not a systemic bug, and it also surfaced that even a *correct-looking* read (`GJ32NG041`, 9 characters, standard Indian plates run 9-10) can still be worth validating against a format pattern before treating it as ground truth for a watchlist match.

This is a stronger interview answer than a hypothetical: **"I don't assume an integration works — I verify the raw data at the boundary before trusting code built against it. In this case, `XRANGE` on the actual Redis stream caught both a naming mismatch and a live example of OCR noise, in about two minutes, before either became a demo-day surprise."**

### Interview Q&A for this specific piece

**"Does the browser send detections to FastAPI?"**
No. Detections never arrive as an HTTP request. They are pushed by the detector directly onto the Redis Stream. FastAPI's only role on the write side is running a background task that reads that stream.

**"Why not write to Postgres directly from the detector, and skip Redis entirely?"**
Two reasons. First, a database write is slower and less predictable than an in-memory append, and I do not want the camera-reading thread ever blocked on a database, since that is exactly the kind of stall that caused the reconnect storm in STAR 2. Second, Redis is also what feeds the live SSE path to the dashboard; one write from the detector, two independent readers (the live path and the Postgres writer) consume it at their own pace.

**"What happens if Postgres is briefly unreachable?"**
The consumer's `INSERT` raises, the code never reaches the `XACK` line, so the events stay in the consumer group's pending list. Redis is not told they succeeded. When Postgres comes back, the next loop iteration reads them again (or `XAUTOCLAIM` reassigns them if the specific worker died). This is exactly the "ack only after commit" logic in the code above, and it is why at-least-once delivery does not lose data on a database outage — the trade-off is that the same event can, rarely, be written twice, which is exactly what the `ON CONFLICT DO NOTHING` unique constraint is there to make harmless.

**"Why batch instead of inserting one row at a time?"**
Each round trip to Postgres has fixed overhead (network + transaction). At even a modest detection rate, one-row-at-a-time inserts spend more time on round trips than on the actual writes. Batching up to 50 events or 200ms, whichever comes first, cuts that overhead by close to the batch size, with a bounded, small added delay before data is durable.

**"Is this the bottleneck of the whole system?"**
Almost certainly not, compared to the detector's inference step. A batched insert of 50 rows takes low single-digit milliseconds; running the ONNX model takes tens of milliseconds per frame, and that happens far more often. If asked to justify that claim, point back to the measured inference time in section 10 and note you would confirm the write-side timing with the same kind of stopwatch measurement before claiming it is not a bottleneck.

---

## 2. STAR stories

Replace every `[MEASURE]` with a number you measured. Delete any claim you cannot back.

### STAR 1 - Handling many live feeds concurrently

- **Situation:** The hackathon gave ~30 live RTSP feeds and needed a vehicle traced across them. My starting point processed a single file offline.
- **Task:** Ingest many feeds at once, detect and read plates, and alert in near real time on a laptop with no GPU.
- **Action:**
  - One thread per camera. The work is I/O-bound and both `cv2.VideoCapture.read()` and ONNX Runtime release the GIL, so threads give real parallelism without process/IPC complexity.
  - One shared model instance behind a lock, so memory does not scale with camera count and I avoid assuming concurrent `Run()` safety.
  - Frame stride so I infer on a fraction of frames, because plates persist over many frames.
  - RTSP forced to TCP (UDP produced corrupt frames that look like model bugs), and exponential backoff reconnects (2s to 30s) because feeds restart.
- **Result:** `[MEASURE]` cameras stable at stride `[S]`, `[X]` inferences/sec, `[Y]`% CPU. Limit found: `[what saturated first]`.

### STAR 2 - Diagnosing a throughput collapse (best story for "tell me about a hard bug")

- **Situation:** At 30 cameras only ~11 were live, RTSP connections were refused, and CoreML inference errors appeared in the log.
- **Task:** Find out whether the limit was my code, the model runtime, or the server.
- **Action:** I did the demand/capacity arithmetic. 30 feeds x 25 fps / stride 10 = 75 inferences/sec demanded. One `predict()` at ~40ms behind a lock gives ~25/sec capacity, so utilization was about 3x over 1. When utilization exceeds 1 a queue grows without bound (Little's Law). In my design a thread waiting on the lock stops reading its stream, the gateway sees a slow client and drops it, and the reconnect adds to the connection storm. I then ran a stepped test (5, 10, 15, 20 cameras) and compared where refusals began against CPU use, to separate a server-side connection cap from my own load. I added staggered startup and jittered backoff so 30 retries do not synchronize, and raised stride until demand was under capacity.
- **Result:** `[MEASURE]` refusals began at `[N]` cameras; after fixes `[N2]` cameras stable. I also identified the structural fix (decouple capture from inference) and documented it as the next step.

### STAR 3 - Latency

- **Situation:** An alert is only useful if it arrives while the vehicle is still near that camera.
- **Task:** Reduce and understand end-to-end alert latency.
- **Action:** I broke latency into stages instead of guessing: gateway GOP buffering, decode, wait for the next sampled frame (a floor of `stride / fps`, e.g. 0.4s at stride 10 and 25 fps), lock wait, inference, Redis publish, consumer read, SSE push, render. I timestamped events at capture and again in the browser and compared. Levers I used or would use: lower stride only where it pays, downscale before detection, hardware decode (VideoToolbox), dropping stale frames, and keeping the alert path free of any database write.
- **Result:** p50 `[X]`ms, p95 `[Y]`ms, before `[A]`ms after `[B]`ms.

### STAR 4 - Detection quality and robustness

- **Situation:** OCR errors and repeated sightings of one vehicle polluted the route and could miss a watchlist hit (one wrong character means no exact match).
- **Task:** Increase true detections and reduce noise.
- **Action:** Normalize plates (uppercase, strip non-alphanumerics) before any comparison, keep OCR and detector confidence per event, and collapse consecutive sightings at one camera into a single route step. Next steps I identified: validate against the Indian plate pattern, fuzzy-match with an OCR-confusion map (O/0, I/1, B/8, S/5), and vote across frames of the same vehicle.
- **Result:** `[MEASURE]` recall/precision on a labeled clip of `[N]` plates. Only claim this if you actually labeled a clip.

---

## 3. Design-choice questions

### Why FastAPI?
- The API layer is I/O-bound (SSE streams, watchlist CRUD, route queries), which suits async. It also gives typed request validation (Pydantic), automatic OpenAPI docs, and easy dependency injection.
- **Key point:** FastAPI does not touch video. Decoding and inference run in a separate process. A blocking `cap.read()` inside an async route would freeze the event loop, including every SSE client. Keeping them apart also means a decoder crash cannot take the API down.
- Alternatives: Flask (no native async), Django (heavier, more than needed), Node (fine, but the ML stack is Python, so one language reduces friction).

### Why two processes instead of one app?
- Different failure modes and restart needs. A stuck decoder should not drop dashboard connections.
- Different dependency footprints (OpenCV, ONNX vs a web stack).
- The Redis boundary makes the detector replaceable or scalable independently.
- Cost: two things to run and a contract (event schema) to keep in sync. For a single-machine demo a one-process version is simpler, and I would say that.

### Why Redis Streams? Why not Pub/Sub, Kafka, RabbitMQ?

| Option | Verdict for this project |
|---|---|
| **Redis Pub/Sub** | Fire-and-forget. A dashboard that is offline or reconnecting misses events, with no replay. Fine for a throwaway signal, wrong for detections. |
| **Redis Streams** | Append-only log with IDs, consumer groups, acknowledgements, and replay from an ID. Gives late-joining dashboards history and at-least-once delivery at very low operational cost. Chosen. |
| **Kafka** | Better for very high throughput, long retention, partitioning by key, multiple independent consumer groups, and multi-datacenter replication. My load is tens of events per second on one machine, so Kafka's ZooKeeper/KRaft, brokers and tuning would cost more than they return. I would move to Kafka at statewide scale. |
| **RabbitMQ** | Good task/work queue with routing. It deletes messages after ack, so no replay or log semantics. |

**Honest caveats to volunteer:** Redis is in-memory, so history is bounded by `MAXLEN` trimming and durability depends on AOF/RDB settings. That is why a durable store (Postgres) sits behind it in the production design. Streams give at-least-once, so consumers must be idempotent.

### Why threads and not processes or asyncio in the detector?
- Both hot calls (`cv2` read/decode, ONNX `session.run`) release the GIL, so threads run in parallel for this workload.
- Threads let one model instance be shared. Processes would each load a model copy and need IPC for frames.
- asyncio does not help because the blocking calls are C-level and not awaitable.
- Limit to admit: if pre/post-processing in Python grows, the GIL returns as a bottleneck, and processes or a worker pool are the answer.

### Why RTSP for detection and not WebRTC or HLS?
- The gateway's own spec assigns them: RTSP for AI inference, WHEP (WebRTC) for low-latency browser preview, HLS for dashboards and restricted networks.
- OpenCV can read RTSP (and HLS), but cannot consume WHEP. HLS adds segment latency (seconds).
- RTSP over TCP because UDP loses packets across NAT and produces corrupt frames that look like model errors.
- HLS is the fallback when port 8554 is blocked.

### Why PostgreSQL (and where does it sit)?
- Durable, queryable history: detections, cameras, watchlist, alerts. Redis is the fast path, Postgres is the record.
- **Write path:** a consumer-group worker batch-inserts from the stream. Never write from the camera threads or the alert path.
- **Schema sketch** (matches section 1D's real Redis field list — `event_id` is the actual idempotency key, not a composite unique constraint):

```sql
cameras(id text PK, name text, geom geography(Point,4326), department text, status text)
detections(event_id text PK, camera_id text REFERENCES cameras, plate_norm text NOT NULL,
           plate_raw text, latitude double precision, longitude double precision,
           location_known boolean NOT NULL, ocr_confidence_percent real, detector_confidence_percent real,
           captured_at timestamptz NOT NULL, pts_seconds double precision, frame int)
watchlist(plate_norm text PK, reason text, added_by text, added_at timestamptz)
alerts(id bigserial PK, detection_event_id text REFERENCES detections(event_id), delivered_at timestamptz)
CREATE INDEX ON detections (plate_norm, captured_at);   -- route lookup
CREATE INDEX ON detections USING brin (captured_at);     -- cheap time-range scans
```
- Route query: `SELECT camera_id, captured_at FROM detections WHERE plate_norm=$1 ORDER BY captured_at`. With PostGIS, `ST_MakeLine(geom ORDER BY captured_at)` builds the path, and `ST_DWithin` answers "cameras within 2 km of the last sighting".
- Scaling: partition `detections` by day and drop old partitions for retention. TimescaleDB is an option for heavy time-series.
- If you did not build this, say: "The prototype used Redis; Postgres is the durable-store design and here is the schema."

### Why SSE and not WebSocket?
- The flow is one-directional (server to browser). SSE is plain HTTP, auto-reconnects, supports `Last-Event-ID` resume, and passes proxies easily.
- WebSocket is right when the client also streams data. The watchlist edits use normal REST, so nothing needs a bidirectional socket.

### Why ONNX Runtime with the CoreML provider?
- No GPU was available. CoreML can use the M-series GPU and Neural Engine, and ONNX Runtime falls back to CPU per node.
- Caveats: CoreML compiles for specific input shapes, which is a plausible source of errors with mixed-resolution feeds. Benchmark CoreML against CPU-only on your own frames and keep whichever wins. Do not assume the accelerator is faster.

### Why frame sampling?
- A vehicle stays in view for many frames, so inferring on every frame wastes compute for the same plate.
- It sets throughput directly: `inferences/sec = cameras x fps / stride`.
- Trade-off: a higher stride raises the latency floor and can miss a fast vehicle. Tune it per camera.

### Why monotonic PTS, and why wall-clock for cross-camera routes?
- The gateway warns that arrival time is misleading (a buffered GOP arrives faster than real time) and feeds loop with hard scene cuts. I derive timestamps from PTS and keep them strictly increasing across resets with an offset.
- PTS is per feed and each feed loops independently, so PTS values are not comparable between cameras. Cross-camera ordering therefore uses wall-clock arrival. State this limitation openly.

---

## 4. System design questions

### "Scale this to 80,000 cameras statewide."
Work from numbers, not adjectives.
- **Compute:** assume 80,000 cameras at 1 inference/sec each = 80,000 inferences/sec. At ~5ms on a GPU with batching that is roughly 400 GPU-seconds/sec, so order of hundreds of GPUs. Cut it with motion gating (run detection only when something moves), lower per-camera rates, and edge inference.
- **Edge over central:** run detection at district edge nodes and ship only events (plate, time, camera, thumbnail), not video. Bandwidth drops by orders of magnitude and it matches the "no centralized video store" requirement.
- **Transport:** Kafka, partitioned by camera or region, replacing Redis Streams.
- **Storage:** Postgres/Timescale partitioned by time, hot/warm/cold tiers, object storage for snapshots.
- **State:** sharded watchlist in a cache with pub/sub invalidation.
- **Ops:** Kubernetes, autoscaling on queue depth (consumer lag), per-region failover.

### "What happens under overload?"
Utilization above 1 grows the queue without limit. Options in order of preference: **shed load deliberately** (drop the oldest frames, keep only the latest per camera), raise stride adaptively, prioritize cameras near a watchlisted vehicle's last sighting, then add capacity. Never let a slow consumer block a producer (the failure I hit in STAR 2).

### "Delivery guarantees?"
Redis Streams consumer groups give at-least-once. Make consumers idempotent: the unique key `(camera_id, captured_at, plate_norm)` makes reprocessing harmless. Exactly-once is unnecessary here because a duplicate sighting is cheap and a missed alert is expensive.

### "How do you avoid duplicate alerts?"
Cooldown per (plate, camera), for example one alert per 30 seconds, kept in Redis with `SET key NX EX 30`. Collapse consecutive same-camera sightings in the route view.

### "What if Redis goes down?"
The detector must not stop. Buffer events locally in a bounded queue and retry with backoff. Detections keep flowing to CSV as a last-resort log. Run Redis with persistence (AOF) and a replica, or Sentinel, in production.

### "What if one camera is bad?"
Isolation is per-thread: one failing reader retries on its own schedule and cannot affect the others. Add health tracking (last frame time, reconnect count), surface it in the UI, and pause cameras that fail repeatedly instead of hammering the gateway.

### "Security and privacy?"
- Credentials in environment variables or a git-ignored `.env`, never in code. Rotate if leaked.
- Role-based access per department, TLS everywhere, audit log of who searched which plate.
- Data minimization: store plates and snapshots, not continuous video. Retention limits and access logging matter for surveillance data.

### "How would you test it?"
- Unit tests: PTS monotonicity, plate normalization, catalogue parsing.
- Integration: a local RTSP server (MediaMTX) looping a known clip with known plates, so recall is measurable. Kill the stream mid-run to test reconnect.
- Load: step up camera count and record the saturation point.
- Chaos: stop Redis, drop the network, restart the gateway.

---

## 5. Levers that improve latency

| Lever | Effect | Trade-off |
|---|---|---|
| Lower stride | Lower latency floor, fewer misses | More compute |
| Adaptive stride (raise under load) | Keeps the pipeline stable | Slower alerts under load |
| Drop stale frames, keep latest per camera | Bounded latency | Some frames skipped |
| Decouple capture from inference | A slow model never stalls a stream | More code |
| Downscale before detection / crop ROI | Faster inference | Small plates may vanish |
| Hardware decode (VideoToolbox) | Frees CPU | Platform-specific |
| Quantized (INT8) model | Faster inference | Small accuracy loss |
| Batching frames | Higher throughput | Adds waiting time per batch |
| Keep the DB off the alert path | Removes ms of blocking | Alert and record are separate writes |
| Motion gating | Skips empty scenes | Needs a motion detector |

**Latency floor to quote:** sampling interval `stride / fps`, plus gateway GOP buffering, plus inference. Lowering the network or Redis time rarely matters compared with these.

## 6. Levers that improve detections and robustness

| Lever | What it fixes | Status |
|---|---|---|
| Plate normalization | Format noise | Built |
| Confidence thresholds | Junk reads | Tune it |
| Gate watchlist matches on the minimum character confidence, not the mean | One misread character hidden by a high average | Would add (current code averages) |
| Larger detector input, or tile the frame | Small, distant plates missed on wide scenes | Would test |
| Indian plate regex (state code + district + series + number) | Impossible strings | Would add |
| Fuzzy match, edit distance <= 1 with O/0, I/1, B/8, S/5 map | Missed watchlist hits from one wrong character | Would add |
| Multi-frame voting per vehicle | Random OCR errors | Would add |
| Tracker (ByteTrack) to associate detections across frames | Duplicates, best-frame choice | Would add |
| Best-frame selection (sharpest, largest plate) | Blur | Would add |
| Per-camera stride and thresholds | Uneven scenes | Would add |
| Fine-tune on Indian plates, night/IR data | Domain gap | Would do with data |
| Reconnect with jitter, staggered startup | Connection storms | Add (small change) |
| Heartbeat and health status per camera | Silent dead feeds | Would add |
| Graceful shutdown, bounded queues, thread-safe counters | Leaks and deadlocks | Partly built |

## 7. Metrics to measure before any interview (about an hour)

| Metric | How to measure |
|---|---|
| Inference time per frame | Time 200 `predict()` calls on real frames, report mean and p95 |
| Max stable cameras | Step 5/10/15/20/30, note when "read failed" or refusals begin |
| Achieved inferences/sec | Frames processed / elapsed (your stats counter already has frames) |
| End-to-end alert latency | Put `capture_wall_time` in the event, subtract from the browser receive time, report p50/p95 (same machine avoids clock skew) |
| Detection recall | Label the plates in a 2-minute clip by hand, count how many your system read correctly |
| Plate read accuracy | Exact-match rate of OCR text against your labels |
| Reconnect behavior | Kill a stream, log time to recover |
| Resource use | CPU %, memory, temperature/throttling from Activity Monitor |

Write these numbers in one place. They are your answers to every "how much / how fast" question.

## 8. Behavioral and reflective questions

- **Biggest challenge?** STAR 2: the throughput collapse, with the utilization arithmetic.
- **What would you do differently?** Decouple capture and inference from the start, benchmark on day one, and use a local RTSP simulator earlier.
- **What are the limitations?** Cross-camera ordering uses arrival time. Exact-match watchlist misses OCR errors. State is memory-bound. No re-identification beyond the plate.
- **What did you learn?** Overload behaves as a feedback loop: slow inference causes dropped streams, which causes reconnects, which cause more load. Measure utilization before adding features.
- **Your part vs the library?** The detector and OCR are open source. I built the live runner, integration and dashboard, and I can point to those files.

## 9. Learning resources (prioritized)

Links are from memory. Search the titles if one has moved.

**Learn first (a few hours each)**
1. **Little's Law and queueing basics.** Search "Little's Law utilization queue" and the "Handling Overload" and "Addressing Cascading Failures" chapters of the free Google SRE Book. This is the theory behind your STAR 2 story.
2. **Redis Streams.** redis.io docs, "Streams" data type page, plus the free Redis University course on Streams. Learn XADD, XREADGROUP, XACK, pending entries, `MAXLEN`.
3. **FastAPI concurrency.** fastapi.tiangolo.com/async/ (when to use `async def` vs `def`, and why blocking calls hurt).
4. **Python threads and the GIL.** Real Python's GIL article. Know exactly which calls release it.

**Then**
5. **Designing Data-Intensive Applications** (Kleppmann): chapter 11 (stream processing, delivery guarantees), chapter 8 (faults), chapter 3 (indexes).
6. **System Design Interview** (Alex Xu) or the ByteByteGo newsletter, for the interview format and estimation habits.
7. **ONNX Runtime execution providers**: onnxruntime.ai docs, CoreML EP page (input shapes, supported ops, CPU fallback).
8. **Kafka fundamentals**: kafka.apache.org/documentation intro, or Confluent's free "Kafka 101" course. Enough to argue when you would switch.
9. **PostgreSQL indexing and partitioning** (official docs: B-tree vs BRIN, declarative partitioning), then PostGIS intro workshops.
10. **Video basics**: what RTSP, RTP, GOP, keyframes (IDR) and PTS are. FFmpeg documentation and GStreamer tutorials. MediaMTX docs for running a local RTSP server to test with.
11. **Tracking and OCR quality**: the ByteTrack paper (Zhang et al., 2021) and the Levenshtein edit distance concept.
12. **WebRTC WHEP/WHIP**: skim the IETF drafts and MediaMTX docs so you can explain why WebRTC is for preview only.

## 10. Numbers cheat sheet (fill in and memorize)

- Cameras attempted / stable: `[ ]` / `[ ]`
- Frame stride and resulting fps per camera: `[ ]`
- Inference time per frame (mean / p95): `[ ]` / `[ ]`
- Demand vs capacity (inferences/sec): `[ ]` vs `[ ]`
- Alert latency p50 / p95: `[ ]` / `[ ]`
- Plate read accuracy and recall on your labeled clip: `[ ]`
- Reconnect time after a killed feed: `[ ]`
- Latency floor formula: `stride / fps` + GOP buffer + inference
- Utilization formula: `arrival rate / service rate` (above 1 means a growing queue)

---

## 11. Rapid-fire round (say each in 30 seconds)

| Question | Short answer |
|---|---|
| What does the project do? | Traces a vehicle across ~30 live CCTV feeds, shows its route on a map, and alerts when a watchlisted plate appears. |
| What did you build vs reuse? | Detector and OCR are open-source fast-alpr. I built the live multi-camera runner, the Redis/FastAPI integration and the dashboard. |
| Which models? | A YOLOv9 plate detector and a CCT OCR model, both ONNX. `[paste exact names]` |
| Why threads? | The heavy calls (OpenCV read, ONNX inference) release the GIL, so threads run in parallel and share one model. |
| What limits throughput? | Inference behind one lock: demand is cameras x fps / stride, and capacity is 1 / time per frame. |
| How did you find that? | Compared demand to capacity, then stepped camera count up and watched where failures began. |
| Why frame stride? | Plates stay in view for many frames, so skipping frames saves compute at the cost of a higher latency floor. |
| Why Redis Streams? | Decouples detector and API and lets a reconnecting dashboard resume; Pub/Sub would lose events. |
| Why not Kafka? | Load is tens of events/sec on one machine; Kafka is for statewide scale, replay and partitioning. |
| Why Postgres? | Redis is trimmed RAM. Postgres is the durable, queryable record. It sits behind Redis, off the alert path. |
| Why FastAPI? | Async HTTP/SSE front door; it never touches video so a decoder fault cannot take it down. |
| Why SSE? | One-way live push over HTTP with auto-reconnect and resume. WebSocket is more than needed. |
| Why RTSP for detection? | The gateway spec assigns it to AI inference. OpenCV cannot consume WebRTC, and HLS adds seconds of delay. |
| Why TCP for RTSP? | UDP drops packets across NAT and produces corrupt frames that look like model bugs. |
| How do you handle a dropped camera? | Per-thread reconnect with exponential backoff (2s to 30s); one camera cannot stall the others. |
| Why monotonic PTS? | Arrival time is misleading (buffered GOP arrives fast) and feeds loop with hard cuts. |
| How is the route ordered across cameras? | By arrival time, since each feed's PTS is independent. That is a stated limitation. |
| How do you reduce OCR errors? | Normalize the plate, threshold confidence, and (next) fuzzy match and multi-frame voting. |
| What is the alert latency floor? | stride / fps, plus gateway buffering, plus inference. |
| What would you change first? | Decouple capture from inference so a slow model never stalls a stream. |

## 12. Questions you can ask the interviewer

- "How does your team decide between a message queue like Redis Streams and Kafka as load grows?"
- "How do you measure and budget end-to-end latency in a video or streaming pipeline?"
- "How do you handle model updates and evaluation for a system where mistakes have real consequences?"
- "What does on-call look like for a system with many flaky external inputs?"

## 13. Three-day study plan

**Day 1 - Understand and measure.**
1. Read sections 0, 1, 1A, 1B, 1C twice. Fill the built-vs-designed table and paste your exact model names.
2. Run the measurements in section 7 and fill the section 10 cheat sheet.

**Day 2 - Depth.**
1. Read Little's Law and the overload chapters (section 9, items 1-4).
2. Practice STAR 1 to 4 out loud, with your real numbers, 2 minutes each.
3. Draw the architecture from memory on paper, twice.

**Day 3 - Mock interview.**
1. Have someone ask you the section 3 and 4 questions cold. Answer without notes.
2. Ask them to interrupt with "why not X?" on every design choice. Practice saying "I chose X because Y, and I would switch to Z when W."
3. Rehearse the honest answers: what you built vs reused, what is measured vs designed, and the known limits.

**If you only have one hour:** do the pitch (section 1), STAR 2 (the throughput collapse), the built-vs-designed table, and the rapid-fire table.
