"""Local dashboard for live FastALPR CCTV detections."""

# The dashboard's HTML, CSS, and JavaScript are kept inline in this standalone demo.
# ruff: noqa: E501

import argparse
import csv
import json
import logging
import os
import re
import threading
import time
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

import cv2
import numpy as np
from db.redis import (
    CameraLocation,
    create_sightings_publisher,
    env_switch,
    prepare_sighting,
)

from fast_alpr.live import (
    LIVE_CSV_FIELDS,
    Camera,
    LiveFeedRunner,
    choose_cameras,
    load_env_file,
)

LOGGER = logging.getLogger("fast_alpr.demo")


class DemoState:
    """Thread-safe event, watchlist, and route state for the dashboard."""

    def __init__(self, watchlist: set[str] | None = None) -> None:
        self._condition = threading.Condition()
        self._watchlist = watchlist or set()
        self._events: deque[dict[str, Any]] = deque(maxlen=500)
        self._sightings: deque[dict[str, Any]] = deque(maxlen=5000)
        self._sequence = 0
        self._cameras: dict[str, dict[str, Any]] = {}

    def set_cameras(self, cameras: list[Camera]) -> None:
        with self._condition:
            self._cameras = {
                camera.camera_id: {
                    "frames": 0,
                    "frame_number": None,
                    "jpeg": None,
                    "received_at": None,
                    "received_monotonic": None,
                }
                for camera in cameras
            }

    def publish_frame(self, camera_id: str, frame: Any, frame_number: int, pts: float | None) -> None:
        encoded_ok, encoded = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72]
        )
        if not encoded_ok:
            return
        received_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        with self._condition:
            camera = self._cameras.get(camera_id)
            if camera is None:
                return
            camera.update(
                {
                    "frames": camera["frames"] + 1,
                    "frame_number": frame_number,
                    "jpeg": encoded.tobytes(),
                    "received_at": received_at,
                    "received_monotonic": time.monotonic(),
                    "pts_seconds": pts,
                }
            )

    def camera_summaries(self) -> list[dict[str, object]]:
        now = time.monotonic()
        with self._condition:
            return [
                {
                    "camera_id": camera_id,
                    "frames": camera["frames"],
                    "frame_number": camera["frame_number"],
                    "received_at": camera["received_at"],
                    "pts_seconds": camera.get("pts_seconds"),
                    "status": (
                        "LIVE"
                        if camera["received_monotonic"] is not None
                        and now - camera["received_monotonic"] < 10
                        else "WAITING"
                    ),
                }
                for camera_id, camera in self._cameras.items()
            ]

    def camera_preview(self, camera_id: str) -> bytes | None:
        with self._condition:
            camera = self._cameras.get(camera_id)
            return None if camera is None else camera["jpeg"]

    @staticmethod
    def normalize_plate(plate: str) -> str:
        return re.sub(r"[^A-Z0-9]", "", plate.upper())

    def publish(self, row: dict[str, object]) -> None:
        plate = str(row.get("plate", ""))
        normalized = self.normalize_plate(plate)
        received_at = row.get("captured_at_utc")
        if not isinstance(received_at, str):
            received_at = datetime.now(timezone.utc).isoformat(timespec="microseconds")
        with self._condition:
            self._sightings.append(
                {
                    "received_at": received_at,
                    "captured_at_utc": received_at,
                    "camera_id": row.get("camera_id", ""),
                    "frame": row.get("frame", ""),
                    "plate": plate,
                    "normalized_plate": normalized,
                }
            )
            self._sequence += 1
            event_row = {**row, "received_at": received_at, "alert": normalized in self._watchlist}
            self._events.append({"id": self._sequence, "row": event_row})
            self._condition.notify_all()

    def events_after(self, sequence: int, timeout: float) -> list[dict[str, Any]]:
        with self._condition:
            self._condition.wait_for(
                lambda: any(event["id"] > sequence for event in self._events), timeout=timeout
            )
            return [event for event in self._events if event["id"] > sequence]

    def watchlist(self) -> list[str]:
        with self._condition:
            return sorted(self._watchlist)

    def add_watchlist_plate(self, plate: str) -> str:
        normalized = self.normalize_plate(plate)
        if not normalized:
            raise ValueError("Enter a plate containing at least one letter or number")
        with self._condition:
            self._watchlist.add(normalized)
        return normalized

    def remove_watchlist_plate(self, plate: str) -> bool:
        normalized = self.normalize_plate(plate)
        with self._condition:
            existed = normalized in self._watchlist
            self._watchlist.discard(normalized)
            return existed

    def route(self, plate: str) -> list[dict[str, object]]:
        normalized = self.normalize_plate(plate)
        with self._condition:
            return [
                {key: value for key, value in sighting.items() if key != "normalized_plate"}
                for sighting in sorted(
                    self._sightings,
                    key=lambda sighting: sighting["captured_at_utc"],
                    reverse=True,
                )
                if sighting["normalized_plate"] == normalized
            ]


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sentinel | Live ALPR</title>
<style>
:root{color-scheme:dark;--bg:#101514;--panel:#171e1c;--line:#2d3935;--muted:#9aa9a2;--text:#edf4ef;--green:#a8e4a0;--red:#ff7168;--yellow:#e8c878}
*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#17211e 0,#101514 42rem);color:var(--text);font:14px/1.45 ui-sans-serif,system-ui,sans-serif;min-height:100vh}
header{height:62px;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between;padding:0 clamp(16px,4vw,52px);background:#101514e8;position:sticky;top:0;z-index:2}
.brand{font-weight:750;letter-spacing:.04em}.brand span{color:var(--green)}.status{color:var(--muted);font-size:12px}.status i{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--green);margin-right:8px}
main{max-width:1440px;margin:auto;padding:28px clamp(16px,4vw,52px)}.topline{display:flex;justify-content:space-between;align-items:end;gap:20px;margin-bottom:20px}h1{font-size:25px;margin:0 0 3px}.sub{color:var(--muted);font-size:13px}.actions{display:flex;gap:9px;align-items:center}
button,a.button{font:inherit;border:1px solid var(--line);background:#202a26;color:var(--text);padding:9px 13px;text-decoration:none;border-radius:4px;cursor:pointer}button:hover,a.button:hover{border-color:var(--green)}input{font:inherit;color:var(--text);background:#101614;border:1px solid var(--line);padding:9px 10px;border-radius:4px;min-width:0}input:focus{outline:1px solid var(--green)}
.grid{display:grid;grid-template-columns:minmax(0,1fr) 290px;gap:22px}.section-title{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--line);padding:0 0 11px;margin:0 0 12px}.section-title h2{font-size:14px;margin:0;font-weight:650}.count{color:var(--muted);font-variant-numeric:tabular-nums}.table-wrap{overflow:auto;border:1px solid var(--line);background:#121917;border-radius:5px}table{border-collapse:collapse;width:100%;min-width:690px;text-align:left}th{font-size:10px;text-transform:uppercase;color:var(--muted);font-weight:650;letter-spacing:.08em;background:#19211e}th,td{padding:10px 12px;border-bottom:1px solid #26312d}td{font-size:12px;font-variant-numeric:tabular-nums}tbody tr:last-child td{border-bottom:0}tbody tr.alert{background:#4a2421}tbody tr.alert td:first-child{box-shadow:inset 3px 0 var(--red)}.plate{font-weight:750;font-size:13px;color:var(--green);cursor:pointer}.alert .plate{color:#ffaaa3}.empty{color:var(--muted);text-align:center;padding:26px}
aside section{padding:0 0 22px;margin-bottom:21px;border-bottom:1px solid var(--line)}.form{display:flex;gap:7px;margin:10px 0}.form input{flex:1;width:100px}.watch-items{display:grid;gap:6px}.watch-item{display:flex;justify-content:space-between;align-items:center;padding:8px 10px;background:#171e1c;border:1px solid var(--line);border-radius:4px;font-weight:650}.remove{padding:2px 8px;border:0;background:transparent;color:var(--muted);font-size:17px}.route-form{display:grid;grid-template-columns:1fr auto;gap:7px;margin-top:10px}.route-items{margin:12px 0 0;padding:0;list-style:none}.route-items li{padding:9px 0;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;gap:12px}.route-items small{display:block;color:var(--muted)}.hint{color:var(--muted);font-size:12px;margin:8px 0}
.feed-section{margin:0 0 26px}.camera-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr));gap:12px}.camera-card{overflow:hidden;border:1px solid var(--line);border-radius:5px;background:#121917}.camera-heading{display:flex;justify-content:space-between;align-items:center;padding:9px 11px;font-size:12px;font-weight:700}.camera-status{font-size:10px;letter-spacing:.06em;color:var(--yellow)}.camera-status.live{color:var(--green)}.camera-image{aspect-ratio:16/9;background:#080c0b;position:relative;display:grid;place-items:center}.camera-image img{width:100%;height:100%;object-fit:contain;display:block}.camera-image .hint{position:absolute}.camera-meta{padding:8px 11px;border-top:1px solid var(--line);color:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}
@media(max-width:850px){.grid{grid-template-columns:1fr}.topline{align-items:start;flex-direction:column}.actions{width:100%;flex-wrap:wrap}aside{display:grid;grid-template-columns:1fr 1fr;gap:20px}aside section{min-width:0}}@media(max-width:540px){aside{grid-template-columns:1fr}.actions>*{flex:1}h1{font-size:22px}}
</style>
</head>
<body>
<header><div class="brand"><span>◉</span> SENTINEL <span style="color:#9aa9a2;font-weight:450">/ LIVE ALPR</span></div><div class="status"><i></i><span id="connection">Connecting to event stream</span></div></header>
<main>
<div class="topline"><div><h1>Camera detections</h1><div class="sub">Incoming recognized plates, watchlist alerts, and camera sightings</div></div><div class="actions"><span id="event-count" class="count">0 detections</span><a class="button" href="/detections.csv" download>Download CSV</a></div></div>
<section class="feed-section"><div class="section-title"><h2>Live camera feeds</h2><span id="camera-summary" class="count">Waiting for frames</span></div><div id="cameras" class="camera-grid"></div></section>
<div class="grid"><section><div class="section-title"><h2>Live event stream</h2><span id="last-event" class="count">Waiting for detections</span></div><div class="table-wrap"><table><thead><tr><th>Received</th><th>Camera</th><th>Plate</th><th>OCR</th><th>Detector</th><th>Frame</th></tr></thead><tbody id="events"><tr><td colspan="6" class="empty">Waiting for camera detections...</td></tr></tbody></table></div></section>
<aside><section><div class="section-title"><h2>Watchlist</h2></div><form class="form" id="watch-form"><input id="watch-input" placeholder="Plate, e.g. ABC 123" required><button type="submit">Add</button></form><div id="watchlist" class="watch-items"></div><p class="hint">Matching plates are highlighted as alerts.</p></section>
<section><div class="section-title"><h2>Plate route</h2></div><form id="route-form" class="route-form"><input id="route-input" placeholder="Search a plate" required><button type="submit">Show</button></form><ol id="route" class="route-items"><li class="hint">Choose a plate from a detection or search above.</li></ol></section></aside></div>
</main>
<script>
const eventBody=document.querySelector('#events'), count=document.querySelector('#event-count'), connection=document.querySelector('#connection');let total=0;
function cell(row,value,cls=''){const el=document.createElement('td');el.textContent=value??'';if(cls)el.className=cls;row.append(el);return el}
function addEvent(item){const row=item.row;if(eventBody.querySelector('.empty'))eventBody.replaceChildren();const tr=document.createElement('tr');if(row.alert)tr.className='alert';cell(tr,new Date(row.received_at).toLocaleTimeString());cell(tr,row.camera_id);const plate=cell(tr,row.plate,'plate');plate.title='Show camera route';plate.onclick=()=>{document.querySelector('#route-input').value=row.plate;showRoute(row.plate)};cell(tr,`${row.ocr_confidence_percent}%`);cell(tr,`${row.detector_confidence_percent}%`);cell(tr,row.frame);eventBody.prepend(tr);while(eventBody.children.length>100)eventBody.lastElementChild.remove();total++;count.textContent=`${total} detection${total===1?'':'s'}`;document.querySelector('#last-event').textContent=row.alert?'WATCHLIST MATCH':'Latest: '+new Date(row.received_at).toLocaleTimeString()}
const source=new EventSource('/events');source.onopen=()=>connection.textContent='Event stream connected';source.onerror=()=>connection.textContent='Reconnecting to event stream';source.onmessage=e=>{try{addEvent(JSON.parse(e.data))}catch(error){console.error(error)}};
const cameraRoot=document.querySelector('#cameras'),cameraSummary=document.querySelector('#camera-summary');
async function refreshCameras(){try{const response=await fetch('/api/cameras');const cameras=await response.json();let live=0;for(const camera of cameras){let card=cameraRoot.querySelector(`[data-camera="${camera.camera_id}"]`);if(!card){card=document.createElement('article');card.className='camera-card';card.dataset.camera=camera.camera_id;const heading=document.createElement('div');heading.className='camera-heading';const name=document.createElement('span');name.textContent=camera.camera_id;const status=document.createElement('span');status.className='camera-status';heading.append(name,status);const imageWrap=document.createElement('div');imageWrap.className='camera-image';const image=document.createElement('img');image.alt='Latest inference frame from '+camera.camera_id;const empty=document.createElement('span');empty.className='hint';empty.textContent='Waiting for first sampled frame';image.onload=()=>{empty.hidden=true};image.onerror=()=>{image.hidden=true;empty.hidden=false};imageWrap.append(image,empty);const meta=document.createElement('div');meta.className='camera-meta';card.append(heading,imageWrap,meta);cameraRoot.append(card)}const status=card.querySelector('.camera-status');status.textContent=camera.status;status.classList.toggle('live',camera.status==='LIVE');if(camera.status==='LIVE')live++;const image=card.querySelector('img');if(camera.received_at&&image.dataset.version!==camera.received_at){image.dataset.version=camera.received_at;image.hidden=false;image.src='/preview/'+encodeURIComponent(camera.camera_id)+'.jpg?v='+encodeURIComponent(camera.received_at)}card.querySelector('.camera-meta').textContent=camera.frames?`${camera.frames} preview samples · frame ${camera.frame_number} · PTS ${camera.pts_seconds??'n/a'}s`:'No sampled frames yet'}cameraSummary.textContent=`${cameras.length} camera${cameras.length===1?'':'s'} · ${live} live`}catch(error){cameraSummary.textContent='Camera status unavailable';console.error(error)}}
refreshCameras();setInterval(refreshCameras,1000);
async function refreshWatchlist(){const response=await fetch('/api/watchlist');const plates=await response.json();const root=document.querySelector('#watchlist');root.replaceChildren();for(const plate of plates){const item=document.createElement('div');item.className='watch-item';const name=document.createElement('span');name.textContent=plate;const remove=document.createElement('button');remove.className='remove';remove.textContent='x';remove.title='Remove from watchlist';remove.onclick=async()=>{await fetch('/api/watchlist?plate='+encodeURIComponent(plate),{method:'DELETE'});refreshWatchlist()};item.append(name,remove);root.append(item)}}
document.querySelector('#watch-form').onsubmit=async e=>{e.preventDefault();const input=document.querySelector('#watch-input');const response=await fetch('/api/watchlist',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({plate:input.value})});const result=await response.json();if(!response.ok){alert(result.error);return}input.value='';refreshWatchlist()};
async function showRoute(plate){const root=document.querySelector('#route');root.replaceChildren();const response=await fetch('/api/route?plate='+encodeURIComponent(plate));const sightings=await response.json();if(!sightings.length){const li=document.createElement('li');li.className='hint';li.textContent='No sightings recorded for '+plate;root.append(li);return}for(const sighting of sightings){const li=document.createElement('li');const camera=document.createElement('span');camera.textContent=sighting.camera_id;const time=document.createElement('small');time.textContent=new Date(sighting.received_at).toLocaleString();camera.append(time);const frame=document.createElement('span');frame.className='count';frame.textContent='Frame '+sighting.frame;li.append(camera,frame);root.append(li)}}
document.querySelector('#route-form').onsubmit=e=>{e.preventDefault();showRoute(document.querySelector('#route-input').value)};refreshWatchlist();
</script>
</body></html>"""


def create_handler(state: DemoState, output_csv: Path) -> type[BaseHTTPRequestHandler]:
    class DemoHandler(BaseHTTPRequestHandler):
        def log_message(self, format_string: str, *args: object) -> None:
            LOGGER.info("%s - %s", self.address_string(), format_string % args)

        def _json(self, value: object, status: int = 200) -> None:
            body = json.dumps(value).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            request = urlsplit(self.path)
            if request.path == "/":
                body = PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif request.path == "/events":
                self._events()
            elif request.path == "/api/watchlist":
                self._json(state.watchlist())
            elif request.path == "/api/cameras":
                self._json(state.camera_summaries())
            elif request.path.startswith("/preview/") and request.path.endswith(".jpg"):
                camera_id = unquote(request.path[len("/preview/") : -len(".jpg")])
                image = state.camera_preview(camera_id)
                if image is None:
                    self._json({"error": "Preview not available"}, 404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(image)))
                self.end_headers()
                self.wfile.write(image)
            elif request.path == "/api/route":
                plate = parse_qs(request.query).get("plate", [""])[0]
                self._json(state.route(plate))
            elif request.path == "/detections.csv":
                self._csv()
            else:
                self._json({"error": "Not found"}, 404)

        def do_POST(self) -> None:
            if urlsplit(self.path).path != "/api/watchlist":
                self._json({"error": "Not found"}, 404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 4096:
                    raise ValueError("Request is too large")
                payload = json.loads(self.rfile.read(length))
                plate = payload.get("plate") if isinstance(payload, dict) else None
                if not isinstance(plate, str):
                    raise ValueError("A plate string is required")
                self._json({"plate": state.add_watchlist_plate(plate)}, 201)
            except (json.JSONDecodeError, ValueError) as error:
                self._json({"error": str(error)}, 400)

        def do_DELETE(self) -> None:
            request = urlsplit(self.path)
            if request.path != "/api/watchlist":
                self._json({"error": "Not found"}, 404)
                return
            plate = parse_qs(request.query).get("plate", [""])[0]
            self._json({"removed": state.remove_watchlist_plate(plate)})

        def _events(self) -> None:
            self.close_connection = True
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            try:
                sequence = int(self.headers.get("Last-Event-ID", "0"))
            except ValueError:
                sequence = 0
            try:
                while True:
                    events = state.events_after(sequence, 12)
                    if not events:
                        self.wfile.write(b": keep-alive\n\n")
                        self.wfile.flush()
                        continue
                    for event in events:
                        sequence = event["id"]
                        body = json.dumps(event["row"]).encode("utf-8")
                        self.wfile.write(b"id: " + str(sequence).encode() + b"\ndata: " + body + b"\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return

        def _csv(self) -> None:
            if output_csv.exists():
                body = output_csv.read_bytes()
            else:
                body = (",".join(LIVE_CSV_FIELDS) + "\n").encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition", 'attachment; filename="live_alpr_detections.csv"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return DemoHandler


def _publish_detection(
    state: DemoState, publisher: Any, row: dict[str, object]
) -> None:
    state.publish(row)
    if publisher is not None:
        try:
            publisher.publish(row)
        except Exception:
            LOGGER.exception("Failed to publish a detection to Redis")


def run_simulation(state: DemoState, output_csv: Path, stop_event: threading.Event) -> None:
    """Generate sample detections to exercise the dashboard without CCTV access."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=LIVE_CSV_FIELDS)
        writer.writeheader()
        csv_file.flush()
        frame = 0
        while not stop_event.is_set():
            camera_id = f"cam{frame % 4 + 1:02}"
            plate = "DEMO 123" if frame % 3 else "FAST 456"
            row: dict[str, object] = {
                "camera_id": camera_id,
                "frame": frame * 10,
                "pts_seconds": round(frame / 2, 3),
                "plate": plate,
                "ocr_confidence_percent": 96.4,
                "detector_confidence_percent": 98.1,
            }
            row = prepare_sighting(row, CameraLocation(0.0, 0.0), location_known=False)
            writer.writerow(row)
            csv_file.flush()
            state.publish(row)
            preview = np.zeros((360, 640, 3), dtype=np.uint8)
            preview[:] = (28, 38, 33)
            cv2.putText(
                preview,
                f"SIMULATED {camera_id} / FRAME {frame * 10}",
                (28, 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (168, 228, 160),
                2,
                cv2.LINE_AA,
            )
            state.publish_frame(camera_id, preview, frame * 10, round(frame / 2, 3))
            frame += 1
            stop_event.wait(1.5)


def _select_cameras(args: argparse.Namespace) -> list[Camera]:
    return choose_cameras(args.camera_id, args.start_index, args.end_index)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a live ALPR dashboard for CCTV detections.")
    parser.add_argument("--simulate", action="store_true", help="Use generated sample detections")
    parser.add_argument("--camera-id", action="append", help="Camera ID; repeat to select")
    parser.add_argument("--start-index", type=int, help="First camera index in the inclusive range")
    parser.add_argument("--end-index", type=int, help="Last camera index in the inclusive range")
    parser.add_argument("--protocol", choices=("rtsp", "hls"), default="rtsp")
    parser.add_argument("--redis-url", default=os.getenv("REDIS_URL"))
    parser.add_argument("--camera-locations", default=os.getenv("CAMERA_LOCATIONS_FILE"))
    parser.add_argument("--output-csv", default="live_alpr_detections.csv")
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser


def main(argv: list[str] | None = None) -> None:
    load_env_file()
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    if args.frame_stride < 1:
        raise SystemExit("--frame-stride must be at least 1")

    output_csv = Path(args.output_csv)
    state = DemoState({"DEMO123"} if args.simulate else None)
    stop_event = threading.Event()
    background: threading.Thread | None = None
    runner: LiveFeedRunner | None = None
    if args.simulate:
        state.set_cameras([Camera(f"cam{number:02}") for number in range(1, 5)])
        background = threading.Thread(
            target=run_simulation, args=(state, output_csv, stop_event), name="alpr-simulation", daemon=True
        )
        background.start()
        LOGGER.info("Simulation enabled; DEMO123 is on the watchlist")
    else:
        try:
            cameras = _select_cameras(args)
            state.set_cameras(cameras)
            publisher = create_sightings_publisher(
                cameras,
                args.redis_url,
                args.camera_locations,
                locations_enabled=env_switch("CAMERA_LOCATIONS_ENABLED"),
            )
            runner = LiveFeedRunner(
                cameras=cameras,
                protocol=args.protocol,
                output_csv=output_csv,
                frame_stride=args.frame_stride,
                on_row=lambda row: _publish_detection(state, publisher, row),
            )
            if publisher is not None:
                runner.set_row_transformer(publisher.prepare)
            runner.set_frame_callback(state.publish_frame)
        except (OSError, RuntimeError, ValueError) as error:
            raise SystemExit(str(error)) from error
        background = threading.Thread(target=runner.run, name="alpr-live-runner", daemon=True)
        background.start()
        LOGGER.info("Starting %d camera(s) using %s", len(cameras), args.protocol.upper())

    server = ThreadingHTTPServer((args.host, args.port), create_handler(state, output_csv))
    LOGGER.info("Dashboard listening at http://%s:%d", args.host, server.server_port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Stopping demo server")
    finally:
        server.shutdown()
        server.server_close()
        stop_event.set()
        if runner is not None:
            runner.stop()
        if background is not None:
            background.join(timeout=2)


if __name__ == "__main__":
    main()
