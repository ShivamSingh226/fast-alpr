# Redis sightings

The live runner always publishes recognized plates to the Redis Stream `alpr:sightings`.
Install `redis-py` and make sure the Redis service is reachable before starting a real feed:

```dotenv
REDIS_URL=redis://localhost:6379/0
CAMERA_LOCATIONS_ENABLED=false
```

`CAMERA_LOCATIONS_ENABLED` defaults to false. In this mode sightings are still published, with
latitude/longitude `0.0` and `location_known=false`; map clients must not plot these as real camera
positions. No location file is required in this mode. Install the client in the active environment:

```sh
python -m pip install redis
```

For real coordinates, set `CAMERA_LOCATIONS_ENABLED=true` and
`CAMERA_LOCATIONS_FILE=db/camera_locations.json` in the root `.env` or shell. Every selected camera
must have verified coordinates in that JSON file:

```json
{
  "cam01": { "latitude": 40.7128, "longitude": -74.0060 },
  "cam02": { "latitude": 40.7134, "longitude": -74.0049 }
}
```

Replace the example coordinates with verified camera locations. Startup rejects missing camera
coordinates and checks that Redis is reachable before starting capture.

The CSV and Redis Stream use the same fields: `event_id`, `camera_id`, `plate`, `normalized_plate`,
`latitude`, `longitude`, `location_known`, `captured_at_utc`, the feed-local `pts_seconds`, frame
number, and confidence values.
The UTC timestamp is captured when the frame is read, before inference. PTS is local to one feed and
must not be used to order sightings across cameras. Redis Stream entry IDs reflect publish order;
consumers should sort cross-camera sightings by `captured_at_utc`.

`REDIS_URL` defaults to `redis://localhost:6379/0`. Run Redis locally, then start the batch runner
with `make live CAMERA_START_INDEX=1 CAMERA_END_INDEX=30` or the dashboard with
`make demo CAMERA_START_INDEX=1 CAMERA_END_INDEX=30`. The end index is inclusive, and the selected
cameras must all appear in the location file when real locations are enabled. Omitting both range
variables prompts for the start and end indexes.
