# Install the project

Use the repository's normal uv project workflow from an activated environment:

```bash
uv sync
uv run fast-alpr-video assets/videos/Cam-1.mp4 \
  --output-video assets/videos/Cam-1-annotated.mp4 \
  --output-csv assets/videos/Cam-1-detections.csv
```

The declared project metadata already includes the ONNX Runtime dependency and the package console entrypoints for the CLI.
