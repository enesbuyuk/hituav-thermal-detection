"""FastAPI inference server for HIT-UAV Thermal Detection."""

import base64
import io
import os
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

MODEL_PATH = Path(os.getenv("MODEL_PATH", Path(__file__).parent / "models" / "best.pt"))
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.25"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:9999").split(",")
MAX_UPLOAD_MB = 15

if not MODEL_PATH.exists():
    raise RuntimeError(
        f"Model not found: {MODEL_PATH}\n"
        "Train a model (make train) or place best.pt in backend/models/"
    )

model = YOLO(str(MODEL_PATH))

CLASS_NAMES = {0: "Person", 1: "Car", 2: "Motorcycle", 3: "Other Vehicle"}

app = FastAPI(
    title="HIT-UAV Thermal Detection API",
    description="YOLOv8n object detection on infrared thermal UAV imagery.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _to_base64_jpeg(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode()


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH.name, "classes": list(CLASS_NAMES.values())}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(400, "Only image files are accepted.")

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"Image is larger than {MAX_UPLOAD_MB} MB.")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(400, "Could not decode the image.") from None

    start = time.perf_counter()
    results = model(image, conf=CONF_THRESHOLD, verbose=False)[0]
    inference_ms = (time.perf_counter() - start) * 1000

    # Draw English labels regardless of the class names stored in the weights
    results.names = CLASS_NAMES
    annotated_bgr = results.plot()
    annotated_pil = Image.fromarray(annotated_bgr[..., ::-1])

    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        detections.append({
            "class_name": CLASS_NAMES.get(cls_id, "Unknown"),
            "class_id": cls_id,
            "confidence": round(float(box.conf[0]), 3),
            "bbox": [round(x, 1) for x in box.xyxy[0].tolist()],
        })

    detections.sort(key=lambda x: x["confidence"], reverse=True)

    counts: dict[str, int] = {}
    for d in detections:
        counts[d["class_name"]] = counts.get(d["class_name"], 0) + 1

    return {
        "detections": detections,
        "counts": counts,
        "total": len(detections),
        "inference_ms": round(inference_ms, 1),
        "annotated_image": _to_base64_jpeg(annotated_pil),
        "original_image": _to_base64_jpeg(image),
    }
