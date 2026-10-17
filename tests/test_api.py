import base64
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.main import CLASS_NAMES, app

SAMPLES = sorted((Path(__file__).parent.parent / "samples").glob("*.jpg"))

client = TestClient(app)


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda p: p.name)
def test_detect_sample(sample):
    with open(sample, "rb") as f:
        res = client.post("/detect", files={"file": (sample.name, f, "image/jpeg")})
    assert res.status_code == 200

    data = res.json()
    assert data["total"] == len(data["detections"]) > 0
    assert sum(data["counts"].values()) == data["total"]
    assert set(data["counts"]) <= set(CLASS_NAMES.values())

    confidences = [d["confidence"] for d in data["detections"]]
    assert confidences == sorted(confidences, reverse=True)
    for d in data["detections"]:
        x1, y1, x2, y2 = d["bbox"]
        assert x1 < x2 and y1 < y2

    assert base64.b64decode(data["annotated_image"])[:2] == b"\xff\xd8"  # JPEG magic bytes


def test_rejects_non_image():
    res = client.post("/detect", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert res.status_code == 400


def test_rejects_corrupt_image():
    res = client.post("/detect", files={"file": ("broken.jpg", b"not really a jpeg", "image/jpeg")})
    assert res.status_code == 400
