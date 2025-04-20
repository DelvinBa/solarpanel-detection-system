import io
import pandas as pd
import pytest
from PIL import Image

# 1) fetch_house_ids.py
from solarpanel_detection_service.src.data_collection_service.fetch_house_ids import clean_vid, get_pids_and_vids

# 2) fetch_images.py
import solarpanel_detection_service.src.data_collection_service.fetch_images as fetch_images

# 3) fetch_location.py
import solarpanel_detection_service.src.data_collection_service.fetch_location as fetch_location


# ─── Tests for clean_vid & get_pids_and_vids ──────────────────────────────────

def test_clean_vid_nan_returns_none():
    assert clean_vid(pd.NA) is None
    assert clean_vid(float("nan")) is None

@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("123.0", "0123"),
        ("0123.0", "0123"),
        (123, "0123"),
        ("0456", "0456"),
        (" 789 ", "0789"),
    ],
)
def test_clean_vid_variants(raw, expected):
    assert clean_vid(raw) == expected

def test_get_pids_and_vids(monkeypatch):
    # prepare fake DataFrame
    df_in = pd.DataFrame({"pid": [1, 2], "vid": ["1.0", pd.NA]})
    monkeypatch.setattr(pd, "read_json", lambda url: df_in)

    calls = {}
    def fake_to_csv(self, path, index):
        calls["path"] = path
        calls["index"] = index
    monkeypatch.setattr(pd.DataFrame, "to_csv", fake_to_csv, raising=False)

    code = "GM0000"
    df_out = get_pids_and_vids(code)

    # pid should all be strings
    assert list(df_out["pid"]) == ["1", "2"]
    # vid cleaned; pd.NA→None→"None"
    assert list(df_out["vid"]) == ["01", "None"]

    assert calls["path"].endswith(f"data/interim/pid_vid_{code}.csv")
    assert calls["index"] is False


# ─── Tests for get_aerial_image ───────────────────────────────────────────────

class DummyResp:
    def __init__(self, status, content=b""):
        self.status_code = status
        self.content = content

def make_jpeg_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (1, 1), (123, 222, 111)).save(buf, format="JPEG")
    return buf.getvalue()

def test_get_aerial_image_success(monkeypatch, tmp_path):
    # stub requests.get → 200 + JPEG bytes
    monkeypatch.setattr(
        fetch_images.requests,
        "get",
        lambda *args, **kw: DummyResp(200, make_jpeg_bytes())
    )

    img = fetch_images.get_aerial_image(10, 20, offset=5)
    assert isinstance(img, Image.Image)
    assert img.size == (1, 1)

    # with save_path
    out = tmp_path / "out.jpg"
    img2 = fetch_images.get_aerial_image(0, 0, offset=1, save_path=str(out))
    assert out.exists()
    assert isinstance(img2, Image.Image)

def test_get_aerial_image_failure(monkeypatch):
    monkeypatch.setattr(
        fetch_images.requests,
        "get",
        lambda *args, **kw: DummyResp(404, b"")
    )
    assert fetch_images.get_aerial_image(0, 0) is None


# ─── Tests for get_coordinates ────────────────────────────────────────────────

class DummyLocResp:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload
    def json(self):
        return self._payload

def test_get_coordinates_success(monkeypatch):
    payload = {
        "verblijfsobject": {
            "geometrie": {"punt": {"coordinates": [100.1, 200.2, 300.3]}}
        }
    }
    monkeypatch.setattr(
        fetch_location.requests,
        "get",
        lambda *args, **kw: DummyLocResp(200, payload)
    )
    coords = fetch_location.get_coordinates("XYZ")
    assert coords == [100.1, 200.2]

def test_get_coordinates_missing_key(monkeypatch):
    monkeypatch.setattr(
        fetch_location.requests,
        "get",
        lambda *args, **kw: DummyLocResp(200, {})
    )
    assert fetch_location.get_coordinates("XYZ") is None

def test_get_coordinates_http_error(monkeypatch):
    monkeypatch.setattr(
        fetch_location.requests,
        "get",
        lambda *args, **kw: DummyLocResp(500, {})
    )
    assert fetch_location.get_coordinates("XYZ") is None
