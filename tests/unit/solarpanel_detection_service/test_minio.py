import io
import pytest
from PIL import Image

# Adjust these imports as needed to match your actual module paths:
import solarpanel_detection_service.src.minio.minio_init as minio_init
import solarpanel_detection_service.src.minio.minio_utils as minio_utils


# ─── Tests for get_minio_client ────────────────────────────────────────────────

class DummyMinio:
    def __init__(self, endpoint, access_key, secret_key, secure):
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key
        self.secure = secure

def test_get_minio_client_uses_expected_params(monkeypatch):
    # Monkey‑patch the Minio constructor
    monkeypatch.setattr(minio_init, "Minio", DummyMinio)

    client = minio_init.get_minio_client()
    assert isinstance(client, DummyMinio)
    assert client.endpoint == "s3:9000"
    assert client.access_key == "minioadmin"
    assert client.secret_key == "minioadmin"
    assert client.secure is False


# ─── Tests for init_minio_buckets ──────────────────────────────────────────────

class FakeInitClient:
    def __init__(self):
        self.buckets_made = []
        self.put_objects = []
    def bucket_exists(self, bucket_name):
        # None exist at first
        return bucket_name in self.buckets_made
    def make_bucket(self, bucket_name):
        self.buckets_made.append(bucket_name)
    def list_objects(self, bucket_name, prefix, recursive):
        # no subfolder objects yet
        return []
    def put_object(self, bucket_name, object_name, data, length):
        self.put_objects.append((bucket_name, object_name))

def test_init_minio_buckets_creates_all(monkeypatch):
    fake = FakeInitClient()
    # stub out get_minio_client
    monkeypatch.setattr(minio_init, "get_minio_client", lambda: fake)

    # Run the initializer
    minio_init.init_minio_buckets()

    # All buckets should have been created
    expected_buckets = {"training-data", "inference-data", "models", "mlflow"}
    assert set(fake.buckets_made) == expected_buckets

    # Subfolders created for the two buckets that list them
    expected_puts = {
        ("training-data", "training_images/"),
        ("training-data", "labels/"),
        ("inference-data", "inference_images/"),
        ("inference-data", "detection_results/"),
    }
    assert set(fake.put_objects) == expected_puts


# ─── Tests for upload_image_to_minio ───────────────────────────────────────────

class FakeUtilsClient:
    def __init__(self, exists=False):
        self._exists = exists
        self.buckets_made = []
        self.put_objects = []
    def bucket_exists(self, bucket_name):
        return self._exists
    def make_bucket(self, bucket_name):
        self.buckets_made.append(bucket_name)
    def put_object(self, bucket_name, object_name, data, length, content_type=None):
        # read a bit to ensure it's bytes-like
        data.read(1)
        self.put_objects.append((bucket_name, object_name, length, content_type))

def make_test_image():
    img = Image.new("RGB", (2, 3), color=(10, 20, 30))
    return img

def test_upload_image_creates_bucket_and_puts(monkeypatch):
    fake = FakeUtilsClient(exists=False)
    # bypass its own get_minio_client
    monkeypatch.setattr(minio_utils, "get_minio_client", lambda: fake)

    img = make_test_image()
    obj = minio_utils.upload_image_to_minio(
        image=img, object_name="foo/bar.jpg", bucket_name="mybucket"
    )
    assert obj == "foo/bar.jpg"

    # because exists=False, it should have made the bucket
    assert fake.buckets_made == ["mybucket"]

    # and put_object called once with correct tuple
    bucket, name, length, ctype = fake.put_objects[0]
    assert bucket == "mybucket"
    assert name == "foo/bar.jpg"
    assert length > 0
    assert ctype == "image/jpeg"

def test_upload_image_uses_existing_bucket(monkeypatch):
    fake = FakeUtilsClient(exists=True)
    monkeypatch.setattr(minio_utils, "get_minio_client", lambda: fake)

    img = make_test_image()
    obj = minio_utils.upload_image_to_minio(
        image=img, object_name="baz.jpg", bucket_name="exbucket", minio_client=fake
    )
    assert obj == "baz.jpg"

    # bucket_exists=True, so make_bucket should not be called
    assert fake.buckets_made == []

    # put_object still called
    assert len(fake.put_objects) == 1
    assert fake.put_objects[0][1] == "baz.jpg"
