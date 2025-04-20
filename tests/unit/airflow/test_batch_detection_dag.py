# import io
# import sys
# import os
# import shutil
# import pandas as pd
# import pytest
# import cv2
# import numpy as np
# from airflow.models import DagBag
# from types import SimpleNamespace
# from pathlib import Path



# # 1) Smoke‐test the DAG definition
# def test_dag_parses():
#     dagbag = DagBag(include_examples=False)
#     assert 'batch_detection' in dagbag.dags
#     dag = dagbag.get_dag('batch_detection')
#     # we only expect one task in this DAG
#     assert len(dag.tasks) == 1
#     assert dag.tasks[0].task_id == 'process_images'


# # Now import your module using ABSOLUTE paths
# from airflow.dags import batch_detection_dag as M  # Works!

# # 2) Test initialize_minio_client
# class FakeMinioClient:
#     def __init__(self, endpoint, access_key, secret_key, secure):
#         self.endpoint = endpoint
#         self.access_key = access_key
#         self.secret_key = secret_key
#         self.secure = secure
#         self._buckets = set()
#     def bucket_exists(self, name):
#         return name in self._buckets
#     def make_bucket(self, name):
#         self._buckets.add(name)
#     def put_object(self, bucket, obj, data, length):
#         # record that we tried to write
#         setattr(self, 'last_put', (bucket, obj))

# def test_initialize_minio_client_creates_bucket_and_folder(monkeypatch):
#     # stub out Minio so we don't talk to real MinIO
#     monkeypatch.setenv('MINIO_ENDPOINT', 'host')
#     monkeypatch.setenv('MINIO_PORT', '1234')
#     monkeypatch.setenv('MINIO_ACCESS_KEY', 'k')
#     monkeypatch.setenv('MINIO_SECRET_KEY', 's')
#     monkeypatch.setenv('MINIO_SECURE', 'false')
#     monkeypatch.setenv('MINIO_BUCKET', 'mybucket')

#     monkeypatch.setattr(M, 'Minio', FakeMinioClient)

#     client = M.initialize_minio_client()
#     # bucket should now exist
#     assert 'mybucket' in client._buckets
#     # and we should have tried to create a test folder
#     assert client.last_put[0] == 'mybucket'
#     assert client.last_put[1].startswith(M.INFERENCE_IMAGES_FOLDER)


# # 3) Test get_latest_model
# class FakeObj:
#     def __init__(self, name, tm):
#         self.object_name = name
#         self.last_modified = tm
# def test_get_latest_model_fallback(monkeypatch):
#     fake = SimpleNamespace(
#       bucket_exists=lambda b: False,
#       list_objects=lambda b, recursive: [],
#     )
#     monkeypatch.setattr(M, 'Minio', lambda *a, **k: fake)
#     # should return (None, fallback_path)
#     model_obj, path = M.get_latest_model(fake)
#     assert model_obj is None
#     assert path == M.YOLO_MODEL_PATH

# def test_get_latest_model_from_bucket(monkeypatch, tmp_path):
#     # simulate two PT files with different timestamps
#     objs = [
#       FakeObj('foo_best.pt', 1),
#       FakeObj('bar_best.pt', 2),
#       FakeObj('ignore.txt',   3)
#     ]
#     fake = SimpleNamespace(
#       bucket_exists=lambda b: True,
#       list_objects=lambda b, recursive: objs,
#       fget_object=lambda bucket, obj, dest: open(dest, 'wb').write(b'')
#     )
#     monkeypatch.setattr(M, 'Minio', lambda *a, **k: fake)
#     name, path = M.get_latest_model(fake)
#     assert name == 'bar_best.pt'
#     assert Path(path).exists()


# # 4) Test update_manifest_with_detection
# def test_update_manifest(monkeypatch, tmp_path):
#     # no existing manifest → new DataFrame
#     fake_stream = io.BytesIO()  # will simulate client.get_object
#     fake = SimpleNamespace(
#       get_object=lambda b,name: SimpleNamespace(read=lambda : b''),
#       put_object=lambda b,name,data,length: setattr(fake, 'upload_bytes', data.read())
#     )
#     monkeypatch.setattr(M, 'MINIO_BUCKET', 'bucket')
#     M.update_manifest_with_detection(fake, 'file1.jpg', 0.42)
#     # CSV uploaded should now contain file1.jpg and confidence 0.42
#     csv = fake.upload_bytes.decode('utf-8')
#     assert 'file1.jpg' in csv
#     assert '0.42' in csv


# # 5) Test process_image
# class SimpleBox:
#     def __init__(self, xyxy, cls, conf):
#         self.xyxy = [xyxy]
#         self.cls = [cls]
#         self.conf = [conf]
# class SimpleResult:
#     def __init__(self, boxes):
#         self.boxes = boxes
#         self.names = {0: 'solar_panel'}
# def test_process_image_full(monkeypatch, tmp_path):
#     # create a fake image file
#     src = tmp_path / "inference_images" / "img.jpg"
#     src.parent.mkdir()
#     src.write_bytes(cv2.imencode('.jpg', np.zeros((5,5,3), np.uint8))[1])

#     # stub client
#     fake = SimpleNamespace(
#         bucket_exists=lambda b: True,
#         list_objects=lambda b, prefix, recursive: [],
#         fget_object=lambda b,o,p: shutil.copy(src, p),
#         put_object=lambda *a, **k: setattr(fake, 'did_put', True),
#         get_object=lambda *a, **k: SimpleNamespace(read=lambda: b""),
#     )
#     # stub YOLO call
#     def fake_model(image):
#         return [SimpleResult([SimpleBox((0,0,1,1), 0, 0.8)])]
#     monkeypatch.setenv('BUCKET_NAME', 'bucket')
#     monkeypatch.setattr(M, 'BUCKET_NAME', 'bucket')
#     monkeypatch.setattr(M, 'INFERENCE_IMAGES_FOLDER', 'inference_images/')
#     monkeypatch.setattr(M, 'DETECTION_RESULTS_FOLDER', 'detection_results/')
#     monkeypatch.setattr(M, 'update_manifest_with_detection', lambda c,f,conf: setattr(fake, 'updated', (f,conf)))
#     monkeypatch.setattr(M, 'cv2', cv2)
#     monkeypatch.setattr(M, 'YOLO', lambda path: fake_model)
#     # run
#     M.process_image(fake, 'inference_images/img.jpg', fake_model)
#     assert getattr(fake, 'did_put', False)
#     assert getattr(fake, 'updated')[1] == 0.8


# # 6) You can likewise test process_images by stubbing
# #    initialize_minio_client, get_latest_model, YOLO, client.list_objects, etc.

