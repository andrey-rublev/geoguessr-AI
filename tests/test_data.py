import io
import itertools
import zipfile

import pytest
from PIL import Image

from geoguessr_ai.model.data import (
    download_osv5m,
    iter_folder_images,
    iter_zip_images,
    load_osv5m_labels,
    zip_image_ids,
)


def jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (10, 120, 200)).save(buf, format="JPEG")
    return buf.getvalue()


def test_zip_images_and_labels(tmp_path):
    shard = tmp_path / "00.zip"
    with zipfile.ZipFile(shard, "w") as zf:
        zf.writestr("00/111.jpg", jpeg_bytes())
        zf.writestr("00/222.jpg", jpeg_bytes())
        zf.writestr("00/readme.txt", "not an image")
    assert zip_image_ids(shard) == {"111", "222"}
    assert [image_id for image_id, _ in iter_zip_images(shard)] == ["111", "222"]

    csv = tmp_path / "train.csv"
    csv.write_text("id,latitude,longitude,country\n111,1.5,2.5,FR\n333,3.0,4.0,DE\n")
    labels = load_osv5m_labels(csv, ids={"111", "222"})
    assert list(labels.index) == ["111"]
    assert labels.loc["111", "latitude"] == 1.5


def test_folder_images(tmp_path):
    Image.new("RGB", (8, 8)).save(tmp_path / "a.png")
    csv = tmp_path / "labels.csv"
    csv.write_text("filename,latitude,longitude\na.png,10,20\nmissing.png,0,0\n")
    assert list(iter_folder_images(tmp_path, csv)) == [("a", tmp_path / "a.png", 10.0, 20.0)]


def test_download_validates_shards(tmp_path):
    with pytest.raises(ValueError, match="0..4"):
        download_osv5m(tmp_path, "test", [5])


def test_embedding_rests_after_each_batch_to_keep_a_laptop_cool(tmp_path, monkeypatch):
    import torch

    from geoguessr_ai.model import train as training
    from geoguessr_ai.model.embed import embed_items

    class Encoder:
        name = "fake"

        def encode(self, images):
            return torch.ones(len(images), 4)

    clock, slept = itertools.count(step=0.1), []
    monkeypatch.setattr(training.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(training.time, "sleep", slept.append)
    jpeg = io.BytesIO()
    Image.new("RGB", (8, 8)).save(jpeg, "JPEG")
    items = [(f"photo{i}", jpeg.getvalue(), 1.0, 2.0) for i in range(5)]

    assert embed_items(Encoder(), items, tmp_path / "x.npz", batch_size=2, rest=1.0) == 5
    assert len(slept) == 3  # after each batch of two, as long as it took


def test_embedding_carries_on_where_a_stopped_run_left_off(tmp_path):
    import numpy as np
    import torch

    from geoguessr_ai.model.embed import embed_items, progress_path
    from geoguessr_ai.model.train import resolve_embedding_files

    class Encoder:
        name = "fake"

        def __init__(self, fail_after=None):
            self.seen, self.fail_after = 0, fail_after

        def encode(self, images):
            if self.fail_after is not None and self.seen >= self.fail_after:
                raise KeyboardInterrupt  # stopped, or the laptop went down
            self.seen += len(images)
            return torch.ones(len(images), 4)

    jpeg = io.BytesIO()
    Image.new("RGB", (8, 8)).save(jpeg, "JPEG")
    items = [(f"photo{i}", jpeg.getvalue(), float(i), 2.0) for i in range(7)]
    out = tmp_path / "osv5m-train-05.npz"

    with pytest.raises(KeyboardInterrupt):
        embed_items(Encoder(fail_after=4), items, out, batch_size=2, save_every=4)
    assert progress_path(out).exists() and not out.exists()
    with pytest.raises(FileNotFoundError):
        resolve_embedding_files([tmp_path])  # the first part isn't taken for the shard

    again = Encoder()
    assert embed_items(again, items, out, batch_size=2, save_every=4) == 7
    assert again.seen == 3  # only what wasn't saved before
    with np.load(out) as saved:
        assert saved["ids"].tolist() == [f"photo{i}" for i in range(7)]
        assert saved["lat"].tolist() == list(range(7))
        assert saved["embeddings"].shape == (7, 4)
    assert not progress_path(out).exists()
