import io
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
