import hashlib
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from geoguessr_ai import fetch

CONTENT = bytes(range(256)) * 5000


@pytest.fixture
def served():
    """A server that cuts off its first answer partway, and carries on where it's asked to."""
    state = {"cut": True, "ranges": []}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            asked = self.headers.get("Range")
            start = int(asked.split("=")[1].rstrip("-")) if asked else 0
            state["ranges"].append(start)
            body = CONTENT[start:]
            self.send_response(206 if asked else 200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if state["cut"]:
                state["cut"] = False
                self.wfile.write(body[: len(body) // 3])
                self.wfile.flush()
                self.connection.close()
                return
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    state["url"] = f"http://127.0.0.1:{server.server_address[1]}/pack.zip"
    yield state
    server.shutdown()


def test_a_download_cut_off_carries_on_where_it_got_to(served, tmp_path):
    to = tmp_path / "data" / "pack.zip"
    expected = hashlib.sha256(CONTENT).hexdigest()

    fetch.download(served["url"], to, expected, len(CONTENT), "the model", wait=0.0)

    assert to.read_bytes() == CONTENT
    assert served["ranges"][0] == 0 and served["ranges"][1] > 0
    assert not (tmp_path / "data" / "pack.zip.partial").exists()


def test_a_damaged_download_is_refused_and_not_kept(served, tmp_path):
    to = tmp_path / "pack.zip"

    with pytest.raises(fetch.FetchFailed, match="damaged"):
        fetch.download(served["url"], to, "0" * 64, len(CONTENT), "the model", wait=0.0)

    assert not to.exists() and not (tmp_path / "pack.zip.partial").exists()


def test_says_how_a_download_went_wrong_when_it_never_works(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "DOWNLOAD_TRIES", 2)

    with pytest.raises(fetch.FetchFailed, match="Couldn't download the model"):
        fetch.download("http://127.0.0.1:9/", tmp_path / "x", "0", 10, "the model", wait=0.0)


def pack(path, names):
    with zipfile.ZipFile(path, "w") as files:
        for name in names:
            files.writestr(name, b"content of " + name.encode())


def test_unpacks_a_pack_into_the_folders_it_belongs_in(tmp_path):
    pack(tmp_path / "pack.zip", ["models/geoguessr.pt", "data/embeddings/a/b.npz"])

    made = fetch.unpack(tmp_path / "pack.zip", tmp_path / "root")

    assert (tmp_path / "root" / "models" / "geoguessr.pt").read_bytes().startswith(b"content")
    assert (tmp_path / "root" / "data" / "embeddings" / "a" / "b.npz").exists()
    assert len(made) == 2


@pytest.mark.parametrize(
    "name", ["../evil.pt", "models/../../evil.pt", "src/geoguessr_ai/cli.py", "evil.exe"]
)
def test_refuses_a_pack_that_would_put_anything_elsewhere(tmp_path, name):
    pack(tmp_path / "pack.zip", [name])

    with pytest.raises(fetch.FetchFailed):
        fetch.unpack(tmp_path / "pack.zip", tmp_path / "root")

    assert not (tmp_path / "evil.pt").exists()


def test_fetches_the_pack_only_if_the_model_isnt_there(tmp_path, monkeypatch, capsys):
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "geoguessr.pt").write_bytes(b"model")
    monkeypatch.setattr(fetch, "download", lambda *_: pytest.fail("it's there already"))
    ran = []

    fetch.fetch_model(tmp_path, (lambda: ran.append("step"),))

    assert ran == ["step"] and "ready to play" in capsys.readouterr().out


def test_fetches_unpacks_and_clears_away_the_pack(tmp_path, monkeypatch):
    def download(url, to, expected, size, what):
        to.parent.mkdir(parents=True)
        pack(to, ["models/geoguessr.pt"])

    monkeypatch.setattr(fetch, "download", download)

    fetch.fetch_model(tmp_path, ())

    assert (tmp_path / "models" / "geoguessr.pt").exists()
    assert not (tmp_path / "data" / "geoguessr-model.zip").exists()


def test_says_a_big_downloads_progress_in_lines_for_the_app(capsys, monkeypatch):
    monkeypatch.setattr(fetch, "SAY_EVERY_S", 0.0)
    lines = fetch._progress_as_lines("the image model")

    with lines(total=3) as counting_files:  # the files of a download, counted one by one
        counting_files.update(1)
    assert capsys.readouterr().out == ""

    with lines(total=10 * fetch.CHUNK, unit="B") as bytes_of_a_file:
        bytes_of_a_file.update(5 * fetch.CHUNK)
    assert "Fetching the image model: 50%" in capsys.readouterr().out
