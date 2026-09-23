import pytest

from geoguessr_ai.files import write_safely


def test_writes_the_file_and_leaves_nothing_else_behind(tmp_path):
    path = tmp_path / "models" / "geoguessr.pt"

    write_safely(path, lambda file: file.write(b"new"))

    assert path.read_bytes() == b"new"
    assert [p.name for p in path.parent.iterdir()] == ["geoguessr.pt"]


def test_a_write_that_fails_keeps_the_old_file_whole(tmp_path):
    path = tmp_path / "round.json"
    path.write_bytes(b"old")

    def fail(file):
        file.write(b"half")
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        write_safely(path, fail)

    assert path.read_bytes() == b"old"
    assert [p.name for p in tmp_path.iterdir()] == ["round.json"]
