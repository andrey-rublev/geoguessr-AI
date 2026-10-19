import numpy as np
from PIL import Image

from geoguessr_ai.model.backbone import in_batches, square_crops


def test_in_batches_pads_the_last_batch_and_drops_what_the_padding_gave():
    pixels = np.arange(5 * 2, dtype=np.float32).reshape(5, 2)
    seen = []

    def infer(batch):
        seen.append(len(batch))
        return batch.sum(axis=1, keepdims=True)

    out = in_batches(pixels, 4, infer)
    assert seen == [4, 4]
    assert out.ravel().tolist() == [1, 5, 9, 13, 17]


def test_in_batches_needs_no_padding_for_a_full_batch():
    out = in_batches(np.ones((3, 2)), 3, lambda batch: batch * 2)
    assert out.shape == (3, 2) and (out == 2).all()


def test_square_crops_add_tiles_across_a_wide_view():
    crops = square_crops(Image.new("RGB", (1600, 900)))
    assert [c.size for c in crops] == [(1600, 900), (900, 900), (900, 900)]
