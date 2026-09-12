"""Geocells: split the globe into regions the model classifies images into.

Cells come from k-means on the training locations, so dense areas (Europe, the US)
get small cells and sparse ones (Siberia, the Sahara) get large ones. At inference we
pick the point with the highest *expected GeoGuessr score* under the model's
probabilities, rather than the single most likely cell: when the model is torn
between Paris and Brussels, a guess between them beats flipping a coin.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geo import EARTH_RADIUS_KM, SCORE_SCALE_KM, from_unit_vectors, to_unit_vectors


@dataclass
class GeoCells:
    centroids: np.ndarray
    """(K, 2) array of cell centres as (lat, lon) degrees."""

    @classmethod
    def fit(cls, lat: np.ndarray, lon: np.ndarray, n_cells: int, seed: int = 0) -> GeoCells:
        from sklearn.cluster import MiniBatchKMeans

        xyz = to_unit_vectors(lat, lon)
        n_cells = min(n_cells, len(xyz))
        kmeans = MiniBatchKMeans(
            n_clusters=n_cells, random_state=seed, batch_size=4096, n_init=3, max_iter=200
        ).fit(xyz)
        counts = np.bincount(kmeans.labels_, minlength=n_cells)
        centres = kmeans.cluster_centers_[counts > 0]  # drop clusters nothing was assigned to
        c_lat, c_lon = from_unit_vectors(centres)
        return cls(np.stack([c_lat, c_lon], axis=1))

    def __len__(self) -> int:
        return len(self.centroids)

    @property
    def unit_vectors(self) -> np.ndarray:
        return to_unit_vectors(self.centroids[:, 0], self.centroids[:, 1])

    def assign(self, lat, lon) -> np.ndarray:
        """Index of the nearest cell for each location."""
        return np.argmax(to_unit_vectors(lat, lon) @ self.unit_vectors.T, axis=-1)

    def distances_km(self, lat, lon) -> np.ndarray:
        """(N, K) great-circle distances from each location to each cell centre."""
        cos = np.clip(to_unit_vectors(lat, lon) @ self.unit_vectors.T, -1.0, 1.0)
        return EARTH_RADIUS_KM * np.arccos(cos)

    def soft_targets(self, lat, lon, tau_km: float) -> np.ndarray:
        """Haversine label smoothing: cells near the true location share the target mass."""
        logits = -self.distances_km(lat, lon) / tau_km
        logits -= logits.max(axis=-1, keepdims=True)
        weights = np.exp(logits)
        return weights / weights.sum(axis=-1, keepdims=True)

    def best_guess(self, probs: np.ndarray, top_n: int = 64) -> tuple[float, float, float]:
        """Choose the candidate centre that maximises expected score.

        Returns ``(lat, lon, expected_score)``; candidates are the ``top_n`` likeliest cells.
        """
        probs = np.asarray(probs, dtype=np.float64)
        top = np.argsort(-probs)[:top_n]
        xyz = self.unit_vectors[top]
        dist = EARTH_RADIUS_KM * np.arccos(np.clip(xyz @ xyz.T, -1.0, 1.0))
        expected = (5000.0 * np.exp(-dist / SCORE_SCALE_KM)) @ probs[top]
        best = top[int(np.argmax(expected))]
        return float(self.centroids[best, 0]), float(self.centroids[best, 1]), float(expected.max())
