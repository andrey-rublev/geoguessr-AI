"""Which way each pixel of Street View looks.

Street View draws the panorama through a pinhole camera: a pixel ``(x, y)`` looks along
``((x - cx) / f, (cy - y) / f, 1)`` from the camera, which faces the compass heading and is
tilted up by the pitch. The optical centre ``(cx, cy)`` is the middle of Google's whole Street
View frame, which OpenGuessr stretches above the top of the page, so it isn't the middle of the
calibrated view, and ``f`` is half the frame's height: the picture spans 90 degrees from top
to bottom. Dragging the picture turns the camera almost as if the spot under the mouse stayed
under it (see :data:`DRAG_SCALE`).

None of that shows on screen, so the bot measures it (see :func:`measure`): it drags the view
sideways and sees how far each detail moved, which depends on where the centre is and on ``f``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from .config import Point, Region

DEFAULT_VIEW_FOV = 105.0
"""Degrees across the calibrated view, assumed until the camera has been measured."""
DRAG_SCALE = 1.15
"""A drag turns the camera as if the picture were this much bigger than it is: dragging 100,
200 and 400 pixels from the centre of a frame 875 pixels high turned it 11.6, 22.0 and 37.2
degrees, against 12.9, 24.6 and 42.4 for keeping the spot under the mouse."""
GROUND_SIDE, GROUND_FAR = 6.0, 10.0
"""How far to the side and ahead :func:`ground_view` shows the road, in camera heights (the
Google car's camera is about 2.5 metres up), where Street View writes the road's name."""
GROUND_SCALE = 60
"""Pixels per camera height in :func:`ground_view`, which makes road names 30 to 50 pixels
tall."""
MIN_MATCHES = 40
MAX_ERROR = 2.5
"""Pixels: how far details may typically stray from where a measured camera says they moved."""


@dataclass(frozen=True)
class Camera:
    cx: float
    cy: float
    """The optical centre, in screen pixels: the point the camera faces."""
    f: float
    """Focal length in pixels."""
    measured: bool = False

    def direction(self, x: float, y: float, heading: float, pitch: float = 0.0):
        """``(azimuth, elevation)`` in degrees of the screen pixel ``(x, y)``, when the camera
        faces ``heading`` and is tilted up by ``pitch``."""
        right, up = (x - self.cx) / self.f, (self.cy - y) / self.f
        tilt = math.radians(pitch)
        forward = math.cos(tilt) - up * math.sin(tilt)
        rise = math.sin(tilt) + up * math.cos(tilt)
        azimuth = (heading + math.degrees(math.atan2(right, forward))) % 360.0
        return azimuth, math.degrees(math.atan2(rise, math.hypot(right, forward)))

    def tilt(self, y0: float, y1: float) -> float:
        """Degrees the camera tilts up when the picture is dragged from row ``y0`` to ``y1``
        (dragging down looks up)."""
        f = self.f * DRAG_SCALE
        return math.degrees(math.atan((self.cy - y0) / f) - math.atan((self.cy - y1) / f))

    def row_for_tilt(self, y0: float, degrees: float) -> float:
        """The row to drag the picture to from ``y0`` to tilt the camera up by ``degrees``."""
        f = self.f * DRAG_SCALE
        angle = math.atan((self.cy - y0) / f) - math.radians(degrees)
        return self.cy - f * math.tan(min(max(angle, -1.5), 1.5))


def default_camera(view: Region) -> Camera:
    """A guess: centred on the view, which spans :data:`DEFAULT_VIEW_FOV` degrees."""
    f = view.width / 2 / math.tan(math.radians(DEFAULT_VIEW_FOV / 2))
    return Camera(view.left + view.width / 2, view.top + view.height / 2, f)


def turned(points: np.ndarray, camera: Camera, yaw: float) -> np.ndarray:
    """Where screen pixels of a level camera end up after it turns ``yaw`` degrees right."""
    right = (points[:, 0] - camera.cx) / camera.f
    up = (camera.cy - points[:, 1]) / camera.f
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    forward = right * s + c
    return np.stack(
        [camera.cx + camera.f * (right * c - s) / forward, camera.cy - camera.f * up / forward],
        axis=1,
    )


def measure(
    before: np.ndarray, after: np.ndarray, view: Region, guess: Camera
) -> tuple[Camera, float] | None:
    """The camera, and how far it turned, from grayscale screenshots of ``view`` before and
    after dragging a level camera sideways. ``None`` when too few details match, as in fog or
    an empty desert, or they don't move as a camera turning on the spot would."""
    from scipy.optimize import least_squares

    orb = cv2.ORB_create(4000)
    k1, d1 = orb.detectAndCompute(before, None)
    k2, d2 = orb.detectAndCompute(after, None)
    if d1 is None or d2 is None or len(k1) < MIN_MATCHES or len(k2) < MIN_MATCHES:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < MIN_MATCHES:
        return None
    offset = np.array([view.left, view.top], dtype=np.float64)
    a = np.array([k1[m.queryIdx].pt for m in good]) + offset
    b = np.array([k2[m.trainIdx].pt for m in good]) + offset
    _, inliers = cv2.findHomography(a, b, cv2.RANSAC, 3.0)
    if inliers is None or inliers.sum() < MIN_MATCHES:
        return None
    a, b = a[inliers.ravel() == 1], b[inliers.ravel() == 1]

    def misses(p: np.ndarray) -> np.ndarray:
        return (turned(a, Camera(p[0], p[1], p[2]), p[3]) - b).ravel()

    shift = float(np.median(a[:, 0] - b[:, 0]))
    start = [guess.cx, guess.cy, guess.f, math.degrees(math.atan2(shift, guess.f))]
    fit = least_squares(misses, start, loss="soft_l1", f_scale=1.5, x_scale=[100, 100, 100, 5])
    cx, cy, f, yaw = fit.x
    error = float(np.median(np.hypot(*misses(fit.x).reshape(-1, 2).T)))
    if error > MAX_ERROR or not 0.2 * view.height < f < 5 * view.height:
        return None
    return Camera(float(cx), float(cy), float(f), measured=True), float(yaw)


def ground_view(image: Image.Image, origin: Point, camera: Camera) -> Image.Image | None:
    """The road round the car as if seen from straight above, from a level view whose top-left
    corner is at ``origin`` on screen. Street View writes road names flat on the road, so seen
    along it they are squashed and slanted; from above they come out straight, running along
    the road. ``None`` if the view doesn't reach below the horizon."""
    bottom = origin.y + image.height - 1
    if bottom <= camera.cy + 1:
        return None
    near = camera.f / (bottom - camera.cy)
    if near >= GROUND_FAR:
        return None
    width = round(2 * GROUND_SIDE * GROUND_SCALE)
    height = round((GROUND_FAR - near) * GROUND_SCALE)
    side = np.arange(width, dtype=np.float32) / GROUND_SCALE - GROUND_SIDE
    ahead = GROUND_FAR - np.arange(height, dtype=np.float32) / GROUND_SCALE  # far at the top
    xs = camera.cx - origin.x + camera.f * side[None, :] / ahead[:, None]
    ys = np.repeat((camera.cy - origin.y + camera.f / ahead)[:, None], width, axis=1)
    rgb = np.asarray(image.convert("RGB"))
    top_down = cv2.remap(rgb, xs.astype(np.float32), ys.astype(np.float32), cv2.INTER_LINEAR)
    return Image.fromarray(top_down)
