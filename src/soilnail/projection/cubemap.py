"""Equirect ↔ cubemap sampling (docs/PROJECTION.md)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import map_coordinates

from soilnail.projection.errors import NotFullEquirectError
from soilnail.projection.geometry import (
    apply_yaw,
    clamp_y,
    lonlat_to_equirect_xy,
    wrap_x,
    xyz_to_lonlat,
)

FACE_IDS: tuple[str, ...] = ("F", "R", "B", "L", "U", "D")
InterpName = Literal["bilinear", "bicubic", "nearest"]

# Face plane normals / centres on the unit cube (half-extent 0.5).
# For each face: (fixed_axis, fixed_sign, u_axis, u_sign, v_axis, v_sign)
# where axes are 0=X, 1=Y, 2=Z; image v increases downward so world "up" uses negative meshgrid y.
_FACE_SPEC: dict[str, tuple[int, float, int, float, int, float]] = {
    # fixed_axis, fixed_val, u_axis, u_mul, v_world_axis, v_world_mul (before image flip)
    "F": (2, +0.5, 0, +1.0, 1, +1.0),  # z=+0.5; u=+X; world_up=+Y
    "R": (0, +0.5, 2, -1.0, 1, +1.0),  # x=+0.5; u=−Z; world_up=+Y
    "B": (2, -0.5, 0, -1.0, 1, +1.0),  # z=−0.5; u=−X; world_up=+Y
    "L": (0, -0.5, 2, +1.0, 1, +1.0),  # x=−0.5; u=+Z; world_up=+Y
    "U": (1, +0.5, 0, +1.0, 2, -1.0),  # y=+0.5; u=+X; toward −Z at top of image
    "D": (1, -0.5, 0, +1.0, 2, +1.0),  # y=−0.5; u=+X; toward +Z at top of image
}


@dataclass(frozen=True)
class CubemapConfig:
    face_size: int = 960
    fov_deg: float = 90.0
    margin_deg: float = 0.0
    interpolation: InterpName = "bilinear"
    yaw_offset_deg: float = 0.0

    @property
    def effective_fov_deg(self) -> float:
        return self.fov_deg + 2.0 * self.margin_deg


def assert_full_equirect(width: int, height: int, *, slack: float = 0.0) -> None:
    if height <= 0 or width <= 0:
        raise NotFullEquirectError(f"invalid size {width}x{height}")
    aspect = width / height
    if abs(aspect - 2.0) > slack:
        raise NotFullEquirectError(
            f"expected aspect 2:1 (full equirect), got {width}x{height} (aspect={aspect})"
        )


def _interp_order(name: InterpName) -> int:
    if name == "nearest":
        return 0
    if name == "bilinear":
        return 1
    if name == "bicubic":
        return 3
    raise ValueError(f"unknown interpolation: {name}")


def _face_xyz_grid(
    face_id: str,
    face_size: int,
    effective_fov_deg: float,
) -> NDArray[np.float64]:
    """Unit directions for each pixel centre of one face. Shape (N, N, 3)."""
    if face_id not in _FACE_SPEC:
        raise KeyError(face_id)
    fixed_axis, fixed_val, u_axis, u_mul, v_axis, v_mul = _FACE_SPEC[face_id]
    # Pixel-centre UV in [-0.5, 0.5] matching py360convert linspace for fov=90.
    # For general FOV: scale so edge of face is tan(fov/2) in the face plane.
    # At fov=90, tan(45°)=1 → plane coords in [-0.5,0.5] with fixed=±0.5 is equivalent
    # to direction through cube face (normalize of those coords).
    extent = math.tan(math.radians(effective_fov_deg) / 2.0)
    # For fov=90, extent=1; py360convert uses plane coords in [-0.5,0.5] with fixed=0.5
    # so half-width of face plane is 0.5. Scale: plane_half = 0.5 * extent / tan(45°) = 0.5*extent
    plane_half = 0.5 * extent / math.tan(math.radians(90.0) / 2.0)  # = 0.5 * extent

    rng = np.linspace(-plane_half, plane_half, num=face_size, dtype=np.float64)
    # meshgrid: u across columns, v_world decreasing down the image (y image down)
    u, v_img = np.meshgrid(rng, rng)  # both increase with col/row
    # Image row increases downward → world "up" component decreases as row increases for side faces.
    # py360convert: x,y = meshgrid(rng, -rng) so first row has y=+plane_half (world up).
    v_world = -v_img

    xyz = np.zeros((face_size, face_size, 3), dtype=np.float64)
    xyz[..., fixed_axis] = fixed_val
    xyz[..., u_axis] = u_mul * u
    xyz[..., v_axis] = v_mul * v_world

    # Special-case U/D: py360convert Up uses y_flip on the Z assignment from meshgrid(-rng)
    # We already set v_world = -v_img; for U: z = v_world_mul * v_world with mul=-1
    # → z = -(-v_img) = v_img … wait need to match py360convert exactly for fov=90.
    #
    # py360convert Up: X=x, Y=0.5, Z=y_flip where x,y=meshgrid(rng,-rng), y_flip=flip(y,0)
    # meshgrid(rng,-rng): row0 y=+0.5, row-1 y=-0.5; flip(y,0): row0 y=-0.5, row-1 y=+0.5
    # So Up Z goes from -0.5 (top) to +0.5 (bottom) … actually flip of -rng rows:
    # y = meshgrid second: [[0.5,...,0.5],...,[-0.5,...,-0.5]]
    # flip(y,0): [[-0.5,...],...,[0.5,...]] → Z top=-0.5 (looking toward -Z? wait Z=-0.5 is back)
    # Top of Up face looks toward -Z (back). Our table says U toward −Z at top → z should be -plane_half at top.
    # With v_world=-v_img: top row v_img=-plane_half? No — linspace meshgrid both increase:
    # v_img row0 = -plane_half, row-1 = +plane_half if we used meshgrid(rng,rng)…
    # meshgrid(rng,rng): row0 all = -plane_half for second arg?
    # np.meshgrid(rng,rng): yi[0,:]=rng[0]=-plane_half. So top v_img=-plane_half, v_world=+plane_half.
    # U: z = (-1)*v_world = -plane_half at top. Good (toward -Z).
    # D: z = (+1)*v_world = +plane_half at top. py360convert Down: Z=y (not flipped) with y=meshgrid(rng,-rng)
    # so top Z=+0.5. Good.

    norms = np.linalg.norm(xyz, axis=-1, keepdims=True)
    out = xyz / np.maximum(norms, 1e-12)
    return np.asarray(out, dtype=np.float64)


def face_pixel_to_direction(
    face_id: str,
    u_px: float,
    v_px: float,
    *,
    face_size: int,
    fov_deg: float = 90.0,
    margin_deg: float = 0.0,
    yaw_offset_deg: float = 0.0,
) -> NDArray[np.float64]:
    """Direction for a continuous face pixel (pixel-centre grid).

    Analytical — does **not** build an N×N grid (that was O(N²) per query).
    """
    if face_id not in _FACE_SPEC:
        raise KeyError(face_id)
    fixed_axis, fixed_val, u_axis, u_mul, v_axis, v_mul = _FACE_SPEC[face_id]
    cfg_fov = fov_deg + 2.0 * margin_deg
    extent = math.tan(math.radians(cfg_fov) / 2.0)
    plane_half = 0.5 * extent
    # Match linspace(-plane_half, plane_half, face_size) at continuous index
    n = face_size
    ui = min(max(float(u_px), 0.0), n - 1.0)
    vi = min(max(float(v_px), 0.0), n - 1.0)
    if n == 1:
        u_plane = 0.0
        v_img = 0.0
    else:
        u_plane = -plane_half + (ui / (n - 1)) * (2.0 * plane_half)
        v_img = -plane_half + (vi / (n - 1)) * (2.0 * plane_half)
    v_world = -v_img
    xyz = np.zeros(3, dtype=np.float64)
    xyz[fixed_axis] = fixed_val
    xyz[u_axis] = u_mul * u_plane
    xyz[v_axis] = v_mul * v_world
    xyz /= max(float(np.linalg.norm(xyz)), 1e-12)
    return apply_yaw(xyz, yaw_offset_deg).reshape(3)


def direction_to_equirect_xy(
    direction: NDArray[np.floating],
    width: int,
    height: int,
) -> tuple[float, float]:
    lon, lat = xyz_to_lonlat(np.asarray(direction, dtype=np.float64).reshape(3))
    x, y = lonlat_to_equirect_xy(float(lon), float(lat), width, height)
    return float(x), float(y)


def equirect_to_cubemap(
    equirect: NDArray[np.uint8],
    config: CubemapConfig | None = None,
) -> dict[str, NDArray[np.uint8]]:
    """Project a full equirect RGB/gray image to 6 faces."""
    if config is None:
        config = CubemapConfig()
    if equirect.ndim == 2:
        equirect = equirect[..., None]
        squeeze = True
    else:
        squeeze = False
    if equirect.ndim != 3:
        raise ValueError(f"expected HxWxC, got {equirect.shape}")
    h, w = equirect.shape[:2]
    assert_full_equirect(w, h)

    order = _interp_order(config.interpolation)
    faces: dict[str, NDArray[np.uint8]] = {}
    for face_id in FACE_IDS:
        xyz = _face_xyz_grid(face_id, config.face_size, config.effective_fov_deg)
        xyz = apply_yaw(xyz, config.yaw_offset_deg)
        lon, lat = xyz_to_lonlat(xyz)
        xs = (lon / (2.0 * math.pi) + 0.5) * w - 0.5
        ys = (-lat / math.pi + 0.5) * h - 0.5
        xs = wrap_x(xs, w)
        ys = clamp_y(ys, h)
        # map_coordinates expects (row, col) = (y, x)
        coords = np.stack([ys.ravel(), xs.ravel()], axis=0)
        channels = []
        for c in range(equirect.shape[2]):
            sampled = map_coordinates(
                equirect[..., c].astype(np.float32),
                coords,
                order=order,
                mode="nearest",
                prefilter=order > 1,
            )
            channels.append(sampled.reshape(config.face_size, config.face_size))
        face = np.stack(channels, axis=-1)
        face = np.clip(np.rint(face), 0, 255).astype(np.uint8)
        if squeeze:
            face = face[..., 0]
        faces[face_id] = face
    return faces


def cubemap_to_equirect(
    faces: dict[str, NDArray[np.uint8]],
    *,
    height: int,
    width: int,
    config: CubemapConfig | None = None,
) -> NDArray[np.uint8]:
    """Inverse: sample equirect pixels from the face that owns each direction.

    Ownership: max |component| axis (standard cube face assignment).
    """
    if config is None:
        config = CubemapConfig()
    assert_full_equirect(width, height)
    # Build equirect lon/lat grid at pixel centres
    xs = np.arange(width, dtype=np.float64)
    ys = np.arange(height, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)
    lon = 2.0 * math.pi * (xx + 0.5) / width - math.pi
    lat = math.pi / 2.0 - math.pi * (yy + 0.5) / height
    # Undo yaw: rotate sphere by -yaw before face lookup
    from soilnail.projection.geometry import lonlat_to_xyz

    xyz = lonlat_to_xyz(lon, lat)
    xyz = apply_yaw(xyz, -config.yaw_offset_deg)
    # Face assignment by dominant axis
    abs_xyz = np.abs(xyz)
    dom = np.argmax(abs_xyz, axis=-1)  # 0=X,1=Y,2=Z
    # Map to face ids
    # X+: R, X-: L, Y+: U, Y-: D, Z+: F, Z-: B
    out_ch = faces["F"].shape[2] if faces["F"].ndim == 3 else 1
    out = np.zeros((height, width, out_ch), dtype=np.float32)

    face_size = config.face_size
    eff = config.effective_fov_deg
    extent = math.tan(math.radians(eff) / 2.0)
    plane_half = 0.5 * extent  # since tan(45)=1

    for face_id in FACE_IDS:
        fixed_axis, fixed_val, u_axis, u_mul, v_axis, v_mul = _FACE_SPEC[face_id]
        if fixed_axis == 0:
            mask = (dom == 0) & ((xyz[..., 0] > 0) == (fixed_val > 0))
        elif fixed_axis == 1:
            mask = (dom == 1) & ((xyz[..., 1] > 0) == (fixed_val > 0))
        else:
            mask = (dom == 2) & ((xyz[..., 2] > 0) == (fixed_val > 0))

        # Project to face plane at |fixed|=0.5
        scale = 0.5 / np.maximum(np.abs(xyz[..., fixed_axis]), 1e-12)
        px = xyz * scale[..., None]
        # plane coords
        u_plane = px[..., u_axis] / u_mul  # in [-0.5,0.5] for fov=90
        v_world = px[..., v_axis] / v_mul
        # Convert to face pixel coords: linspace(-plane_half, plane_half)
        # index = (val - (-plane_half)) / (2*plane_half) * (N-1)
        u_px = (u_plane - (-plane_half)) / (2.0 * plane_half) * (face_size - 1)
        v_img = (-v_world - (-plane_half)) / (2.0 * plane_half) * (face_size - 1)
        # Only inside face
        inside = (
            mask
            & (u_px >= -0.5)
            & (u_px <= face_size - 0.5)
            & (v_img >= -0.5)
            & (v_img <= face_size - 0.5)
        )
        if not np.any(inside):
            continue
        face = faces[face_id]
        if face.ndim == 2:
            face = face[..., None]
        yy_i, xx_i = np.where(inside)
        coords = np.stack([v_img[inside], u_px[inside]], axis=0)
        order = _interp_order(config.interpolation)
        for c in range(out_ch):
            sampled = map_coordinates(
                face[..., c].astype(np.float32),
                coords,
                order=order,
                mode="nearest",
                prefilter=order > 1,
            )
            out[yy_i, xx_i, c] = sampled

    out_u8 = np.clip(np.rint(out), 0, 255).astype(np.uint8)
    if out_ch == 1 and faces["F"].ndim == 2:
        return out_u8[..., 0]
    return out_u8


def make_cross_layout(faces: dict[str, NDArray[np.uint8]]) -> NDArray[np.uint8]:
    """Rasterize the 3×4 cross: empty, U, empty, empty / L F R B / empty D empty empty."""
    n = next(iter(faces.values())).shape[0]
    sample = faces["F"]
    ch = sample.shape[2] if sample.ndim == 3 else 0
    canvas: NDArray[np.uint8]
    if ch:
        canvas = np.zeros((3 * n, 4 * n, ch), dtype=np.uint8)
    else:
        canvas = np.zeros((3 * n, 4 * n), dtype=np.uint8)

    def put(face: str, row: int, col: int) -> None:
        r0, c0 = row * n, col * n
        canvas[r0 : r0 + n, c0 : c0 + n] = faces[face]

    put("U", 0, 1)
    put("L", 1, 0)
    put("F", 1, 1)
    put("R", 1, 2)
    put("B", 1, 3)
    put("D", 2, 1)
    return canvas
