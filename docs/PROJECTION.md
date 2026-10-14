# 360° equirectangular ↔ cubemap conventions

**Status:** binding for `src/soilnail/projection/`. Implement against this document; do not invent alternate axes in call sites.

**Independent reference used for numerical cross-check:** [`py360convert`](https://github.com/sunset1995/py360convert) (installed in the venv). We match its FRBLUD / Y-up / Z-forward geometry so disagreements are bugs, not “style”.

---

## 1. Equirectangular image

| Symbol | Meaning |
|--------|---------|
| `H`, `W` | Image height, width in pixels. **Required:** `W == 2 * H` (full 360°×180°). Otherwise reject with `NotFullEquirectError`. |
| Longitude `λ` (lon) | `[-π, +π]` radians. **Image centre column** (`x = (W-1)/2`) is **λ = 0** (forward). |
| Latitude `φ` (lat) | `[-π/2, +π/2]`. Top row → `+π/2` (zenith), bottom → `−π/2` (nadir). |

### Pixel ↔ angle (pixel-centre convention)

For pixel indices `x ∈ [0, W)`, `y ∈ [0, H)` (float allowed when sampling):

```
λ = 2π * (x + 0.5) / W − π
φ =  π/2 − π * (y + 0.5) / H
```

Inverse:

```
x = (λ / 2π + 0.5) * W − 0.5
y = (−φ / π + 0.5) * H − 0.5
```

Longitude sampling **wraps** modulo `W` (not clamped). Latitude **clamps** to `[0, H−1]` after mapping (poles).

This matches `py360convert.utils.uv2coor` / `xyz2uv` (their `u≡λ`, `v≡φ`).

---

## 2. 3D axes (right-handed)

```
+X  right
+Y  up
+Z  forward  (looking out of the Front face)
```

Unit direction from angles:

```
x = cos(φ) * sin(λ)
y = sin(φ)
z = cos(φ) * cos(λ)
```

(Equivalent to `atan2(x,z)=λ`, `atan2(y, hypot(x,z))=φ` as in py360convert.)

---

## 3. Cubemap faces (FRBLUD)

Face names and cube planes (unit cube half-extent 0.5, as in py360convert `xyzcube`):

| Face id | Name | Cube plane | “Right” in face u | “Up” in face v (image up = −world when y decreases down the image) |
|---------|------|------------|-------------------|---------------------------------------------------------------------|
| `F` | Front | `z = +0.5` | `+X` | `+Y` |
| `R` | Right | `x = +0.5` | `−Z` | `+Y` |
| `B` | Back | `z = −0.5` | `−X` | `+Y` |
| `L` | Left | `x = −0.5` | `+Z` | `+Y` |
| `U` | Up | `y = +0.5` | `+X` | `−Z` |
| `D` | Down | `y = −0.5` | `+X` | `+Z` |

Face image: shape `(face_size, face_size, C)`, **no mirroring**. Row 0 is the top of the face image (increasing `v` goes down the image, matching OpenCV/PIL).

For a face of FOV `fov_deg` (baseline **90°**) and size `N = face_size`:

- Face UV in `[-0.5, 0.5]` (pixel centres), with tan-FOV mapping:
  - `extent = tan(fov/2)`
  - sample direction = normalize( face_basis(u * extent, v * extent) )

With `fov_deg=90`, `extent=1` and the mapping coincides with the unit-cube face construction used by py360convert (when margin is 0).

**Yaw offset:** optional rotation about `+Y` applied to all directions before face assignment (default 0). Used for a second-pass seam ablation.

**Margin (`margin_deg`):** increases effective FOV to `fov_deg + 2*margin_deg` so adjacent faces overlap on the sphere. Padding increases the angular footprint of a fixed `face_size` (lower px/deg) unless `face_size` is scaled up in config.

---

## 4. Face sizes (W/4 vs centre-density vs SAM-native)

Measured: `W=3840`, `H=1920`. Verified script: `outputs/pilot_cubemap/face_density_table.json`.

**Equirect equator sampling:** `W/(2π) ≈ 611.15` px/rad (`W/360 ≈ 10.667` px/deg).

**Gnomonic face of `N` px at 90° FOV:**

| Location | Density | Formula |
|----------|---------|---------|
| Face centre | `N/2` px/rad | `N / (2 · tan 45°) = N/2` |
| Edge midpoints | `N` px/rad | rises by `1/cos²θ` to θ=45° |

| Candidate `N` | Centre vs source | Edge-mid vs source | Meaning |
|---------------|------------------|--------------------|---------|
| **960 (= W/4)** | 480/611 ≈ **0.785×** (~21% under-sampled at centre) | ~1.57× | Longitude span of 90° on the equirect — **not** a centre-density match |
| **1024** | 512/611 ≈ 0.84× | ~1.67× | **SAM-native** — avoids double resampling |
| **1222 (≈ W/π)** | 611/611 ≈ **1.00×** | ~2.00× | **Centre-density match** to equator |

After margin/face-size ablation (`outputs/pilot_cubemap/ablation_decision.json`), restart baseline uses **`face_size=1024`**, `margin_deg=0` (smallest margin with count-once on stub seam tests; sphere merge still applied).

---

## 5. Diagram (face cross / dice layout)

```
        [ U ]
[ L ] [ F ] [ R ] [ B ]
        [ D ]
```

Horizon should be continuous across L–F–R–B. Zenith content in U; nadir (often operator) in D.

---

## 6. Colour / dtype

- Input/output: uint8 RGB, same colour space as stored JPEGs (no sRGB↔linear conversion).
- Interpolation: bilinear baseline; bicubic opt-in (tested).
- Rounding: sample in float32, clip to [0,255], round to nearest uint8.

---

## 7. Errors

| Error | When |
|-------|------|
| `NotFullEquirectError` | `W != 2*H`, or configured max aspect slack exceeded |
| `PartialPanoError` | GPano cropped-area fields present and non-full (none seen on this dataset) |
