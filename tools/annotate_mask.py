#!/usr/bin/env python3
"""
Interactive SAM mask annotator for soil-nail reference images.

Uses a plain browser UI (FastAPI + HTML) so clicks give real pixel coordinates.
Gradio's Image.select does NOT return (x,y) in Gradio 6 — that caused
"click not registered".

  python tools/annotate_mask.py --variant B
  # open http://127.0.0.1:7860

Variant B (default) → data/refs/<cls>/masks/   sleeve + bar + hose roots
Variant A           → data/refs/<cls>/masks_A/ hole + contents
"""
from __future__ import annotations

import argparse
import base64
import sys
import threading
import webbrowser
from pathlib import Path

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.device import pick_device  # noqa: E402
from src.preprocess import overlay_mask  # noqa: E402

VARIANT_DIRS = {"B": "masks", "A": "masks_A"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cls", default="soil_nail")
    p.add_argument("--variant", choices=["A", "B"], default="B")
    p.add_argument("--image", type=str, default=None)
    p.add_argument("--sam_type", default="vit_h", choices=["vit_h", "vit_b", "vit_l", "vit_t"])
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--no-browser", action="store_true")
    return p.parse_args()


def mask_dir(cls: str, variant: str) -> Path:
    return ROOT / "data" / "refs" / cls / VARIANT_DIRS[variant]


def default_ckpt(sam_type: str) -> Path:
    if sam_type == "vit_t":
        return ROOT / "weights" / "mobile_sam.pt"
    return ROOT / "weights" / "sam_vit_h_4b8939.pth"


def load_sam(sam_type: str, ckpt: Path, device: torch.device):
    from segment_anything import SamPredictor, sam_model_registry

    if not ckpt.is_file():
        raise FileNotFoundError(f"SAM checkpoint not found: {ckpt}")
    if sam_type == "vit_t":
        sys.path.insert(0, str(ROOT / "third_party" / "Personalize-SAM"))
        from per_segment_anything import SamPredictor as PSamPredictor
        from per_segment_anything import sam_model_registry as preg

        sam = preg[sam_type](checkpoint=str(ckpt))
        sam.to(device=device).eval()
        return PSamPredictor(sam)
    key = sam_type if sam_type in sam_model_registry else "vit_h"
    sam = sam_model_registry[key](checkpoint=str(ckpt))
    sam.to(device=device).eval()
    return SamPredictor(sam)


def predict_from_clicks(predictor, points, labels):
    if not points:
        return None, None
    out = predictor.predict(
        point_coords=np.array(points, dtype=np.float32),
        point_labels=np.array(labels, dtype=np.int32),
        multimask_output=True,
    )
    return out[0], out[1]


def save_mask(out_dir: Path, stem: str, image_rgb: np.ndarray, mask: np.ndarray, variant: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    binary = (mask > 0).astype(np.uint8) * 255
    mask_path = out_dir / f"{stem}.png"
    overlay_path = out_dir / f"{stem}_overlay.jpg"
    cv2.imwrite(str(mask_path), binary)
    color = (0, 255, 0) if variant == "B" else (255, 0, 0)
    ov = overlay_mask(image_rgb, binary, color=color)
    cv2.imwrite(str(overlay_path), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
    area = 100.0 * (binary > 0).mean()
    msg = f"Saved {mask_path.name} (area {area:.1f}%)"
    if area > 25:
        msg += " — WARNING: area > 25%"
    return str(mask_path), area, msg


def encode_jpeg_b64(rgb: np.ndarray, max_side: int = 1400) -> tuple[str, float]:
    h, w = rgb.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        disp = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        disp = rgb
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(disp, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii"), scale


def render_preview(rgb, points, labels, masks, mask_idx):
    vis = rgb.copy()
    if masks is not None:
        vis = overlay_mask(vis, masks[mask_idx], color=(0, 220, 0))
    for (x, y), lab in zip(points, labels):
        color = (0, 255, 0) if lab == 1 else (255, 40, 40)
        cv2.circle(vis, (int(x), int(y)), 12, color, -1)
        cv2.circle(vis, (int(x), int(y)), 13, (255, 255, 255), 2)
    return vis


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Soil-nail mask annotator</title>
<style>
  :root { font-family: system-ui, sans-serif; color: #111; }
  body { margin: 0; background: #f4f4f4; }
  header { background: #1a1a1a; color: #fff; padding: 12px 18px; }
  header h1 { margin: 0; font-size: 18px; font-weight: 600; }
  header p { margin: 6px 0 0; font-size: 13px; opacity: .85; }
  .wrap { display: grid; grid-template-columns: 1.4fr 0.9fr; gap: 16px; padding: 16px; max-width: 1400px; margin: 0 auto; }
  @media (max-width: 980px) { .wrap { grid-template-columns: 1fr; } }
  .card { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 12px; }
  #stage { position: relative; display: inline-block; max-width: 100%; cursor: crosshair; user-select: none; }
  #stage img { max-width: 100%; height: auto; display: block; border-radius: 4px; }
  .row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; align-items: center; }
  button, label.mode {
    border: 1px solid #bbb; background: #fafafa; border-radius: 6px;
    padding: 8px 12px; font-size: 14px; cursor: pointer;
  }
  button.primary { background: #0b5fff; color: #fff; border-color: #0b5fff; }
  button:disabled { opacity: .45; cursor: not-allowed; }
  label.mode { display: inline-flex; gap: 6px; align-items: center; }
  label.mode.exclude { background: #ffe8e8; border-color: #e88; }
  #status {
    white-space: pre-wrap; font-family: ui-monospace, Menlo, Consolas, monospace;
    font-size: 12.5px; line-height: 1.4; background: #f7f7f7; border: 1px solid #e5e5e5;
    border-radius: 6px; padding: 10px; min-height: 280px;
  }
  .hint { font-size: 13px; color: #333; margin: 0 0 8px; }
</style>
</head>
<body>
<header>
  <h1>Soil-nail mask annotator — variant __VARIANT__</h1>
  <p>Click the photo to place points. Green = include, red = exclude. Saves to <code>__OUTDIR__</code></p>
</header>
<div class="wrap">
  <div class="card">
    <p class="hint"><b>Click directly on the image</b> (crosshair cursor). Do not click empty grey areas.</p>
    <div id="stage"><img id="img" alt="reference photo" draggable="false"/></div>
    <div class="row">
      <label class="mode" id="modeLabel"><input type="checkbox" id="exclude"/> Click = EXCLUDE (red)</label>
      <button id="undo">Undo</button>
      <button id="alt">Try alternate mask</button>
      <button id="reset">Reset clicks</button>
      <button id="save" class="primary">Save mask</button>
    </div>
    <div class="row">
      <button id="prev">← Previous</button>
      <button id="next">Next →</button>
      <span id="imgname" style="font-size:13px;color:#555"></span>
    </div>
  </div>
  <div class="card">
    <div id="status">Loading…</div>
  </div>
</div>
<script>
const excludeEl = document.getElementById('exclude');
const modeLabel = document.getElementById('modeLabel');
const img = document.getElementById('img');
const statusEl = document.getElementById('status');
const imgname = document.getElementById('imgname');

excludeEl.addEventListener('change', () => {
  modeLabel.classList.toggle('exclude', excludeEl.checked);
});

async function api(path, body) {
  const r = await fetch(path, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body || {})
  });
  const data = await r.json();
  if (!r.ok) throw new Error(data.detail || data.error || r.statusText);
  return data;
}

function applyState(data) {
  if (data.image_b64) img.src = 'data:image/jpeg;base64,' + data.image_b64;
  if (data.status) statusEl.textContent = data.status;
  if (data.name) imgname.textContent = data.name;
}

async function refresh() {
  applyState(await api('/api/state', {}));
}

img.addEventListener('click', async (e) => {
  const rect = img.getBoundingClientRect();
  if (rect.width < 1 || rect.height < 1) return;
  // Coordinates in *displayed* image pixels → server maps to full-res
  const x = (e.clientX - rect.left) * (img.naturalWidth / rect.width);
  const y = (e.clientY - rect.top) * (img.naturalHeight / rect.height);
  statusEl.textContent = 'Predicting…';
  try {
    applyState(await api('/api/click', {
      x: Math.round(x), y: Math.round(y),
      exclude: !!excludeEl.checked
    }));
  } catch (err) {
    statusEl.textContent = 'Error: ' + err.message;
  }
});

document.getElementById('undo').onclick = async () => applyState(await api('/api/undo', {}));
document.getElementById('alt').onclick = async () => applyState(await api('/api/alt', {}));
document.getElementById('reset').onclick = async () => applyState(await api('/api/reset', {}));
document.getElementById('save').onclick = async () => applyState(await api('/api/save', {}));
document.getElementById('prev').onclick = async () => applyState(await api('/api/nav', {dir: -1}));
document.getElementById('next').onclick = async () => applyState(await api('/api/nav', {dir: 1}));

refresh().catch(e => statusEl.textContent = 'Failed to load: ' + e.message);
</script>
</body>
</html>
"""


class ClickIn(BaseModel):
    x: int
    y: int
    exclude: bool = False


class NavIn(BaseModel):
    dir: int = Field(..., description="-1 or +1")


class AnnotatorApp:
    def __init__(self, images: list[Path], predictor, out_dir: Path, variant: str):
        self.images = images
        self.predictor = predictor
        self.out_dir = out_dir
        self.variant = variant
        self.idx = 0
        self.rgb = None
        self.disp_scale = 1.0  # display/full-res (encode_jpeg scale)
        self.points: list[list[int]] = []
        self.labels: list[int] = []
        self.masks = None
        self.scores = None
        self.mask_idx = 0
        self.last_msg = "Click the photo to add INCLUDE points."
        self._lock = threading.Lock()
        self.load_image(0)

    def load_image(self, i: int):
        i = max(0, min(i, len(self.images) - 1))
        self.idx = i
        path = self.images[i]
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise FileNotFoundError(path)
        self.rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(self.rgb)
        self.points, self.labels = [], []
        self.masks, self.scores, self.mask_idx = None, None, 0
        self.last_msg = f"Loaded {path.name}. Click sleeve / bar / hose roots (INCLUDE)."

    def status_text(self) -> str:
        n_pos = sum(1 for x in self.labels if x == 1)
        n_neg = sum(1 for x in self.labels if x == 0)
        lines = [
            f"Image {self.idx + 1}/{len(self.images)}: {self.images[self.idx].name}",
            f"Variant {self.variant} → {self.out_dir}",
            "",
            "HOW TO USE",
            "  1. Leave EXCLUDE unchecked. Click object (green).",
            "  2. Tick EXCLUDE. Click hole/board/soil to carve out (red).",
            "  3. Try alternate mask if blob is wrong.",
            "  4. Save mask, then Next.",
            "",
            f"Clicks: {n_pos} include, {n_neg} exclude",
        ]
        if self.masks is not None:
            area = 100.0 * float(self.masks[self.mask_idx].mean())
            score = float(self.scores[self.mask_idx])
            lines.append(
                f"Mask #{self.mask_idx + 1}/{len(self.masks)}  score={score:.3f}  area={area:.1f}%"
            )
        else:
            lines.append("Mask: (none yet)")
        lines += ["", self.last_msg]
        if self.variant == "B":
            lines += [
                "",
                "Variant B: sleeve + bar end + hose roots ONLY.",
                "Exclude: dark hole, rim soil, whiteboard, hand, boot, scaffold, date stamp.",
            ]
        return "\n".join(lines)

    def state_payload(self) -> dict:
        vis = render_preview(self.rgb, self.points, self.labels, self.masks, self.mask_idx)
        b64, scale = encode_jpeg_b64(vis)
        self.disp_scale = scale
        return {
            "image_b64": b64,
            "status": self.status_text(),
            "name": f"{self.idx + 1}/{len(self.images)}  {self.images[self.idx].name}",
            "disp_scale": scale,
            "full_hw": [int(self.rgb.shape[0]), int(self.rgb.shape[1])],
        }

    def add_click(self, x_disp: int, y_disp: int, exclude: bool):
        # Browser sends coords in the JPEG we returned (already scaled)
        scale = self.disp_scale or 1.0
        x = int(round(x_disp / scale))
        y = int(round(y_disp / scale))
        h, w = self.rgb.shape[:2]
        x = int(np.clip(x, 0, w - 1))
        y = int(np.clip(y, 0, h - 1))
        self.points.append([x, y])
        self.labels.append(0 if exclude else 1)
        self.masks, self.scores = predict_from_clicks(self.predictor, self.points, self.labels)
        self.mask_idx = int(np.argmax(self.scores))
        kind = "EXCLUDE" if exclude else "INCLUDE"
        self.last_msg = f"Added {kind} at full-res ({x}, {y})."


def build_server(app_state: AnnotatorApp, variant: str, out_dir: Path) -> FastAPI:
    app = FastAPI(title="Soil-nail mask annotator")

    @app.get("/", response_class=HTMLResponse)
    def index():
        html = HTML.replace("__VARIANT__", variant).replace("__OUTDIR__", str(out_dir))
        return HTMLResponse(html)

    @app.post("/api/state")
    def state():
        with app_state._lock:
            return JSONResponse(app_state.state_payload())

    @app.post("/api/click")
    def click(body: ClickIn):
        with app_state._lock:
            app_state.add_click(body.x, body.y, body.exclude)
            return JSONResponse(app_state.state_payload())

    @app.post("/api/undo")
    def undo():
        with app_state._lock:
            if app_state.points:
                app_state.points.pop()
                app_state.labels.pop()
                if app_state.points:
                    app_state.masks, app_state.scores = predict_from_clicks(
                        app_state.predictor, app_state.points, app_state.labels
                    )
                    app_state.mask_idx = int(np.argmax(app_state.scores))
                else:
                    app_state.masks = app_state.scores = None
                    app_state.mask_idx = 0
                app_state.last_msg = "Undid last click."
            else:
                app_state.last_msg = "Nothing to undo."
            return JSONResponse(app_state.state_payload())

    @app.post("/api/alt")
    def alt():
        with app_state._lock:
            if app_state.masks is None:
                app_state.last_msg = "No mask yet — click first."
            else:
                app_state.mask_idx = (app_state.mask_idx + 1) % len(app_state.masks)
                app_state.last_msg = f"Showing alternate mask #{app_state.mask_idx + 1}."
            return JSONResponse(app_state.state_payload())

    @app.post("/api/reset")
    def reset():
        with app_state._lock:
            app_state.load_image(app_state.idx)
            return JSONResponse(app_state.state_payload())

    @app.post("/api/save")
    def save():
        with app_state._lock:
            if app_state.masks is None:
                app_state.last_msg = "Nothing to save — click the photo first."
            else:
                stem = app_state.images[app_state.idx].stem
                _p, _a, msg = save_mask(
                    app_state.out_dir,
                    stem,
                    app_state.rgb,
                    app_state.masks[app_state.mask_idx],
                    app_state.variant,
                )
                app_state.last_msg = "✓ " + msg
            return JSONResponse(app_state.state_payload())

    @app.post("/api/nav")
    def nav(body: NavIn):
        with app_state._lock:
            app_state.load_image(app_state.idx + int(body.dir))
            return JSONResponse(app_state.state_payload())

    return app


def main():
    args = parse_args()
    out_dir = mask_dir(args.cls, args.variant)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = ROOT / "data" / "refs" / args.cls / "images"
    if args.image:
        images = [Path(args.image)]
    else:
        images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    if not images:
        print(f"No images under {img_dir}")
        return 1

    ckpt = Path(args.ckpt) if args.ckpt else default_ckpt(args.sam_type)
    device = pick_device()
    print(f"Loading SAM {args.sam_type} from {ckpt} on {device} …")
    predictor = load_sam(args.sam_type, ckpt, device)

    state = AnnotatorApp(images, predictor, out_dir, args.variant)
    app = build_server(state, args.variant, out_dir)

    url = f"http://{args.host}:{args.port}"
    print("\n" + "=" * 60)
    print("Annotator ready — open in your browser:")
    print(f"  {url}")
    print("=" * 60)
    print("Click the IMAGE (crosshair). Tick EXCLUDE for red points. Save → Next.\n")

    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
