"""Interactive manual ROI selection backends (browser and OpenCV GUI).

This module centralizes the *interactive* part of manual ROI drawing so the
heavy frame-reading functions (gray/asc/npy) only have to:

1. produce a display frame (BGR uint8),
2. ask the user to draw the ROI via :func:`select_roi_geometry`,
3. turn the returned geometry into a mask via :func:`build_manual_roi_mask`.

Two backends are provided:

* ``"cv2"``     – the classic OpenCV ``selectROI`` / mouse-callback windows. Only
  works on a machine with a display (``$DISPLAY`` / native GUI).
* ``"browser"`` – a tiny self-contained HTTP server (Python stdlib only) that
  serves an HTML canvas. The user draws the ROI in a web browser and the
  coordinates are POSTed back. This is the headless-server friendly backend
  (e.g. when running over VS Code Remote SSH, where the forwarded ``localhost``
  port can be opened in the local browser).

The backend is chosen by :func:`select_roi_geometry` from the ``backend``
argument or the ``MANUAL_ROI_BACKEND`` environment variable (``auto`` picks
``browser`` when no ``$DISPLAY`` is available).

Geometry returned by :func:`select_roi_geometry` is a plain dict, one of::

    {"mode": "rectangle", "rect": [x, y, w, h]}
    {"mode": "polygon",   "points": [[x, y], ...]}
    {"mode": "point",     "point": [x, y]}
    {"mode": "point_surrounding", "point": [x, y]}

All coordinates are in native image-pixel space.
"""

from __future__ import annotations

import base64
import json
import os
import socket
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, List, Optional, Union

import cv2
import numpy as np


MaskOrDict = Union[np.ndarray, Dict[str, np.ndarray]]


# ============================================================
# Backend selection
# ============================================================

def _resolve_backend(backend: Optional[str]) -> str:
    """Resolve the requested backend ("auto"/"browser"/"cv2")."""
    backend = (backend or os.environ.get("MANUAL_ROI_BACKEND", "auto")).strip().lower()
    if backend == "auto":
        # No display available -> use the browser backend (headless servers).
        backend = "cv2" if os.environ.get("DISPLAY") else "browser"
    if backend not in ("browser", "cv2"):
        raise ValueError(f"Unknown manual ROI backend: {backend!r} (use 'auto', 'browser' or 'cv2').")
    return backend


def select_roi_geometry(
    frame_display: np.ndarray,
    mode: str,
    display_title: str = "Draw ROI",
    backend: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
) -> Dict[str, object]:
    """Ask the user to draw an ROI and return its geometry.

    Parameters:
        frame_display: BGR uint8 image to draw on (native resolution).
        mode: "rectangle", "polygon", "point" or "point_surrounding".
        display_title: title shown in the window / browser page.
        backend: "auto" (default), "browser" or "cv2".
        host/port: only used by the browser backend (defaults from env).

    Returns:
        Geometry dict (see module docstring).
    """
    mode = str(mode).strip().lower()
    if mode not in ("rectangle", "polygon", "point", "point_surrounding"):
        raise ValueError(f"Unknown manual ROI mode: {mode!r}")

    chosen = _resolve_backend(backend)
    print(f"[manual ROI] Using '{chosen}' backend for ROI '{display_title}' (mode={mode}).")
    if chosen == "cv2":
        return _select_roi_cv2(frame_display, mode, display_title)
    return _select_roi_browser(frame_display, mode, display_title, host=host, port=port)


# ============================================================
# Mask construction (shared by all backends / frame sources)
# ============================================================

def build_manual_roi_mask(
    geometry: Dict[str, object],
    height: int,
    width: int,
    roi_name: str,
) -> MaskOrDict:
    """Build a binary mask (or dict of masks for ``point_surrounding``).

    Mirrors the original inline mask-building logic so behaviour is unchanged.
    """
    mode = str(geometry.get("mode", "")).strip().lower()
    mask = np.zeros((height, width), dtype=np.uint8)

    if mode == "rectangle":
        rect = geometry.get("rect")
        if not rect or len(rect) != 4:
            raise ValueError("No ROI selected. Please draw a rectangle ROI and press ENTER/Confirm.")
        x, y, w, h = (int(round(v)) for v in rect)
        if w <= 0 or h <= 0:
            raise ValueError("No ROI selected. Please draw a rectangle ROI and press ENTER/Confirm.")
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, thickness=-1)
        return mask

    if mode == "polygon":
        points = geometry.get("points") or []
        if len(points) < 3:
            raise ValueError("Polygon ROI needs at least 3 points. Please try again.")
        poly = np.array([[int(round(px)), int(round(py))] for px, py in points], dtype=np.int32)
        cv2.fillPoly(mask, [poly], 255)
        return mask

    if mode in ("point", "point_surrounding"):
        pt = geometry.get("point")
        if not pt or len(pt) != 2:
            raise ValueError("Point ROI needs a single click. Please try again.")
        x, y = int(round(pt[0])), int(round(pt[1]))
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError("Point ROI is out of bounds. Please try again.")
        if mode == "point":
            mask[y, x] = 255
            return mask
        # point_surrounding: center + 8 neighbours, each as its own sub-ROI mask.
        offsets = [
            (0, 0, "center"),
            (-1, 0, "left"),
            (1, 0, "right"),
            (0, -1, "up"),
            (0, 1, "down"),
            (-1, -1, "up_left"),
            (1, -1, "up_right"),
            (-1, 1, "down_left"),
            (1, 1, "down_right"),
        ]
        masks: Dict[str, np.ndarray] = {}
        for dx, dy, suffix in offsets:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                sub_mask = np.zeros((height, width), dtype=np.uint8)
                sub_mask[ny, nx] = 255
                masks[f"{roi_name}_{suffix}"] = sub_mask
        if not masks:
            raise ValueError("No valid surrounding pixels found for the selected point.")
        return masks

    raise ValueError(f"Unknown manual ROI mode: {mode!r}")


def save_selection_overlay(
    frame_display: np.ndarray,
    mask: MaskOrDict,
    save_path: str,
    color: tuple = (0, 255, 255),
    alpha: float = 0.45,
) -> None:
    """Save a PNG showing the selected ROI overlaid on the display frame.

    This is the manual-selection equivalent of the other saved plots: instead of
    only showing the selection in a window, the confirmed ROI is persisted so it
    can be reviewed later (e.g. in the batch plots directory).
    """
    if frame_display.ndim == 2:
        overlay_base = cv2.cvtColor(frame_display, cv2.COLOR_GRAY2BGR)
    else:
        overlay_base = frame_display.copy()

    # Union of all masks (dict -> single combined mask for visualization).
    if isinstance(mask, dict):
        combined = np.zeros(overlay_base.shape[:2], dtype=np.uint8)
        for sub in mask.values():
            combined = cv2.bitwise_or(combined, sub.astype(np.uint8))
    else:
        combined = mask.astype(np.uint8)

    overlay = overlay_base.copy()
    overlay[combined > 0] = color
    blended = cv2.addWeighted(overlay, alpha, overlay_base, 1.0 - alpha, 0)
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, color, 1)

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(save_path, blended)
    print(f"[manual ROI] Saved selection overlay to: {save_path}")


# ============================================================
# OpenCV (local GUI) backend
# ============================================================

def _select_roi_cv2(frame_display: np.ndarray, mode: str, display_title: str) -> Dict[str, object]:
    height, width = frame_display.shape[:2]

    if mode == "rectangle":
        cv2.namedWindow(display_title, cv2.WINDOW_NORMAL)
        roi_rect = cv2.selectROI(display_title, frame_display, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(display_title)
        x, y, w, h = roi_rect
        return {"mode": "rectangle", "rect": [int(x), int(y), int(w), int(h)]}

    if mode == "polygon":
        points: List[tuple] = []
        display = frame_display.copy()

        def draw_overlay(img, pts):
            canvas = img.copy()
            if pts:
                cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, (0, 255, 255), 1)
                for p in pts:
                    cv2.circle(canvas, p, 2, (0, 255, 255), -1)
            return canvas

        def on_mouse(event, x, y, flags, param):
            nonlocal points
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
            elif event == cv2.EVENT_RBUTTONDOWN and points:
                points.pop()

        cv2.namedWindow(display_title, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(display_title, on_mouse)
        while True:
            canvas = draw_overlay(display, points)
            cv2.imshow(display_title, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key == 13:  # Enter
                break
            if key == ord("c"):
                points = []
        cv2.destroyWindow(display_title)
        return {"mode": "polygon", "points": [[int(px), int(py)] for px, py in points]}

    # point / point_surrounding share the single-click interaction.
    point: List[tuple] = []
    display = frame_display.copy()

    def on_mouse_point(event, x, y, flags, param):
        nonlocal point
        if event == cv2.EVENT_LBUTTONDOWN:
            point = [(x, y)]

    cv2.namedWindow(display_title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(display_title, on_mouse_point)
    while True:
        canvas = display.copy()
        if point:
            cv2.circle(canvas, point[0], 2, (0, 255, 255), -1)
        cv2.imshow(display_title, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key == 13:  # Enter
            break
        if key == ord("c"):
            point = []
    cv2.destroyWindow(display_title)
    if len(point) != 1:
        return {"mode": mode, "point": None}
    return {"mode": mode, "point": [int(point[0][0]), int(point[0][1])]}


# ============================================================
# Browser (headless) backend
# ============================================================

_BROWSER_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>__TITLE__</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 16px; background: #1e1e1e; color: #eee; }
  h2 { margin: 0 0 8px 0; font-size: 18px; }
  p.hint { margin: 4px 0 12px 0; color: #aaa; }
  #c { image-rendering: pixelated; border: 1px solid #888; cursor: crosshair;
       width: min(95vw, __CSSW__px); height: auto; touch-action: none; }
  .bar { margin-top: 12px; }
  button { font-size: 14px; padding: 8px 14px; margin-right: 8px; cursor: pointer;
           border: 0; border-radius: 4px; background: #0e639c; color: #fff; }
  button.secondary { background: #444; }
  #status { margin-top: 10px; color: #6cc070; min-height: 20px; }
</style>
</head>
<body>
  <h2>__TITLE__</h2>
  <p class="hint" id="hint"></p>
  <canvas id="c"></canvas>
  <div class="bar">
    <button id="confirm">Confirm</button>
    <button id="undo" class="secondary">Undo last point</button>
    <button id="clear" class="secondary">Clear</button>
  </div>
  <div id="status"></div>
<script>
const MODE = "__MODE__";
const W = __W__, H = __H__;
const HINTS = {
  rectangle: "Drag to draw a rectangle, then press Confirm.",
  polygon: "Click to add polygon points (\u22653). Use Undo / Clear, then press Confirm.",
  point: "Click a single pixel, then press Confirm.",
  point_surrounding: "Click a single pixel (its 8 neighbours are added too), then press Confirm."
};
document.getElementById('hint').textContent = HINTS[MODE] || "";
const canvas = document.getElementById('c');
canvas.width = W; canvas.height = H;
const ctx = canvas.getContext('2d');
const img = new Image();
let rectStart = null, rectCur = null, dragging = false;
let points = [];
let point = null;
let submitted = false;

img.onload = redraw;
img.src = "data:image/png;base64,__IMG__";

function toImg(e){
  const r = canvas.getBoundingClientRect();
  let x = Math.round((e.clientX - r.left) * (canvas.width / r.width));
  let y = Math.round((e.clientY - r.top) * (canvas.height / r.height));
  x = Math.max(0, Math.min(W - 1, x));
  y = Math.max(0, Math.min(H - 1, y));
  return [x, y];
}
function redraw(){
  ctx.clearRect(0, 0, W, H);
  ctx.drawImage(img, 0, 0, W, H);
  ctx.lineWidth = 1; ctx.strokeStyle = 'yellow'; ctx.fillStyle = 'yellow';
  if (MODE === 'rectangle' && rectStart && rectCur){
    const x = Math.min(rectStart[0], rectCur[0]);
    const y = Math.min(rectStart[1], rectCur[1]);
    const w = Math.abs(rectCur[0] - rectStart[0]);
    const h = Math.abs(rectCur[1] - rectStart[1]);
    ctx.strokeRect(x + 0.5, y + 0.5, w, h);
  } else if (MODE === 'polygon' && points.length){
    ctx.beginPath();
    ctx.moveTo(points[0][0] + 0.5, points[0][1] + 0.5);
    for (let i = 1; i < points.length; i++) ctx.lineTo(points[i][0] + 0.5, points[i][1] + 0.5);
    ctx.stroke();
    for (const p of points){ ctx.fillRect(p[0] - 0.5, p[1] - 0.5, 2, 2); }
  } else if ((MODE === 'point' || MODE === 'point_surrounding') && point){
    ctx.beginPath(); ctx.arc(point[0] + 0.5, point[1] + 0.5, 2, 0, 2 * Math.PI); ctx.fill();
  }
}
canvas.addEventListener('mousedown', e => {
  const p = toImg(e);
  if (MODE === 'rectangle'){ rectStart = p; rectCur = p; dragging = true; }
  else if (MODE === 'polygon'){ points.push(p); }
  else { point = p; }
  redraw();
});
canvas.addEventListener('mousemove', e => {
  if (MODE === 'rectangle' && dragging){ rectCur = toImg(e); redraw(); }
});
window.addEventListener('mouseup', e => {
  if (MODE === 'rectangle' && dragging){ rectCur = toImg(e); dragging = false; redraw(); }
});
document.getElementById('undo').addEventListener('click', () => {
  if (MODE === 'polygon'){ points.pop(); redraw(); }
});
document.getElementById('clear').addEventListener('click', () => {
  points = []; point = null; rectStart = null; rectCur = null; redraw();
});
document.getElementById('confirm').addEventListener('click', () => {
  if (submitted) return;
  let payload = { mode: MODE };
  if (MODE === 'rectangle'){
    if (!rectStart || !rectCur){ alert('Draw a rectangle first.'); return; }
    const x = Math.min(rectStart[0], rectCur[0]);
    const y = Math.min(rectStart[1], rectCur[1]);
    const w = Math.abs(rectCur[0] - rectStart[0]);
    const h = Math.abs(rectCur[1] - rectStart[1]);
    if (w < 1 || h < 1){ alert('Rectangle is too small.'); return; }
    payload.rect = [x, y, w, h];
  } else if (MODE === 'polygon'){
    if (points.length < 3){ alert('Add at least 3 points.'); return; }
    payload.points = points;
  } else {
    if (!point){ alert('Click a point first.'); return; }
    payload.point = point;
  }
  submitted = true;
  fetch('/submit', { method: 'POST', headers: { 'Content-Type': 'application/json' },
                     body: JSON.stringify(payload) })
    .then(() => { document.getElementById('status').textContent =
                    'ROI submitted \u2713  You can close this tab and return to the terminal.'; })
    .catch(() => { submitted = false; alert('Failed to submit, please retry.'); });
});
</script>
</body>
</html>
"""


def _render_browser_page(frame_display: np.ndarray, mode: str, title: str) -> str:
    ok, png = cv2.imencode(".png", frame_display)
    if not ok:
        raise RuntimeError("Failed to encode frame for the browser ROI selector.")
    png_b64 = base64.b64encode(png.tobytes()).decode("ascii")
    h, w = frame_display.shape[:2]
    display_scale = max(1, int(round(800 / max(1, w))))
    cssw = int(w * display_scale)
    safe_title = title.replace("<", "&lt;").replace(">", "&gt;")
    return (
        _BROWSER_PAGE
        .replace("__TITLE__", safe_title)
        .replace("__MODE__", mode)
        .replace("__W__", str(int(w)))
        .replace("__H__", str(int(h)))
        .replace("__CSSW__", str(cssw))
        .replace("__IMG__", png_b64)
    )


def _pick_port(host: str, preferred: int) -> int:
    """Return ``preferred`` if free, otherwise an OS-assigned free port."""
    for candidate in (preferred, 0):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((host, candidate))
                return s.getsockname()[1]
        except OSError:
            continue
    return 0


def _select_roi_browser(
    frame_display: np.ndarray,
    mode: str,
    display_title: str,
    host: Optional[str] = None,
    port: Optional[int] = None,
) -> Dict[str, object]:
    host = host or os.environ.get("MANUAL_ROI_HOST", "127.0.0.1")
    if port is None:
        port = int(os.environ.get("MANUAL_ROI_PORT", "8000") or "8000")
    port = _pick_port(host, port)

    page = _render_browser_page(frame_display, mode, display_title)
    page_bytes = page.encode("utf-8")

    result: Dict[str, object] = {}
    done = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args, **kwargs):  # silence default logging
            return

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page_bytes)))
                self.end_headers()
                self.wfile.write(page_bytes)
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path != "/submit":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8"))
            except Exception:
                payload = {}
            result.update(payload)
            body = b'{"status":"ok"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            done.set()

    server = HTTPServer((host, port), Handler)
    actual_port = server.server_address[1]
    url = f"http://{host}:{actual_port}/"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    print("\n" + "=" * 70)
    print(f"[manual ROI] Draw ROI '{display_title}' (mode={mode}) in your browser:")
    print(f"             {url}")
    print("             (VS Code Remote auto-forwards this localhost port; "
          "Ctrl+Click the URL or open it locally.)")
    print("=" * 70 + "\n")
    try:
        webbrowser.open(url)
    except Exception:
        pass

    timeout_env = os.environ.get("MANUAL_ROI_TIMEOUT")
    timeout = float(timeout_env) if timeout_env else None
    finished = done.wait(timeout=timeout)
    server.shutdown()
    server.server_close()
    if not finished:
        raise TimeoutError(
            f"Manual ROI selection timed out after {timeout}s for '{display_title}'."
        )
    return result
