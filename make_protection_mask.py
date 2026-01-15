#!/usr/bin/env python3
"""
Interactive protection mask creator for seam carving.

White (255)  = protected region
Black (0)    = not protected

Controls:
  - Left mouse drag: paint (white in paint mode, black in erase mode)
  - E: toggle erase mode
  - [: decrease brush size
  - ]: increase brush size
  - R: reset mask to all black
  - S: save mask
  - Q / ESC: quit

Example:
  python make_protection_mask.py --image input/plane.jpg --out input/plane_mask.png
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np


def _default_out_path(image_path: str) -> str:
    p = Path(image_path)
    return str(p.with_name(f"{p.stem}_mask.png"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Draw a white protection mask for seam carving.")
    ap.add_argument("--image", "-i", required=True, help="Path to input image (jpg/png/bmp).")
    ap.add_argument(
        "--out",
        "-o",
        default=None,
        help="Output mask path (PNG recommended). Default: <image_stem>_mask.png next to input image.",
    )
    ap.add_argument("--brush", type=int, default=30, help="Initial brush radius in pixels (default: 30).")
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Overlay opacity in [0..1] (default: 0.45).",
    )
    args = ap.parse_args()

    image_path = os.path.normpath(args.image)
    out_path = os.path.normpath(args.out) if args.out else _default_out_path(image_path)

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Could not read image: {image_path}")

    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    brush = max(1, int(args.brush))
    alpha = float(args.alpha)
    alpha = 0.0 if alpha < 0 else 1.0 if alpha > 1 else alpha

    drawing = False
    erase_mode = False
    window = "Mask Editor (E=erase, [ ]=brush, S=save, Q=quit)"

    def draw_at(x: int, y: int) -> None:
        nonlocal mask
        color = 0 if erase_mode else 255
        cv2.circle(mask, (x, y), brush, color, thickness=-1, lineType=cv2.LINE_AA)

    def on_mouse(event, x, y, flags, param) -> None:
        nonlocal drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            draw_at(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            draw_at(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)

    while True:
        # red overlay where mask is white
        overlay = img.copy()
        red = np.zeros_like(img)
        red[:, :, 2] = mask  # BGR: put mask into R channel
        overlay = cv2.addWeighted(overlay, 1.0, red, alpha, 0.0)

        mode_txt = "ERASE" if erase_mode else "PAINT"
        status = f"mode={mode_txt}  brush={brush}px  white_pixels={(mask > 0).sum():,}  out={out_path}"
        cv2.putText(
            overlay,
            status,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(window, overlay)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):  # q or ESC
            break
        if key == ord("e"):
            erase_mode = not erase_mode
        if key == ord("["):
            brush = max(1, brush - 2)
        if key == ord("]"):
            brush = min(500, brush + 2)
        if key == ord("r"):
            mask[:] = 0
        if key == ord("s"):
            out_dir = os.path.dirname(out_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            ok = cv2.imwrite(out_path, mask)
            if ok:
                print(f"Saved mask to: {out_path}")
            else:
                print(f"Failed to save mask to: {out_path}")

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

