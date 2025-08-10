from typing import Tuple
import numpy as np
from PIL import Image, ImageOps, ImageFilter

def _to_grayscale(img: Image.Image) -> Image.Image:
    if img.mode != "L":
        img = img.convert("L")
    return img

def _maybe_invert(img: Image.Image) -> Image.Image:
    arr = np.asarray(img).astype(np.float32) / 255.0
    if arr.mean() > 0.5:
        img = ImageOps.invert(img)
    return img

def _binarize(img: Image.Image) -> Image.Image:
    arr = np.asarray(img)
    thr = np.percentile(arr, 80)
    arr = (arr > thr).astype(np.uint8) * 255
    return Image.fromarray(arr, mode="L")

def _bbox(arr: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(arr > 0)
    if len(xs) == 0 or len(ys) == 0:
        return 0, 0, arr.shape[1], arr.shape[0]
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

def preprocess_drawn_image(pil_img: Image.Image, out_size: int = 28, pad_ratio: float = 0.2) -> Image.Image:
    img = _to_grayscale(pil_img)
    img = ImageOps.autocontrast(img)
    img = _maybe_invert(img)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    img = _binarize(img)
    arr = np.asarray(img)
    x0, y0, x1, y1 = _bbox(arr)
    cropped = img.crop((x0, y0, x1, y1))
    w, h = cropped.size
    side = max(w, h)
    pad = int(side * pad_ratio)
    side_padded = side + 2 * pad
    canvas = Image.new("L", (side_padded, side_padded), 0)
    ox = (side_padded - w) // 2
    oy = (side_padded - h) // 2
    canvas.paste(cropped, (ox, oy))
    canvas = canvas.resize((out_size, out_size), Image.BICUBIC)
    canvas = ImageOps.autocontrast(canvas)
    return canvas
