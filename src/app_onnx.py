from typing import Union
import numpy as np
import gradio as gr
from PIL import Image
from src.infer_onnx import OnnxPredictor
from src.preprocess import preprocess_drawn_image
from PIL import ImageOps

pred = OnnxPredictor()

def _canvas_to_pil(data: Union[np.ndarray, dict]):
    if data is None:
        return None
    arr = data["image"] if isinstance(data, dict) and "image" in data else data
    arr = np.asarray(arr).astype("uint8")
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L")
    if arr.ndim == 3 and arr.shape[2] == 3:
        return Image.fromarray(arr, mode="RGB").convert("L")
    return Image.fromarray(arr, mode="RGBA").convert("L")

def predict_from_canvas(data) -> dict:
    img = _canvas_to_pil(data)
    if img is None:
        return {}
    out = pred.predict_pil(img, topk=5)
    return {k: float(round(v, 4)) for k, v in out}

def predict_from_upload(file) -> dict:
    if file is None:
        return {}
    path = file if isinstance(file, str) else getattr(file, "name", None)
    if path is None:
        return {}
    img = Image.open(path).convert("L")
    out = pred.predict_pil(img, topk=5)
    return {k: float(round(v, 4)) for k, v in out}

def preview_preprocess(data):
    img = _canvas_to_pil(data)
    if img is None:
        return None
    proc = preprocess_drawn_image(img, out_size=28, pad_ratio=0.2)
    proc = ImageOps.mirror(proc).rotate(90, expand=False)
    return proc

with gr.Blocks(title="EMNIST Handwrite (ONNX)") as demo:
    gr.Markdown("# EMNIST Handwrite (ONNX)\nRysuj cyfrę lub wgraj obraz. Otrzymasz Top-k z prawdopodobieństwami.")
    with gr.Tab("Rysuj"):
        with gr.Row():
            canvas = gr.Image(label="Panel rysowania", image_mode="L", type="numpy", height=280, width=280,
                              sources=None)
            out_tbl = gr.Label(num_top_classes=5, label="Top-k")
            prev = gr.Image(label="Podgląd 28×28", image_mode="L", type="pil", height=112, width=112)
        btn = gr.Button("Rozpoznaj")
        btn_prev = gr.Button("Pokaż preprocess")
        btn.click(predict_from_canvas, inputs=canvas, outputs=out_tbl)
        btn_prev.click(preview_preprocess, inputs=canvas, outputs=prev)
    with gr.Tab("Upload"):
        up = gr.File(label="Wgraj obraz PNG/JPG", type="filepath")
        out_tbl2 = gr.Label(num_top_classes=5, label="Top-k")
        btn2 = gr.Button("Rozpoznaj")
        btn2.click(predict_from_upload, inputs=up, outputs=out_tbl2)

if __name__ == "__main__":
    demo.launch()
