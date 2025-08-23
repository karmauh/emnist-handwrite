from typing import List, Tuple, Union
import numpy as np
import gradio as gr
from PIL import Image
from src.infer import Predictor
from src.preprocess import preprocess_drawn_image
from PIL import ImageOps

pred = Predictor()

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

def predict_from_canvas(data, threshold: float):
    img = _canvas_to_pil(data)
    if img is None:
        return {}, "Brak obrazu"
    out = pred.predict_pil(img, topk=5)
    top1 = out[0][1] if out else 0.0
    msg = "OK" if top1 >= threshold else "Niepewne, narysuj ponownie"
    return {k: float(round(v, 4)) for k, v in out}, msg

def predict_from_upload(file, threshold: float):
    if file is None:
        return {}, "Brak pliku"
    path = file if isinstance(file, str) else getattr(file, "name", None)
    if path is None:
        return {}, "Brak pliku"
    img = Image.open(path).convert("L")
    out = pred.predict_pil(img, topk=5)
    top1 = out[0][1] if out else 0.0
    msg = "OK" if top1 >= threshold else "Niepewne, narysuj ponownie"
    return {k: float(round(v, 4)) for k, v in out}, msg

def preview_preprocess(data):
    img = _canvas_to_pil(data)
    if img is None:
        return None
    proc = preprocess_drawn_image(img, out_size=28, pad_ratio=0.2)
    proc = ImageOps.mirror(proc).rotate(90, expand=False)
    return proc

with gr.Blocks(title="EMNIST Handwrite") as demo:
    gr.Markdown("# EMNIST Handwrite\nRysuj cyfrę lub wgraj obraz. Otrzymasz Top-k z prawdopodobieństwami.")
    with gr.Tab("Rysuj"):
        with gr.Row():
            canvas = gr.Image(label="Panel rysowania", image_mode="L", type="numpy", height=280, width=280, sources=None)
            out_tbl = gr.Label(num_top_classes=5, label="Top-k")
            prev = gr.Image(label="Podgląd 28×28", image_mode="L", type="pil", height=112, width=112)
        thr = gr.Slider(0.0, 1.0, value=0.6, step=0.01, label="Próg niepewności")
        msg = gr.Markdown()
        btn = gr.Button("Rozpoznaj")
        btn_prev = gr.Button("Pokaż preprocess")
        btn.click(predict_from_canvas, inputs=[canvas, thr], outputs=[out_tbl, msg])
        btn_prev.click(preview_preprocess, inputs=canvas, outputs=prev)
    with gr.Tab("Upload"):
        up = gr.File(label="Wgraj obraz PNG/JPG", type="filepath")
        thr2 = gr.Slider(0.0, 1.0, value=0.6, step=0.01, label="Próg niepewności")
        out_tbl2 = gr.Label(num_top_classes=5, label="Top-k")
        msg2 = gr.Markdown()
        btn2 = gr.Button("Rozpoznaj")
        btn2.click(predict_from_upload, inputs=[up, thr2], outputs=[out_tbl2, msg2])

if __name__ == "__main__":
    demo.launch()
