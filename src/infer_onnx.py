import os
import json
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps
from torchvision import transforms
from src.preprocess import preprocess_drawn_image

class OnnxPredictor:
    def __init__(self, model_path: str = os.path.join("data","models","model.onnx")):
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        meta_classes = os.path.join(os.path.dirname(model_path), "classes.json")
        if os.path.exists(meta_classes):
            with open(meta_classes, "r", encoding="utf-8") as f:
                self.classes = json.load(f).get("classes", [str(i) for i in range(10)])
        else:
            self.classes = [str(i) for i in range(10)]
        calib_path = os.path.join(os.path.dirname(model_path), "calib.json")
        self.temperature = 1.0
        if os.path.exists(calib_path):
            with open(calib_path, "r", encoding="utf-8") as f:
                self.temperature = float(json.load(f).get("temperature", 1.0))
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def _prep(self, img: Image.Image):
        img = ImageOps.mirror(img).rotate(90, expand=False)
        x = self.tf(img).unsqueeze(0).numpy()
        return x

    def predict_pil(self, img: Image.Image, topk: int = 5):
        proc = preprocess_drawn_image(img, out_size=28, pad_ratio=0.2)
        x = self._prep(proc)
        logits = self.sess.run(["logits"], {"input": x})[0]
        logits = logits / self.temperature
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        p = probs[0]
        k = min(topk, p.shape[0])
        idxs = np.argsort(-p)[:k].tolist()
        labels = [self.classes[i] for i in idxs]
        scores = [float(p[i]) for i in idxs]
        return list(zip(labels, scores))