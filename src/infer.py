import os
import json
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms
from src.model import SmallCNN
from src.preprocess import preprocess_drawn_image

class Predictor:
    def __init__(self, ckpt_path: str = os.path.join("data","models","best.pt"), align_emnist: bool = True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model" in state:
            model_sd = state["model"]
            classes = state.get("classes")
        else:
            model_sd = state
            classes = None
        self.classes = classes if classes is not None else [str(i) for i in range(10)]
        self.model = SmallCNN(num_classes=len(self.classes)).to(self.device)
        self.model.load_state_dict(model_sd, strict=True)
        self.model.eval()
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.align_emnist = align_emnist
        self.temperature = 1.0
        calib_path = os.path.join("data","models","calib.json")
        if os.path.exists(calib_path):
            with open(calib_path, "r", encoding="utf-8") as f:
                self.temperature = float(json.load(f).get("temperature", 1.0))

    def _align(self, img):
        return ImageOps.mirror(img).rotate(90, expand=False) if self.align_emnist else img

    def _prep_pil(self, img: Image.Image) -> torch.Tensor:
        proc = preprocess_drawn_image(img, out_size=28, pad_ratio=0.2)
        proc = self._align(proc)
        x = self.tf(proc)
        return x.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict_pil(self, img: Image.Image, topk: int = 5):
        x = self._prep_pil(img)
        logits = self.model(x)
        probs = F.softmax(logits, dim=1).squeeze(0)
        k = min(topk, probs.size(0))
        vals, idxs = torch.topk(probs, k=k)
        labels = [self.classes[i] for i in idxs.cpu().tolist()]
        scores = [float(v) for v in vals.cpu().tolist()]
        return list(zip(labels, scores))
