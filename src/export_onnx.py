import os
import json
import torch
from src.model import SmallCNN

def main():
    ckpt = os.path.join("data","models","best.pt")
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        sd = state["model"]
        classes = state.get("classes", [str(i) for i in range(10)])
    else:
        sd = state
        classes = [str(i) for i in range(10)]
    model = SmallCNN(num_classes=len(classes))
    model.load_state_dict(sd, strict=True)
    model.eval()
    dummy = torch.randn(1, 1, 28, 28)
    out_dir = os.path.join("data","models")
    os.makedirs(out_dir, exist_ok=True)
    onnx_path = os.path.join(out_dir, "model.onnx")
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=13,
        do_constant_folding=True,
    )
    with open(os.path.join(out_dir, "classes.json"), "w", encoding="utf-8") as f:
        json.dump({"classes": classes}, f)
    print(f"saved: {onnx_path}")

if __name__ == "__main__":
    main()
