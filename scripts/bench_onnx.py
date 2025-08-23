import os
import time
import onnxruntime as ort
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

def load_inputs(n=200, batch=1):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    val = datasets.EMNIST(root="data", split="balanced", train=False, download=True, transform=tf)
    idx = list(range(min(len(val), n)))
    loader = DataLoader(Subset(val, idx), batch_size=batch, shuffle=False, num_workers=0)
    xs = []
    for x, _ in loader:
        xs.append(x.numpy())
    return xs

def bench(path, xs, warmup=20, runs=100):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    t = []
    for _ in range(warmup):
        for x in xs:
            sess.run(["logits"], {"input": x})
    t0 = time.perf_counter()
    c = 0
    for _ in range(runs):
        for x in xs:
            sess.run(["logits"], {"input": x})
            c += 1
    dt = time.perf_counter() - t0
    avg = dt / c
    return avg

def main():
    xs = load_inputs(n=200, batch=1)
    fp32 = os.path.join("data","models","model.onnx")
    int8a = os.path.join("data","models","model.int8.onnx")
    int8b = os.path.join("data","models","model.int8.dynamic.onnx")
    if os.path.exists(fp32):
        a = bench(fp32, xs)
        print(f"fp32_avg_s: {a:.6f}")
    if os.path.exists(int8a):
        b = bench(int8a, xs)
        print(f"int8_static_avg_s: {b:.6f}")
    if os.path.exists(int8b):
        c = bench(int8b, xs)
        print(f"int8_dynamic_avg_s: {c:.6f}")

if __name__ == "__main__":
    main()
