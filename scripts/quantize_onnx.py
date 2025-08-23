import os
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from onnxruntime.quantization import quantize_static, quantize_dynamic, CalibrationDataReader, QuantType

class EmnistCalibReader(CalibrationDataReader):
    def __init__(self, batch=128, limit=1024):
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        val = datasets.EMNIST(root="data", split="balanced", train=False, download=True, transform=tf)
        idx = list(range(min(len(val), limit)))
        self.loader = DataLoader(Subset(val, idx), batch_size=batch, shuffle=False, num_workers=0)
        self.it = iter(self.loader)
    def get_next(self):
        try:
            x, _ = next(self.it)
            return {"input": x.numpy()}
        except StopIteration:
            return None
    def rewind(self):
        self.it = iter(self.loader)

def main():
    model_fp32 = os.path.join("data","models","model.onnx")
    model_int8 = os.path.join("data","models","model.int8.onnx")
    reader = EmnistCalibReader(batch=128, limit=1024)
    try:
        quantize_static(model_input=model_fp32, model_output=model_int8, calibration_data_reader=reader, activation_type=QuantType.QInt8, weight_type=QuantType.QInt8, per_channel=True)
        print(f"saved: {model_int8}")
    except Exception as e:
        model_dyn = os.path.join("data","models","model.int8.dynamic.onnx")
        quantize_dynamic(model_input=model_fp32, model_output=model_dyn, weight_type=QuantType.QInt8)
        print(f"static quantization failed: {e}")
        print(f"saved dynamic: {model_dyn}")

if __name__ == "__main__":
    main()
