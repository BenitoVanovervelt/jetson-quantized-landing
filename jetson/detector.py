"""Eén detector-klasse voor alle modelbestanden: .pt (pc), .onnx en .engine (Jetson).

Ultralytics laadt een TensorRT-engine met dezelfde API als een PyTorch-model.
Daardoor gebruiken de server, de benchmark en de evaluatie exact dezelfde code,
en verschilt tussen de 6 engines enkel het bestand.
"""
from pathlib import Path

IMAGE_SIZE = 640


def load_model(model_path):
    """RT-DETR heeft in Ultralytics een eigen klasse; herkend aan de bestandsnaam."""
    from ultralytics import RTDETR, YOLO   # pas hier importeren: protocol.py heeft geen ultralytics nodig

    name = Path(model_path).name.lower()
    if "rtdetr" in name:
        return RTDETR(str(model_path))   # RTDETR kent enkel de taak detect
    # task expliciet: bij .onnx/.engine kan Ultralytics de taak niet altijd uit het bestand halen.
    return YOLO(str(model_path), task="detect")


class Detector:
    def __init__(self, model_path, conf=0.25, device=0):
        self.model_path = str(model_path)
        self.conf = conf
        self.device = device
        self.model = load_model(model_path)

    def warmup(self, runs=10):
        """De eerste inferenties zijn trager (geheugen reserveren, kernels laden): niet meten."""
        import numpy as np

        dummy = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        for _ in range(runs):
            self.predict(dummy)

    def predict(self, image_bgr):
        """Geef (detecties, tijden in ms) terug voor één BGR-beeld.

        De tijden komen van Ultralytics zelf: preprocess, inference, postprocess.
        "inference" is de tijd van het model (de engine) alleen.
        """
        result = self.model.predict(image_bgr, imgsz=IMAGE_SIZE, conf=self.conf,
                                    device=self.device, verbose=False)[0]
        detections = [
            {"box": [round(v, 1) for v in box.tolist()], "conf": round(float(conf), 4)}
            for box, conf in zip(result.boxes.xyxy.cpu(), result.boxes.conf.cpu())
        ]
        timings = {
            "preprocess_ms": result.speed["preprocess"],
            "inference_ms": result.speed["inference"],
            "postprocess_ms": result.speed["postprocess"],
        }
        return detections, timings
