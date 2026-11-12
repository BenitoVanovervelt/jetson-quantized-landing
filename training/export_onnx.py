"""Exporteer een getraind model naar ONNX (in train-env, op de pc).

Waarom ONNX op de pc: het is een draagbaar formaat dat je naar de Jetson kopieert.
Daar bouwt TensorRT er de engine van (die is gebonden aan de hardware, dus niet op
de pc bouwen). Ultralytics doet die ONNX-stap ook zelf bij een export naar
"engine"; dit bestand is nodig voor plan B met trtexec.

Gebruik:
    train-env\\Scripts\\python training/export_onnx.py --weights runs/yolo26s/weights/best.pt --name yolo26s
    train-env\\Scripts\\python training/export_onnx.py --weights yolo26s.pt --name yolo26s_coco   # COCO-test

Resultaat: models/<name>.onnx
"""
import argparse
import shutil
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "jetson"))
from detector import load_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--name", required=True, help="bevat 'rtdetr' voor een RT-DETR-model")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    model = load_model(args.weights)
    # Vaste invoer 640 x 640 en batch 1: zo ziet de engine er later ook uit.
    onnx_path = model.export(format="onnx", imgsz=640, opset=args.opset, dynamic=False,
                             simplify=True, batch=1)
    target = PROJECT_DIR / "models" / f"{args.name}.onnx"
    target.parent.mkdir(exist_ok=True)
    shutil.move(onnx_path, target)
    print(f"ONNX opgeslagen: {target}")


if __name__ == "__main__":
    main()
