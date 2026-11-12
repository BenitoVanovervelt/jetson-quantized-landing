"""Fine-tune YOLO26s of RT-DETR-l op de landingsplatform-dataset (in train-env).

Beide modellen vertrekken van COCO-gewichten (transfer learning): de lagen die
randen en vormen herkennen zijn al getraind, enkel de taak "landingsplatform"
moet bijgeleerd worden. Dat vraagt veel minder data dan van nul trainen.

Gebruik (vanuit de projectmap):
    train-env\\Scripts\\python training/train.py --model yolo26s
    train-env\\Scripts\\python training/train.py --model rtdetr-l
    train-env\\Scripts\\python training/train.py --model yolo26s --data coco8.yaml --epochs 1   # snelle test
    train-env\\Scripts\\python training/train.py --model rtdetr-l --resume   # onderbroken training hervatten

Resultaat: runs/<model>/weights/best.pt (+ grafieken en val-metrieken in dezelfde map).
"""
import argparse
from pathlib import Path

from ultralytics import RTDETR, YOLO

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA = PROJECT_DIR / "data" / "landingpad" / "data.yaml"

# Batchgrootte per model voor een GPU met 6 GB (RTX 3060 Laptop).
# RT-DETR-l is groter en de transformer gebruikt meer geheugen per beeld.
MODELS = {
    "yolo26s": {"weights": "yolo26s.pt", "cls": YOLO, "batch": 16},
    "rtdetr-l": {"weights": "rtdetr-l.pt", "cls": RTDETR, "batch": 4},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=list(MODELS))
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, help="standaard: zie MODELS")
    parser.add_argument("--name", help="naam van de run, standaard de modelnaam")
    parser.add_argument("--resume", action="store_true",
                        help="onderbroken training hervatten vanaf runs/<name>/weights/last.pt")
    args = parser.parse_args()

    spec = MODELS[args.model]
    if args.resume:
        # last.pt bevat gewichten, optimizer, epoch en alle trainingsinstellingen:
        # de training gaat verder bij de volgende epoch, met dezelfde instellingen.
        last = PROJECT_DIR / "runs" / (args.name or args.model) / "weights" / "last.pt"
        spec["cls"](str(last)).train(resume=True)
        return
    model = spec["cls"](spec["weights"])     # downloadt de COCO-gewichten de eerste keer
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=640,                 # zelfde als de camera: geen herschaling
        batch=args.batch or spec["batch"],
        patience=20,               # stop als val-mAP 20 epochs niet verbetert
        seed=0, deterministic=True,  # herhaalbaar
        workers=4,
        project=str(PROJECT_DIR / "runs"),
        name=args.name or args.model,
        exist_ok=False,
        # Camera kijkt recht naar beneden: een gespiegeld of op zijn kop staand beeld
        # is even realistisch. De H is symmetrisch, dus het label blijft correct.
        flipud=0.5, fliplr=0.5,
    )


if __name__ == "__main__":
    main()   # guard nodig op Windows: de dataloader start nieuwe processen
