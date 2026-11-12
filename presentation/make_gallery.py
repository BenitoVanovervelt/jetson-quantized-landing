"""Galerij 'slecht zicht' voor de presentatie: één testbeeld in elke conditie, met de detectie van beide modellen.

Het beeld staat op dezelfde positie in alle condities (data/hard/<conditie>/images/test/<bestand>).
Beide modellen (PyTorch op de pc, = FP32) voorspellen erop; de boxen met confidence >= 0,5 (de drempel
van de landingslogica) komen in presentation/js/gallery_data.js. De recall per rekenprecisie op de
website komt niet hiervandaan maar uit de Jetson-metingen (build_data.py).

Gebruik (vanuit de projectmap, train-env):
    train-env\\Scripts\\python.exe presentation\\make_gallery.py
    train-env\\Scripts\\python.exe presentation\\make_gallery.py --file test_00006.jpg
"""
import argparse
import csv
import json
import shutil
from pathlib import Path

from ultralytics import RTDETR, YOLO

PROJECT_DIR = Path(__file__).resolve().parent.parent
HARD = PROJECT_DIR / "data" / "hard"
OUT_IMG = PROJECT_DIR / "presentation" / "img" / "gallery"
OUT_JS = PROJECT_DIR / "presentation" / "js" / "gallery_data.js"
CONDITIONS = ["clean", "fog_05", "fog_10", "dust_05", "dust_10", "dark", "blur", "noise"]
THRESHOLD = 0.5   # zelfde drempel als de landingslogica


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="test_00006.jpg", help="testbeeld (zelfde naam in elke conditie)")
    args = parser.parse_args()

    models = {"yolo26s": YOLO(str(PROJECT_DIR / "models" / "yolo26s.pt")),
              "rtdetr-l": RTDETR(str(PROJECT_DIR / "models" / "rtdetr-l.pt"))}
    OUT_IMG.mkdir(parents=True, exist_ok=True)
    gallery = {"file": args.file, "threshold": THRESHOLD, "conditions": {}}
    for cond in CONDITIONS:
        src = HARD / cond / "images" / "test" / args.file
        with open(HARD / cond / "metadata.csv", newline="", encoding="utf-8") as f:
            meta = next(r for r in csv.DictReader(f) if r["file"].endswith(args.file))
        shutil.copy(src, OUT_IMG / f"{cond}.jpg")
        entry = {"theta": float(meta["theta_deg"]), "distance": float(meta["distance_m"]),
                 "label": [float(meta[k]) for k in ("box_x1", "box_y1", "box_x2", "box_y2")] if meta["box_x1"] else None,
                 "detections": {}}
        for name, model in models.items():
            result = model.predict(str(src), conf=THRESHOLD, verbose=False)[0]
            entry["detections"][name] = [
                {"box": [round(v, 1) for v in box.xyxy[0].tolist()], "conf": round(float(box.conf[0]), 3)}
                for box in result.boxes]
        gallery["conditions"][cond] = entry
        print(cond, {k: [d["conf"] for d in v] for k, v in entry["detections"].items()})
    OUT_JS.parent.mkdir(parents=True, exist_ok=True)
    OUT_JS.write_text("// Gegenereerd door presentation/make_gallery.py, niet met de hand aanpassen.\n"
                      f"window.GALLERY = {json.dumps(gallery, indent=1)};\n", encoding="utf-8")
    print(f"Klaar: {OUT_JS}")


if __name__ == "__main__":
    main()
