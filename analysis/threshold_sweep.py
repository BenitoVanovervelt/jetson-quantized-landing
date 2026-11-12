"""Recall en vals alarm bij de drempel van de landingslogica (0,5), en voor drempels 0,05..0,95.

Waarom: Ultralytics rapporteert precision en recall bij de drempel met de beste F1-score,
niet bij de vaste drempel 0,5 die de landingslogica gebruikt. Dit script meet per beeld:
  matched_conf    hoogste confidence van een box die het platform raakt (IoU >= 0,5 met het label)
  unmatched_conf  hoogste confidence van een box die er naast ligt (IoU < 0,5): een valse detectie
Daaruit volgt voor elke drempel t:
  recall(t)        aandeel beelden met platform waar matched_conf >= t
  vals_alarm(t)    aandeel beelden zonder platform met een box >= t   (gevaar: foute_landing)
  valse_box(t)     aandeel beelden met platform met een valse box >= t
Zo is te zien of 0,5 een goede keuze is, en hoe de moeilijkere omstandigheden doorwerken
op wat de landingslogica echt ziet.

Draait op de pc met de PyTorch-modellen (models/*.pt, ± FP32), in train-env:
    train-env\\Scripts\\python analysis/threshold_sweep.py
    train-env\\Scripts\\python analysis/threshold_sweep.py --conditions normal fog_05 --models yolo26s

Resultaat in results/threshold_sweep/:
    per_image.csv     één rij per beeld, model en conditie
    sweep.csv         recall, vals alarm en valse box per drempel
    examples/         voorbeelden van valse detecties (beeld met box), voor het verslag
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import RTDETR, YOLO

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_DIR / "results" / "threshold_sweep"
CONDITION_DIRS = {"normal": PROJECT_DIR / "data" / "landingpad"}
CONDITION_DIRS.update({p.name: p for p in sorted((PROJECT_DIR / "data" / "hard").glob("*")) if p.is_dir()})
MODELS = {"yolo26s": "models/yolo26s.pt", "rtdetr-l": "models/rtdetr-l.pt"}
IMAGE_SIZE = 640
MIN_CONF = 0.05                         # zo laag mogelijk meten; de drempel wordt achteraf toegepast
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)
LANDING_THRESHOLD = 0.5
EXAMPLES_PER_CASE = 6


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def read_label(path):
    """YOLO-label -> box in pixels, of None (leeg = geen platform in beeld)."""
    text = path.read_text().strip() if path.exists() else ""
    if not text:
        return None
    _, x, y, w, h = map(float, text.split()[:5])
    return ((x - w / 2) * IMAGE_SIZE, (y - h / 2) * IMAGE_SIZE, (x + w / 2) * IMAGE_SIZE, (y + h / 2) * IMAGE_SIZE)


def test_rows(data_dir):
    with open(data_dir / "metadata.csv", newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["split"] == "test"]


def save_example(image_path, label, boxes, path, caption):
    img = cv2.imread(str(image_path))
    if label:
        cv2.rectangle(img, tuple(map(int, label[:2])), tuple(map(int, label[2:])), (0, 200, 0), 2)
    for box, conf in boxes:
        cv2.rectangle(img, tuple(map(int, box[:2])), tuple(map(int, box[2:])), (0, 0, 255), 2)
        cv2.putText(img, f"{conf:.2f}", (int(box[0]), max(12, int(box[1]) - 4)), 0, 0.5, (0, 0, 255), 1)
    cv2.putText(img, caption, (5, 20), 0, 0.55, (0, 0, 255), 2)
    cv2.imwrite(str(path), img)


def evaluate(model_name, condition, data_dir, writer, examples):
    model = (RTDETR if "rtdetr" in model_name else YOLO)(str(PROJECT_DIR / MODELS[model_name]))
    rows = test_rows(data_dir)
    for row in rows:
        image_path = data_dir / row["file"]
        label = read_label(data_dir / "labels" / "test" / (Path(row["file"]).stem + ".txt"))
        if row["negative"] == "0" and label is None:
            continue    # platform toch buiten beeld (rand van het beeld): niet meetellen
        result = model.predict(str(image_path), imgsz=IMAGE_SIZE, conf=MIN_CONF, verbose=False)[0]
        boxes = [(tuple(b), float(c)) for b, c in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist())]
        matched = [c for b, c in boxes if label and iou(b, label) >= 0.5]
        unmatched = [(b, c) for b, c in boxes if not (label and iou(b, label) >= 0.5)]
        record = {"model": model_name, "condition": condition, "file": row["file"],
                  "negative": row["negative"], "theta_deg": row["theta_deg"], "distance_m": row["distance_m"],
                  "matched_conf": round(max(matched, default=0.0), 4),
                  "unmatched_conf": round(max((c for _, c in unmatched), default=0.0), 4)}
        writer.writerow(record)
        # Voorbeelden van valse detecties boven de landingsdrempel (wat zag het model?).
        false_boxes = [(b, c) for b, c in unmatched if c >= LANDING_THRESHOLD]
        key = (model_name, condition)
        if false_boxes and examples.get(key, 0) < EXAMPLES_PER_CASE:
            examples[key] = examples.get(key, 0) + 1
            kind = "geen_platform" if row["negative"] == "1" else f"d{float(row['distance_m']):g}"
            save_example(image_path, label, false_boxes,
                         OUT_DIR / "examples" / f"{model_name}_{condition}_{kind}_{Path(row['file']).stem}.jpg",
                         f"{model_name} {condition}: rood = valse detectie, groen = platform")


def sweep(per_image_path):
    with open(per_image_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for key in sorted({(r["model"], r["condition"]) for r in rows}):
        sel = [r for r in rows if (r["model"], r["condition"]) == key]
        pos = [r for r in sel if r["negative"] == "0"]
        neg = [r for r in sel if r["negative"] == "1"]
        for t in THRESHOLDS:
            out.append({"model": key[0], "condition": key[1], "threshold": t,
                        "positives": len(pos), "negatives": len(neg),
                        "recall": round(np.mean([float(r["matched_conf"]) >= t for r in pos]), 4),
                        "false_alarm": round(np.mean([float(r["unmatched_conf"]) >= t for r in neg]), 4),
                        "false_box_on_positive": round(np.mean([float(r["unmatched_conf"]) >= t for r in pos]), 4)})
    with open(OUT_DIR / "sweep.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--conditions", nargs="+", default=list(CONDITION_DIRS), choices=list(CONDITION_DIRS))
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    args = parser.parse_args()

    (OUT_DIR / "examples").mkdir(parents=True, exist_ok=True)
    per_image_path = OUT_DIR / "per_image.csv"
    fields = ["model", "condition", "file", "negative", "theta_deg", "distance_m", "matched_conf", "unmatched_conf"]
    examples = {}
    with open(per_image_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for model_name in args.models:
            for condition in args.conditions:
                print(f"{model_name} {condition} ...", flush=True)
                evaluate(model_name, condition, CONDITION_DIRS[condition], writer, examples)
                f.flush()

    rows = sweep(per_image_path)
    print(f"\nBij drempel {LANDING_THRESHOLD}:")
    print(f"{'model':9s} {'conditie':8s} {'recall':>7s} {'vals alarm':>11s} {'valse box':>10s}")
    for r in rows:
        if r["threshold"] == LANDING_THRESHOLD:
            print(f"{r['model']:9s} {r['condition']:8s} {r['recall']:7.3f} {r['false_alarm']:11.3f} "
                  f"{r['false_box_on_positive']:10.3f}")
    print(f"\nResultaat: {OUT_DIR}")


if __name__ == "__main__":
    main()
