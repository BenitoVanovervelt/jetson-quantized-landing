"""Offline evaluatie van een engine op de testset: precision, recall, mAP50, inferentietijd,
in totaal en per naderingshoek θ en afstand d (deelvragen 5 en 7).

Werkwijze: per groep (bv. θ = 30°) een lijst met beeldpaden maken en daarop de
standaard-validatie van Ultralytics draaien. Zo is elke groep exact op dezelfde
manier berekend als het totaal.

Let op: precision en recall van Ultralytics gelden bij de confidence-drempel met
de beste F1-score. De landingslogica gebruikt een vaste drempel; die wordt apart
gekozen op de val-set.

Negatieve beelden (zonder platform) zitten in het totaal: een detectie daarop
verlaagt de precision. Daarnaast wordt het aandeel negatieve beelden met een
detectie apart gerapporteerd (vals alarm, het gevaar voor foute_landing).

Gebruik (op de Jetson, vanuit de projectmap):
    python3 jetson/evaluate.py --model engines/yolo26s_fp16.engine --precision fp16

Resultaat: één rij per groep in results/offline/offline_metrics.csv

Moeilijkere testsets (data_gen/generate_hard_testset.py): zelfde beelden en labels
onder mist, stof, weinig licht, ... Enkel totaal en per afstand, om tijd te sparen:
    python3 jetson/evaluate.py --model engines/yolo26s_fp16.engine --precision fp16         --data data/hard/fog_10 --condition fog_10 --groups all distance         --out results/robustness/robustness_metrics.csv
"""
import argparse
import csv
import sys
import tempfile
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "jetson"))
from detector import IMAGE_SIZE, Detector, load_model  # noqa: E402

DATA_DIR = PROJECT_DIR / "data" / "landingpad"
RESULT_PATH = PROJECT_DIR / "results" / "offline" / "offline_metrics.csv"
ALL_GROUPS = ["all", "theta", "distance", "theta_x_distance"]
COLUMNS = ["timestamp", "model_file", "model", "precision_mode", "group", "value", "images",
           "precision", "recall", "map50", "map50_95", "inference_ms", "false_alarm_rate"]


def set_data_dir(path):
    """Andere testset gebruiken (zelfde opbouw: metadata.csv, images/test, labels/test)."""
    global DATA_DIR
    DATA_DIR = Path(path).resolve()


def read_test_metadata():
    with open(DATA_DIR / "metadata.csv", newline="", encoding="utf-8") as f:
        return [row for row in csv.DictReader(f) if row["split"] == "test"]


def groups(rows):
    """(groepnaam, waarde, rijen): totaal, per θ, per d en per combinatie θ x d."""
    positives = [r for r in rows if r["negative"] == "0"]
    yield "all", "", rows
    for key, label in (("theta_deg", "theta"), ("distance_m", "distance")):
        for value in sorted({float(r[key]) for r in positives}):
            yield label, value, [r for r in positives if float(r[key]) == value]
    for theta in sorted({float(r["theta_deg"]) for r in positives}):
        for distance in sorted({float(r["distance_m"]) for r in positives}):
            subset = [r for r in positives if float(r["theta_deg"]) == theta and float(r["distance_m"]) == distance]
            yield "theta_x_distance", f"{theta:g}_{distance:g}", subset


def validate(model, rows, tmp_dir, tag):
    list_file = Path(tmp_dir) / f"{tag}.txt"
    list_file.write_text("\n".join(str(DATA_DIR / r["file"]) for r in rows))
    yaml_file = Path(tmp_dir) / f"{tag}.yaml"
    yaml_file.write_text(f"path: {DATA_DIR.as_posix()}\ntrain: {list_file.as_posix()}\n"
                         f"val: {list_file.as_posix()}\nnames:\n  0: landingsplatform\n")
    metrics = model.val(data=str(yaml_file), imgsz=IMAGE_SIZE, batch=1, device=0,
                        plots=False, verbose=False, project=tmp_dir, name=tag)
    return metrics.box.mp, metrics.box.mr, metrics.box.map50, metrics.box.map, metrics.speed["inference"]


def false_alarm_rate(model_path, rows, conf):
    import cv2
    negatives = [r for r in rows if r["negative"] == "1"]
    if not negatives:
        return ""
    detector = Detector(model_path, conf=conf)
    hits = sum(bool(detector.predict(cv2.imread(str(DATA_DIR / r["file"])))[0]) for r in negatives)
    return round(hits / len(negatives), 4)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help=".engine (of .pt om op de pc te testen)")
    parser.add_argument("--precision", required=True, help="fp32, fp16, int8 of pt (voor de log)")
    parser.add_argument("--conf", type=float, default=0.5, help="drempel voor het vals-alarmpercentage")
    parser.add_argument("--data", default=str(DATA_DIR), help="map met de testset (standaard de gewone)")
    parser.add_argument("--condition", default="", help="naam van de conditie (extra kolom), bv. fog_10")
    parser.add_argument("--groups", nargs="+", default=ALL_GROUPS, choices=ALL_GROUPS)
    parser.add_argument("--out", default=str(RESULT_PATH), help="CSV waar de rijen bij komen")
    args = parser.parse_args()

    set_data_dir(args.data)
    result_path = Path(args.out)
    # De kolom 'condition' enkel als die gevraagd is: zo blijft het bestaande offline-CSV ongewijzigd.
    columns = COLUMNS + (["condition"] if args.condition else [])
    rows = read_test_metadata()
    model = load_model(args.model)
    model_name = Path(args.model).stem.rsplit("_", 1)[0]
    result_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not result_path.exists()

    with tempfile.TemporaryDirectory() as tmp, open(result_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if new_file:
            writer.writeheader()
        for group, value, subset in groups(rows):
            if not subset or group not in args.groups:
                continue
            p, r, map50, map50_95, inference_ms = validate(model, subset, tmp, f"{group}_{value}")
            row = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "model_file": Path(args.model).name,
                   "model": model_name, "precision_mode": args.precision, "group": group, "value": value,
                   "images": len(subset), "precision": round(p, 4), "recall": round(r, 4),
                   "map50": round(map50, 4), "map50_95": round(map50_95, 4),
                   "inference_ms": round(inference_ms, 2)}
            if args.condition:
                row["condition"] = args.condition
            if group == "all":
                row["false_alarm_rate"] = false_alarm_rate(args.model, rows, args.conf)
            writer.writerow(row)
            f.flush()
            print(f"{group:>17} {str(value):>8}  n={len(subset):4d}  P={p:.3f}  R={r:.3f}  "
                  f"mAP50={map50:.3f}  mAP50-95={map50_95:.3f}")
    print(f"Resultaat: {result_path}")


if __name__ == "__main__":
    main()
