"""Zet de meetresultaten om naar presentation/js/data.js en kopieert de beelden voor de website.

Zo komt elk getal op de presentatiewebsite rechtstreeks uit de metingen in results/; niets is met de
hand overgetypt. Opnieuw uitvoeren na een nieuwe meting volstaat.

Leest:
  results/summary.csv                               per engine: offline, ketentest, live (make_figures.py)
  results/chain/chain_<model>_<prec>_*.csv          tijd per stap in de keten (nieuwste bestand)
  results/robustness_jetson/robustness_metrics.csv  moeilijkere testsets op de Jetson
  data/landingpad/metadata.csv                      labels van de voorbeeldbeelden per afstand

Gebruik (vanuit de projectmap, gewone Python, geen extra pakketten):
    python presentation/build_data.py
Voor de donkere grafieken eerst: train-env\\Scripts\\python.exe analysis\\make_figures.py --dark
"""
import csv
import json
import shutil
from pathlib import Path
from statistics import mean

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_DIR / "results"
OUT = PROJECT_DIR / "presentation"
MODELS = ["yolo26s", "rtdetr-l"]
PRECISIONS = ["fp32", "fp16", "int8"]

# Voorbeeldbeelden per afstand, recht boven het platform (θ = 0), uit de testset.
DISTANCE_EXAMPLES = {5: "test_00002.jpg", 10: "test_00001.jpg", 20: "test_00011.jpg", 30: "test_00004.jpg"}

# Vaste beelden: (bron, doel in presentation/img/).
STATIC_IMAGES = [
    (RESULTS / "label_check" / "train_grid.jpg", "train_grid.jpg"),
    (RESULTS / "live" / "close_range_view.jpg", "close_range_view.jpg"),
    (RESULTS / "figures" / "rtdetr_fog_false_detections.jpg", "rtdetr_fog_false_detections.jpg"),
    (RESULTS / "figures_dark" / "live_confidence.png", "fig_live_confidence.png"),
    (RESULTS / "figures_dark" / "threshold_sweep.png", "fig_threshold_sweep.png"),
    (RESULTS / "figures_dark" / "offline_map50_95_heatmap.png", "fig_offline_heatmap.png"),
    (RESULTS / "figures_dark" / "live_outcomes.png", "fig_live_outcomes.png"),
]


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def latest(pattern):
    paths = sorted(RESULTS.glob(pattern))
    return paths[-1] if paths else None


def engines_summary():
    rows = {}
    for r in read_csv(RESULTS / "summary.csv"):
        rows[f"{r['model']}_{r['precision_mode']}"] = {k: (float(v) if v not in ("", None) else None)
                                                      for k, v in r.items() if k not in ("model", "precision_mode")}
    return rows


def chain_parts():
    """Gemiddelde tijd per stap (ms). 'rest_pc' = JPEG maken en beslissen op de pc (frame min de rest)."""
    parts = {}
    for m in MODELS:
        for p in PRECISIONS:
            path = latest(f"chain/chain_{m}_{p}_*.csv")
            rows = read_csv(path)
            avg = {k: mean(float(r[k]) for r in rows) for k in ("frame_ms", "capture_ms", "inference_ms", "server_ms", "network_ms")}
            parts[f"{m}_{p}"] = {
                "capture": round(avg["capture_ms"], 1),
                "network": round(avg["network_ms"], 1),
                "inference": round(avg["inference_ms"], 1),
                "server_rest": round(avg["server_ms"] - avg["inference_ms"], 1),
                "rest_pc": round(avg["frame_ms"] - avg["capture_ms"] - avg["network_ms"] - avg["server_ms"], 1),
                "frame": round(avg["frame_ms"], 1),
                "source": path.name,
            }
    return parts


def robustness():
    out = {}
    for r in read_csv(RESULTS / "robustness_jetson" / "robustness_metrics.csv"):
        if r["group"] != "all":
            continue
        out.setdefault(r["condition"], {})[f"{r['model']}_{r['precision_mode']}"] = {
            "recall": float(r["recall"]), "precision": float(r["precision"]),
            "false_alarm": float(r["false_alarm_rate"]) if r["false_alarm_rate"] else None}
    return out


def distance_examples():
    meta = {Path(r["file"]).name: r for r in read_csv(PROJECT_DIR / "data" / "landingpad" / "metadata.csv")}
    (OUT / "img" / "distance").mkdir(parents=True, exist_ok=True)
    out = []
    for d, name in DISTANCE_EXAMPLES.items():
        r = meta[name]
        shutil.copy(PROJECT_DIR / "data" / "landingpad" / "images" / "test" / name, OUT / "img" / "distance" / f"d{d}.jpg")
        out.append({"distance": d, "img": f"img/distance/d{d}.jpg",
                    "box": [float(r[k]) for k in ("box_x1", "box_y1", "box_x2", "box_y2")],
                    "width_px": round(float(r["box_w_px"]))})
    return out


def main():
    (OUT / "img").mkdir(parents=True, exist_ok=True)
    for src, dst in STATIC_IMAGES:
        if src.exists():
            shutil.copy(src, OUT / "img" / dst)
        else:
            print(f"Ontbreekt (overgeslagen): {src}")
    data = {"engines": engines_summary(), "chain": chain_parts(), "robustness": robustness(),
            "distance_examples": distance_examples()}
    (OUT / "js").mkdir(exist_ok=True)
    (OUT / "js" / "data.js").write_text("// Gegenereerd door presentation/build_data.py, niet met de hand aanpassen.\n"
                                       f"window.DATA = {json.dumps(data, indent=1)};\n", encoding="utf-8")
    print(f"Klaar: {OUT / 'js' / 'data.js'}")
    for k, v in data["chain"].items():
        print(f"  {k:15s} frame {v['frame']:5.1f} ms = ophalen {v['capture']} + netwerk {v['network']} + "
              f"inferentie {v['inference']} + rest Jetson {v['server_rest']} + rest pc {v['rest_pc']}")


if __name__ == "__main__":
    main()
