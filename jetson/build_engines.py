"""Bouw TensorRT-engines in FP32, FP16 en INT8 en meet de inferentietijd (op de Jetson).

Elke rekenprecisie wordt apart geprobeerd. Mislukt er één (bekend risico: RT-DETR
in INT8), dan wordt de foutmelding bewaard en gaat het script door: een mislukte
export is een resultaat, geen reden om te stoppen.

Ultralytics doet intern .pt -> ONNX -> TensorRT. INT8 heeft kalibratiebeelden nodig:
TensorRT stuurt die door het model om per laag het bereik van de waarden te bepalen.

Gebruik (op de Jetson, vanuit de projectmap):
    # Exporttest week 1, met COCO-gewichten en de kleine voorbeeldset coco8:
    python3 jetson/build_engines.py --weights yolo26s.pt --name yolo26s_coco --calib-data coco8.yaml
    python3 jetson/build_engines.py --weights rtdetr-l.pt --name rtdetr-l_coco --calib-data coco8.yaml
    # Echte engines, met de eigen kalibratieset (los van de testset):
    python3 jetson/build_engines.py --weights models/yolo26s.pt --name yolo26s --calib-data data/landingpad/calib.yaml

Tip: bewaar ook de volledige uitvoer, TensorRT schrijft details naar de terminal:
    python3 jetson/build_engines.py ... 2>&1 | tee results/engines/build_yolo26s.log

Resultaat: engines/<name>_<fp32|fp16|int8>.engine en results/engines/summary.csv
"""
import argparse
import csv
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "jetson"))
from detector import IMAGE_SIZE, Detector, load_model  # noqa: E402

ENGINE_DIR = PROJECT_DIR / "engines"
RESULT_DIR = PROJECT_DIR / "results" / "engines"
PRECISIONS = {"fp32": {}, "fp16": {"half": True}, "int8": {"int8": True}}
SUMMARY_COLUMNS = ["timestamp", "name", "precision_mode", "status", "mem_free_mb", "build_s", "engine_mb",
                   "inference_ms_mean", "inference_ms_p95", "fps_inference_only", "error_file",
                   "jetpack", "tensorrt", "ultralytics", "torch", "power_mode"]


def command_output(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def versions():
    """Versies horen bij elk resultaat: een engine is gebonden aan TensorRT en de hardware."""
    import torch
    import ultralytics
    try:
        import tensorrt
        trt_version = tensorrt.__version__
    except ImportError:
        trt_version = "niet gevonden"
    release = Path("/etc/nv_tegra_release")
    return {
        "jetpack": release.read_text().split("\n")[0].strip() if release.exists() else platform.platform(),
        "tensorrt": trt_version,
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        # Vermogensmodus bepaalt de kloksnelheid: noteer hem bij elke meting.
        # In Docker bestaat nvpmodel niet: dan komt de modus via de omgevingsvariabele POWER_MODE.
        "power_mode": (command_output(["nvpmodel", "-q"]).split("\n")[0]
                       or os.environ.get("POWER_MODE", "onbekend")),
    }


def memory_mb():
    """Vrij en beschikbaar geheugen (MB). Op de Jetson delen CPU en GPU dit geheugen."""
    info = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":")
        info[key] = int(value.split()[0]) // 1024
    return info["MemFree"], info["MemAvailable"]


def free_gpu_memory():
    """Geef het geheugen van de vorige engine vrij voor de volgende gebouwd wordt."""
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build(weights, name, precision, calib_data, workspace_gb):
    model = load_model(weights)
    options = dict(PRECISIONS[precision])
    if precision == "int8":
        options["data"] = calib_data
    engine_path = model.export(format="engine", imgsz=IMAGE_SIZE, batch=1, dynamic=False,
                               device=0, workspace=workspace_gb, **options)
    target = ENGINE_DIR / f"{name}_{precision}.engine"
    shutil.move(engine_path, target)   # anders overschrijft de volgende rekenprecisie hem
    return target


def local_data_yaml(yaml_path):
    """Herschrijf 'path:' in een eigen dataset-yaml naar de map waar het yaml-bestand nu staat.

    data.yaml en calib.yaml worden op de pc gemaakt en bevatten dus een Windows-pad.
    Op de Jetson staat de dataset elders. Een naam zonder map (bv. coco8.yaml) laat
    Ultralytics zelf downloaden: die blijft ongewijzigd.
    """
    path = Path(yaml_path)
    if not path.exists():
        return yaml_path
    lines = path.read_text().splitlines()
    lines = [f"path: {path.resolve().parent.as_posix()}" if line.startswith("path:") else line for line in lines]
    local = RESULT_DIR / f"local_{path.name}"
    local.write_text("\n".join(lines) + "\n")
    return str(local)


def benchmark(engine_path, runs, image_dir):
    detector = Detector(engine_path)
    detector.warmup(20)
    images = []
    if image_dir:
        import cv2
        images = [cv2.imread(str(p)) for p in sorted(Path(image_dir).glob("*.jpg"))[:50]]
    if not images:
        images = [np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)]
    times = [detector.predict(images[i % len(images)])[1]["inference_ms"] for i in range(runs)]
    return float(np.mean(times)), float(np.percentile(times, 95))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True, help=".pt-bestand (bevat 'rtdetr' voor RT-DETR)")
    parser.add_argument("--name", required=True, help="bv. yolo26s of rtdetr-l_coco; bevat 'rtdetr' voor RT-DETR")
    parser.add_argument("--calib-data", required=True, help="yaml voor INT8-kalibratie (de 'val'-map wordt gebruikt)")
    parser.add_argument("--precisions", nargs="+", default=list(PRECISIONS), choices=list(PRECISIONS))
    parser.add_argument("--runs", type=int, default=200, help="aantal inferenties voor de tijdmeting")
    parser.add_argument("--bench-images", help="map met .jpg-beelden voor de tijdmeting (bv. de testset)")
    parser.add_argument("--workspace", type=float, default=2.0, help="GB werkgeheugen voor TensorRT bij het bouwen")
    args = parser.parse_args()

    if "rtdetr" in Path(args.weights).name.lower() and "rtdetr" not in args.name.lower():
        sys.exit("FOUT: --name moet 'rtdetr' bevatten voor een RT-DETR-model (zo wordt het later correct geladen).")

    ENGINE_DIR.mkdir(exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    args.calib_data = local_data_yaml(args.calib_data)
    info = versions()
    print("Versies:", info)

    summary_path = RESULT_DIR / "summary.csv"
    new_file = not summary_path.exists()
    with open(summary_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        if new_file:
            writer.writeheader()
        for precision in args.precisions:
            free_gpu_memory()
            mem_free, mem_available = memory_mb()
            print(f"\n=== {args.name} {precision.upper()} === (geheugen vrij {mem_free} MB, "
                  f"beschikbaar {mem_available} MB)")
            row = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "name": args.name,
                   "precision_mode": precision, "mem_free_mb": mem_free, **info}
            t_start = time.time()
            try:
                try:
                    engine = build(args.weights, args.name, precision, args.calib_data, args.workspace)
                except RuntimeError as error:
                    # Op de Jetson kan de GPU soms niet starten als het geheugen even vol zit
                    # (CUDA "out of memory" bij het opstarten). Eén nieuwe poging na opruimen;
                    # mislukt die ook, dan is het een echt resultaat.
                    if "out of memory" not in str(error):
                        raise
                    print("Geheugen vol bij het opstarten van de GPU: opruimen en opnieuw proberen")
                    free_gpu_memory()
                    time.sleep(10)
                    engine = build(args.weights, args.name, precision, args.calib_data, args.workspace)
                row["build_s"] = round(time.time() - t_start)
                row["engine_mb"] = round(engine.stat().st_size / 1e6, 1)
                mean_ms, p95_ms = benchmark(engine, args.runs, args.bench_images)
                row.update(status="ok", inference_ms_mean=round(mean_ms, 2), inference_ms_p95=round(p95_ms, 2),
                           fps_inference_only=round(1000 / mean_ms, 1))
                print(f"OK: {engine.name}, inferentie {mean_ms:.1f} ms ({1000 / mean_ms:.0f} FPS, enkel het model)")
            except Exception:
                error_file = RESULT_DIR / f"{args.name}_{precision}_error.txt"
                error_file.write_text(traceback.format_exc())
                row.update(status="mislukt", build_s=round(time.time() - t_start), error_file=error_file.name)
                print(f"MISLUKT: foutmelding bewaard in {error_file}")
            writer.writerow(row)
            f.flush()
    print(f"\nSamenvatting: {summary_path}")


if __name__ == "__main__":
    main()
