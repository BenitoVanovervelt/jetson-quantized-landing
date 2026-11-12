"""Ketentest: FPS van beeld tot beslissing, inclusief netwerk (deelvraag 6).

Per frame:  beeld ophalen -> JPEG -> TCP naar server -> detectie terug -> beslissing.
De tijd van die hele lus is frame_ms; FPS = 1000 / frame_ms.

Bronnen van beelden:
  --source airsim   live van bottom_cam (Unreal moet op Play staan)
  --source folder   beelden uit een map, bv. de testset (zonder simulator)

Gebruik (vanuit de projectmap, in airsim-env):
    python -m landing.chain_test --host 192.168.55.1 --model yolo26s --precision fp16 --source airsim
    python -m landing.chain_test --host 127.0.0.1 --model yolo26s --precision pt --source folder --folder results

Resultaat: één CSV-rij per frame in results/chain/, en een samenvatting in de terminal.
"""
import argparse
import itertools
import time
from pathlib import Path

import cv2
import numpy as np

from landing.csv_log import CsvLogger
from landing.decision import LandingDecider
from landing.inference_client import InferenceClient, best_detection

PROJECT_DIR = Path(__file__).resolve().parent.parent
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def folder_frames(folder):
    paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise SystemExit(f"FOUT: geen beelden gevonden in {folder}")
    for path in itertools.cycle(paths):
        yield cv2.imread(str(path))


def airsim_frames(altitude_m):
    from sim.airsim_utils import connect, get_bottom_image, unlock_frame_rate

    client = connect()
    unlock_frame_rate(client)
    # Boven het platform hangen, zodat elk beeld een echte detectie bevat (zoals tijdens een landing).
    pad = client.simGetObjectPose("LandingPad").position
    print(f"Opstijgen naar {altitude_m:.0f} m boven het platform...")
    client.takeoffAsync().join()
    client.moveToPositionAsync(pad.x_val, pad.y_val, pad.z_val - altitude_m, 3).join()
    client.hoverAsync().join()
    while True:
        yield get_bottom_image(client)


def summarize(frame_ms, network_ms, inference_ms):
    frame_ms = np.array(frame_ms)
    # Gemiddelde FPS = 1000 / gemiddelde frametijd (= frames per seconde over de hele reeks).
    # 5e percentiel FPS = de 5 % traagste frames, dus 1000 / 95e percentiel van de frametijd.
    print(f"\nFrames gemeten:         {len(frame_ms)}")
    print(f"Gemiddelde FPS:         {1000 / frame_ms.mean():.1f}")
    print(f"5e percentiel FPS:      {1000 / np.percentile(frame_ms, 95):.1f}")
    print(f"Frametijd gemiddeld:    {frame_ms.mean():.1f} ms")
    print(f"  waarvan inferentie:   {np.mean(inference_ms):.1f} ms")
    print(f"  waarvan netwerk:      {np.mean(network_ms):.1f} ms")
    print(f"Eis >= 20 FPS gehaald:  {'ja' if 1000 / frame_ms.mean() >= 20 else 'nee'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--model", required=True, help="naam voor de log, bv. yolo26s of rtdetr-l")
    parser.add_argument("--precision", required=True, help="rekenprecisie voor de log: fp32, fp16, int8 (of pt)")
    parser.add_argument("--source", choices=["airsim", "folder"], default="airsim")
    parser.add_argument("--folder", help="map met beelden bij --source folder")
    parser.add_argument("--frames", type=int, default=500, help="minstens 500 volgens het meetplan")
    parser.add_argument("--warmup", type=int, default=20, help="eerste frames niet meetellen")
    parser.add_argument("--altitude", type=float, default=10.0, help="hoogte bij --source airsim")
    args = parser.parse_args()

    frames = folder_frames(args.folder) if args.source == "folder" else airsim_frames(args.altitude)
    client = InferenceClient(args.host, args.port)
    decider = LandingDecider(max_search_frames=10**9)   # hier meten we tijd, geen landing
    log_path = PROJECT_DIR / "results" / "chain" / f"chain_{args.model}_{args.precision}_{time.strftime('%Y%m%d_%H%M%S')}.csv"

    frame_ms, network_ms, inference_ms = [], [], []
    with CsvLogger(log_path) as log:
        for frame_id in range(args.warmup + args.frames):
            t_start = time.perf_counter()
            image = next(frames)
            t_captured = time.perf_counter()
            reply = client.infer(frame_id, image)
            best = best_detection(reply)
            decision = decider.update(best is not None, best["conf"] if best else 0.0)
            t_end = time.perf_counter()

            if frame_id < args.warmup:
                continue
            ms = (t_end - t_start) * 1000
            frame_ms.append(ms)
            network_ms.append(reply["network_ms"])
            inference_ms.append(reply["inference_ms"])
            box = best["box"] if best else ["", "", "", ""]
            log.log(model=args.model, precision_mode=args.precision, frame_id=frame_id,
                    detected=int(best is not None), confidence=best["conf"] if best else "",
                    box_x1=box[0], box_y1=box[1], box_x2=box[2], box_y2=box[3],
                    capture_ms=f"{(t_captured - t_start) * 1000:.2f}",
                    inference_ms=f"{reply['inference_ms']:.2f}", server_ms=f"{reply['server_ms']:.2f}",
                    network_ms=f"{reply['network_ms']:.2f}", frame_ms=f"{ms:.2f}",
                    fps=f"{1000 / ms:.2f}", decision=decision)
    client.close()
    summarize(frame_ms, network_ms, inference_ms)
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
