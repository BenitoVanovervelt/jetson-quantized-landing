"""Moeilijkere testsets: dezelfde 1000 testbeelden onder slechtere omstandigheden.

Waarom: op de gewone testset en in de live test scoren alle 6 engines (bijna) perfect.
Dan is niet te zien waar de detectie faalt en of kwantisatie daar een verschil maakt
(deelvragen 7 en 8). Zonder opnieuw te trainen maken we de test moeilijker.

Elke conditie gebruikt exact dezelfde posities (drone_x/y/z en yaw uit
data/landingpad/metadata.csv): enkel het beeld verandert.

Let op, gimbal: de gewone dataset is gemaakt zonder gimbal (camera draaide mee met de
yaw van de drone). Nu staat de gimbal aan en kijkt de camera altijd in dezelfde richting.
Opnieuw gerenderde beelden zijn op dezelfde positie dus gedraaid t.o.v. de oude labels.
Daarom krijgen de weercondities nieuwe labels uit simGetDetections (zelfde methode als de
gewone dataset), en is er een conditie 'clean': dezelfde posities, opnieuw gerenderd
zonder weer. Mist en stof vergelijk je met 'clean', niet met de gewone testset.
De beeldvervormingen (dark, blur, noise) gebruiken de originele beelden en labels.

Twee soorten condities:
  weer (in AirSim opnieuw gerenderd, Unreal op Play)
    clean              geen weer: referentie voor mist en stof
    fog_05, fog_10     mist met sterkte 0,5 en 1,0 (simSetWeatherParameter Fog)
    dust_05, dust_10   stof met sterkte 0,5 en 1,0 (Dust)
    Regen en sneeuw zijn getest maar weggelaten: in de Blocks-scène veranderen ze het
    beeld van een camera die recht naar beneden kijkt bijna niet.
  beeldvervorming (op de bestaande testbeelden, geen Unreal nodig, vaste seed)
    dark    weinig licht: helderheid x 0,25 (2 stops onderbelicht) + ruis sigma 6
    blur    bewegingsonscherpte: lijn van 15 px in een willekeurige richting
    noise   sensorruis: gaussisch, sigma 20 (op 0..255)

Gebruik (vanuit de projectmap, in airsim-env):
    python data_gen/generate_hard_testset.py --conditions dark blur noise          # zonder Unreal
    python data_gen/generate_hard_testset.py --conditions clean fog_05 fog_10 dust_05 dust_10   # Unreal op Play

Resultaat per conditie in data/hard/<conditie>/:
    images/test/*.jpg, labels/test/*.txt, metadata.csv
    (beeldvervorming: kopie van labels en metadata; weer: nieuwe labels en box-kolommen)
Evalueren op de Jetson: jetson/evaluate.py --data data/hard/<conditie> --condition <conditie>
"""
import argparse
import csv
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_DIR / "data" / "landingpad"
HARD_DIR = PROJECT_DIR / "data" / "hard"
JPEG_QUALITY = 90       # zelfde als de gewone dataset en de stream naar de Jetson
MAX_DRIFT_M = 0.01      # zelfde controle als bij de datageneratie

WEATHER = {"clean": ("Fog", 0.0), "fog_05": ("Fog", 0.5), "fog_10": ("Fog", 1.0), "dust_05": ("Dust", 0.5), "dust_10": ("Dust", 1.0)}
CORRUPTIONS = ["dark", "blur", "noise"]


def corrupt(image, condition, rng):
    """Pas een beeldvervorming toe op een BGR-beeld (uint8)."""
    img = image.astype(np.float32)
    if condition == "dark":
        img = img * 0.25 + rng.normal(0, 6, img.shape)
    elif condition == "noise":
        img = img + rng.normal(0, 20, img.shape)
    elif condition == "blur":
        length, angle = 15, rng.uniform(0, 180)
        kernel = np.zeros((length, length), np.float32)
        kernel[length // 2, :] = 1.0
        rotation = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1.0)
        kernel = cv2.warpAffine(kernel, rotation, (length, length))
        img = cv2.filter2D(img, -1, kernel / kernel.sum())
    return np.clip(img, 0, 255).astype(np.uint8)


def read_test_rows():
    with open(SOURCE_DIR / "metadata.csv", newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "test"]
    return rows, list(rows[0].keys())


def prepare_output(condition, rows, columns):
    out = HARD_DIR / condition
    if out.exists():
        sys.exit(f"FOUT: {out} bestaat al. Verwijder die map eerst, zodat er geen oude en nieuwe beelden "
                 "door elkaar staan.")
    (out / "images" / "test").mkdir(parents=True)
    shutil.copytree(SOURCE_DIR / "labels" / "test", out / "labels" / "test")   # zelfde posities = zelfde labels
    with open(out / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return out


def make_corruption(condition, rows, columns):
    out = prepare_output(condition, rows, columns)
    rng = np.random.default_rng(sum(map(ord, condition)))   # vaste seed per conditie
    for i, row in enumerate(rows):
        image = cv2.imread(str(SOURCE_DIR / row["file"]))
        cv2.imwrite(str(out / row["file"]), corrupt(image, condition, rng), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if (i + 1) % 250 == 0:
            print(f"  {condition}: {i + 1}/{len(rows)}")


def make_weather(client, condition, rows, columns):
    import airsim
    sys.path.insert(0, str(PROJECT_DIR / "data_gen"))
    from generate_dataset import capture, to_yolo_label   # zelfde teleportatie en labels als de gewone dataset

    out = prepare_output(condition, rows, columns)
    new_rows = []
    params = [airsim.WeatherParameter.Fog, airsim.WeatherParameter.Dust]
    for p in params:
        client.simSetWeatherParameter(p, 0)
    name, strength = WEATHER[condition]
    client.simSetWeatherParameter(getattr(airsim.WeatherParameter, name), strength)
    bad = 0
    for i, row in enumerate(rows):
        target = (float(row["drone_x"]), float(row["drone_y"]), float(row["drone_z"]))
        # capture() probeert zelf al 5 keer; zelden (3 op 4000) staat de drone toch verkeerd.
        # Dan gewoon opnieuw: de positie is bekend, dus het beeld blijft exact vergelijkbaar.
        for _ in range(5):
            image, detections, actual, drift = capture(client, target, float(row["yaw_deg"]))
            if drift <= MAX_DRIFT_M:
                break
        if drift > MAX_DRIFT_M:
            bad += 1
            print(f"  Let op: {row['file']} staat {drift:.3f} m naast de oorspronkelijke positie")
        cv2.imwrite(str(out / row["file"]), image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        # Nieuw label uit de detectie-API (de camera kijkt nu anders dan bij de gewone dataset).
        clipped, yolo = to_yolo_label(detections[0].box2D) if detections else (None, None)
        label_file = out / "labels" / "test" / (Path(row["file"]).stem + ".txt")
        label_file.write_text("0 {:.6f} {:.6f} {:.6f} {:.6f}\n".format(*yolo) if yolo else "")
        new = dict(row, visible=int(yolo is not None))
        for k, v in zip(["box_x1", "box_y1", "box_x2", "box_y2"], clipped or [""] * 4):
            new[k] = round(v, 1) if clipped else ""
        new["box_w_px"] = round(clipped[2] - clipped[0], 1) if clipped else ""
        new["box_h_px"] = round(clipped[3] - clipped[1], 1) if clipped else ""
        new_rows.append(new)
        if (i + 1) % 250 == 0:
            print(f"  {condition}: {i + 1}/{len(rows)}")
    client.simSetWeatherParameter(getattr(airsim.WeatherParameter, name), 0)
    with open(out / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(new_rows)
    lost = sum(r["visible"] == "1" and n["visible"] == 0 for r, n in zip(rows, new_rows))
    print(f"  {condition}: {lost} positieve beelden waar het platform nu buiten beeld valt (andere camerarichting)")
    print(f"  {condition}: {bad} beelden met afwijkende positie")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--conditions", nargs="+", required=True, choices=list(WEATHER) + CORRUPTIONS)
    args = parser.parse_args()
    rows, columns = read_test_rows()
    print(f"{len(rows)} testbeelden uit {SOURCE_DIR}")

    for condition in [c for c in args.conditions if c in CORRUPTIONS]:
        make_corruption(condition, rows, columns)

    weather = [c for c in args.conditions if c in WEATHER]
    if weather:
        import airsim
        sys.path.insert(0, str(PROJECT_DIR / "sim"))
        from airsim_utils import check_simulator_running
        check_simulator_running()
        client = airsim.MultirotorClient(timeout_value=60)
        client.confirmConnection()
        sys.path.insert(0, str(PROJECT_DIR / "data_gen"))
        from generate_dataset import setup
        setup(client)                # detectiefilter op LandingPad* (voor de nieuwe labels)
        client.simPause(True)        # zelfde werkwijze als generate_dataset.py
        client.simEnableWeather(True)
        try:
            for condition in weather:
                make_weather(client, condition, rows, columns)
        finally:
            for p in [airsim.WeatherParameter.Fog, airsim.WeatherParameter.Dust]:
                client.simSetWeatherParameter(p, 0)
            client.simEnableWeather(False)
            client.simPause(False)
    print(f"Klaar: {HARD_DIR}")


if __name__ == "__main__":
    main()
