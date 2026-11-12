"""Genereer de gelabelde dataset in AirSim (deelvraag 4).

Per beeld: drone teleporteren (simSetVehiclePose), beeld van bottom_cam ophalen,
box van het platform via de detectie-API (simGetDetections), omzetten naar YOLO.

Posities (NED, t.o.v. het platform):  x = d sinθ cosφ,  y = d sinθ sinφ,  z = -d cosθ

Splitsing zonder lekkage tussen train en test:
  train / val / calib   θ willekeurig in [0°, 45°], d willekeurig in [4, 32] m,
                        φ uit {0°, 10°, ..., 350°}
  test                  θ ∈ {0, 15, 30, 45}°, d ∈ {5, 10, 20, 30} m (het raster uit het
                        contractplan, zodat de resultaten per θ en d te rapporteren zijn),
                        φ uit {5°, 15°, ..., 355°}: nooit dezelfde richting als in train
In elke split is ±10 % negatief: de drone hangt zo ver opzij dat het platform buiten
beeld valt (nodig om afbreken te leren en te testen). De gierhoek (yaw) van de drone
is willekeurig, zodat het platform in alle oriëntaties in beeld komt.

Gebruik (Unreal op Play, platform in de scène, in airsim-env):
    python data_gen/generate_dataset.py --scale 0.01      # eerst een mini-run van ±50 beelden
    python data_gen/generate_dataset.py                   # volledige dataset (±4800 beelden)

Resultaat in data/landingpad/:
    images/<split>/*.jpg, labels/<split>/*.txt, metadata.csv, data.yaml, calib.yaml
"""
import argparse
import csv
import math
import random
import sys
import time
from pathlib import Path

import airsim
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sim"))
from airsim_utils import CAMERA_NAME, check_simulator_running, get_bottom_image  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "data" / "landingpad"

SPLIT_SIZES = {"train": 3000, "val": 500, "test": 1000, "calib": 300}
NEGATIVE_FRACTION = 0.10
TEST_THETAS_DEG = [0, 15, 30, 45]
TEST_DISTANCES_M = [5, 10, 20, 30]
TRAIN_PHIS_DEG = list(range(0, 360, 10))
TEST_PHIS_DEG = list(range(5, 360, 10))
SEEDS = {"train": 1, "val": 2, "test": 3, "calib": 4}   # vaste seeds: dataset is reproduceerbaar

PAD_NAME_REGEX = "LandingPad.*"
PAD_NAME_WILDCARD = "LandingPad*"
DETECTION_RADIUS_CM = 200 * 100
IMAGE_SIZE = 640
JPEG_QUALITY = 90          # zelfde als de stream naar de Jetson
MIN_BOX_PX = 2             # kleinere (afgesneden) boxes tellen niet als zichtbaar
STEP_S = 0.01              # simulatietijd na het teleporteren (zie capture)
MAX_DRIFT_M = 0.01         # drone mag na het beeld niet verder dan dit van de doelpositie staan
MAX_TRIES = 5              # aantal pogingen om de drone op zijn plaats te krijgen

METADATA_COLUMNS = ["file", "split", "negative", "theta_deg", "distance_m", "phi_deg", "yaw_deg",
                    "drone_x", "drone_y", "drone_z", "pad_x", "pad_y", "pad_z",
                    "visible", "box_x1", "box_y1", "box_x2", "box_y2", "box_w_px", "box_h_px"]


def sample_positive(split, rng):
    """Geef (θ, d, φ) in graden en meter."""
    if split == "test":
        return rng.choice(TEST_THETAS_DEG), rng.choice(TEST_DISTANCES_M), rng.choice(TEST_PHIS_DEG)
    return rng.uniform(0, 45), rng.uniform(4, 32), rng.choice(TRAIN_PHIS_DEG)


def test_grid_positions(count, rng):
    """Test: elke (θ, d)-combinatie even vaak, zodat elke cel van de tabel genoeg beelden heeft."""
    combos = [(t, d) for t in TEST_THETAS_DEG for d in TEST_DISTANCES_M]
    positions = [(*combos[i % len(combos)], rng.choice(TEST_PHIS_DEG)) for i in range(count)]
    rng.shuffle(positions)
    return positions


def relative_position(theta_deg, distance_m, phi_deg):
    theta, phi = math.radians(theta_deg), math.radians(phi_deg)
    return (distance_m * math.sin(theta) * math.cos(phi),
            distance_m * math.sin(theta) * math.sin(phi),
            -distance_m * math.cos(theta))


def negative_position(rng):
    """Drone op 5–30 m hoogte, zo ver opzij dat het platform zeker buiten beeld valt.

    Met FOV 90° ziet de camera op hoogte h tot h·√2 opzij (hoek van het beeld).
    Daar komt 2 m marge bij, plus een willekeurig stuk zodat de achtergrond varieert.
    """
    height = rng.uniform(5, 30)
    offset = height * math.sqrt(2) + 2 + rng.uniform(0, 20)
    direction = rng.uniform(0, 2 * math.pi)
    return offset * math.cos(direction), offset * math.sin(direction), -height


def to_yolo_label(box):
    """Clip de box tot het beeld en zet om naar YOLO. None als er (bijna) niets zichtbaar is."""
    x1 = min(max(box.min.x_val, 0), IMAGE_SIZE)
    y1 = min(max(box.min.y_val, 0), IMAGE_SIZE)
    x2 = min(max(box.max.x_val, 0), IMAGE_SIZE)
    y2 = min(max(box.max.y_val, 0), IMAGE_SIZE)
    if x2 - x1 < MIN_BOX_PX or y2 - y1 < MIN_BOX_PX:
        return None, None
    clipped = (x1, y1, x2, y2)
    yolo = ((x1 + x2) / 2 / IMAGE_SIZE, (y1 + y2) / 2 / IMAGE_SIZE,
            (x2 - x1) / IMAGE_SIZE, (y2 - y1) / IMAGE_SIZE)
    return clipped, yolo


def setup(client):
    names = client.simListSceneObjects(PAD_NAME_REGEX)
    if not names:
        sys.exit("FOUT: geen object 'LandingPad' in de scène.")
    pad = client.simGetObjectPose(names[0]).position
    if math.isnan(pad.x_val):
        sys.exit(f"FOUT: positie van '{names[0]}' onbekend.")
    client.simSetDetectionFilterRadius(CAMERA_NAME, airsim.ImageType.Scene, DETECTION_RADIUS_CM)
    client.simClearDetectionMeshNames(CAMERA_NAME, airsim.ImageType.Scene)
    client.simAddDetectionFilterMeshName(CAMERA_NAME, airsim.ImageType.Scene, PAD_NAME_WILDCARD)
    return pad


def capture(client, target, yaw_deg):
    """Teleporteer de drone en neem een beeld, terwijl de simulatie gepauzeerd is.

    Getest (zie docs/INSTALL.md, sectie datageneratie):
    - simSetVehiclePose zet enkel de positie, niet de snelheid: de drone blijft
      vallen en versnelt over de beelden heen. simSetKinematics zet positie en
      snelheid (0) samen.
    - In pauze wordt de nieuwe positie pas toegepast bij een stap van de fysica.
      simContinueForTime(0.01) laat 10 ms lopen: genoeg om de positie toe te passen,
      te kort om te vallen (afwijking ± 0,3 mm). 2 ms was soms te kort.
    - Ook 10 ms was in 20 van de 4800 beelden te kort (Unreal even trager). Daarom
      wordt de positie gecontroleerd en de stap herhaald tot de drone op zijn plaats staat.
    """
    state = airsim.KinematicsState()   # snelheden en versnellingen standaard 0
    state.position = airsim.Vector3r(*target)
    state.orientation = airsim.to_quaternion(0, 0, math.radians(yaw_deg))
    for _ in range(MAX_TRIES):
        client.simSetKinematics(state, True)
        client.simContinueForTime(STEP_S)
        while not client.simIsPause():
            time.sleep(0.005)
        actual = client.simGetVehiclePose().position
        drift = math.dist(target, (actual.x_val, actual.y_val, actual.z_val))
        if drift <= MAX_DRIFT_M:
            break
    image = get_bottom_image(client)
    detections = client.simGetDetections(CAMERA_NAME, airsim.ImageType.Scene)
    actual = client.simGetVehiclePose().position
    drift = math.dist(target, (actual.x_val, actual.y_val, actual.z_val))
    return image, detections, actual, drift


def save_sample(client, pad, split, index, negative, theta, distance, phi, yaw_deg, rel):
    """Neem het beeld, schrijf beeld en label weg, en geef de metadata-rij terug."""
    target = (pad.x_val + rel[0], pad.y_val + rel[1], pad.z_val + rel[2])
    image, detections, actual, drift = capture(client, target, yaw_deg)
    if drift > MAX_DRIFT_M:
        print(f"  Let op: drone {drift:.2f} m van de doelpositie bij {split} #{index}")

    clipped, yolo = (None, None)
    if detections:
        clipped, yolo = to_yolo_label(detections[0].box2D)

    stem = f"{split}_{index:05d}"
    cv2.imwrite(str(OUTPUT_DIR / "images" / split / f"{stem}.jpg"), image,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    # Een leeg labelbestand = beeld zonder platform (achtergrond voor YOLO).
    (OUTPUT_DIR / "labels" / split / f"{stem}.txt").write_text(
        "0 {:.6f} {:.6f} {:.6f} {:.6f}\n".format(*yolo) if yolo else "")

    return {
        "file": f"images/{split}/{stem}.jpg", "split": split, "negative": int(negative),
        "theta_deg": theta if theta == "" else round(theta, 3),
        "distance_m": distance if distance == "" else round(distance, 3),
        "phi_deg": phi, "yaw_deg": round(yaw_deg, 2),
        "drone_x": round(actual.x_val, 3), "drone_y": round(actual.y_val, 3), "drone_z": round(actual.z_val, 3),
        "pad_x": round(pad.x_val, 3), "pad_y": round(pad.y_val, 3), "pad_z": round(pad.z_val, 3),
        "visible": int(yolo is not None),
        "box_x1": round(clipped[0], 1) if clipped else "", "box_y1": round(clipped[1], 1) if clipped else "",
        "box_x2": round(clipped[2], 1) if clipped else "", "box_y2": round(clipped[3], 1) if clipped else "",
        "box_w_px": round(clipped[2] - clipped[0], 1) if clipped else "",
        "box_h_px": round(clipped[3] - clipped[1], 1) if clipped else "",
    }


def generate_split(client, pad, split, count, writer):
    rng = random.Random(SEEDS[split])
    n_negative = round(count * NEGATIVE_FRACTION)
    positives = (test_grid_positions(count - n_negative, rng) if split == "test"
                 else [sample_positive(split, rng) for _ in range(count - n_negative)])
    samples = [("pos", p) for p in positives] + [("neg", None)] * n_negative
    rng.shuffle(samples)

    (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_visible = 0
    for index, (kind, position) in enumerate(samples):
        yaw_deg = rng.uniform(0, 360)
        if kind == "pos":
            theta, distance, phi = position
            rel = relative_position(theta, distance, phi)
        else:
            theta = distance = phi = ""
            rel = negative_position(rng)
        row = save_sample(client, pad, split, index, kind == "neg", theta, distance, phi, yaw_deg, rel)
        writer.writerow(row)
        n_visible += row["visible"]
        if (index + 1) % 100 == 0 or index + 1 == count:
            print(f"  {split}: {index + 1}/{count} beelden, waarvan {n_visible} met platform")


def needs_repair(row, pad):
    """Positief beeld: opgeslagen positie wijkt af van θ, d, φ. Negatief beeld: platform toch zichtbaar."""
    if row["negative"] == "1":
        return row["visible"] == "1"
    rel = relative_position(float(row["theta_deg"]), float(row["distance_m"]), float(row["phi_deg"]))
    target = (pad.x_val + rel[0], pad.y_val + rel[1], pad.z_val + rel[2])
    actual = (float(row["drone_x"]), float(row["drone_y"]), float(row["drone_z"]))
    return math.dist(target, actual) > MAX_DRIFT_M


def repair(client, pad):
    """Maak enkel de mislukte beelden opnieuw, met dezelfde θ, d, φ en yaw; de rest blijft staan."""
    path = OUTPUT_DIR / "metadata.csv"
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fixed = 0
    for i, row in enumerate(rows):
        if not needs_repair(row, pad):
            continue
        split = row["split"]
        index = int(Path(row["file"]).stem.split("_")[-1])
        if row["negative"] == "1":
            # De oorspronkelijke negatieve positie is niet bewaard: neem een nieuwe, met een vaste seed.
            rel = negative_position(random.Random(SEEDS[split] * 100000 + index))
            theta = distance = phi = ""
        else:
            theta, distance, phi = float(row["theta_deg"]), float(row["distance_m"]), float(row["phi_deg"])
            rel = relative_position(theta, distance, phi)
        rows[i] = save_sample(client, pad, split, index, row["negative"] == "1",
                              theta, distance, phi, float(row["yaw_deg"]), rel)
        fixed += 1
        print(f"  Hersteld: {row['file']}")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{fixed} beelden hersteld")


def write_yaml_files():
    names = "names:\n  0: landingsplatform\n"
    (OUTPUT_DIR / "data.yaml").write_text(
        f"path: {OUTPUT_DIR.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n{names}")
    # Ultralytics kalibreert INT8 op de 'val'-map van het yaml-bestand: hier dus de kalibratieset.
    (OUTPUT_DIR / "calib.yaml").write_text(
        f"path: {OUTPUT_DIR.as_posix()}\ntrain: images/calib\nval: images/calib\n{names}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scale", type=float, default=1.0, help="0.01 = 1 %% van elke split, om te testen")
    parser.add_argument("--splits", nargs="+", default=list(SPLIT_SIZES), choices=list(SPLIT_SIZES))
    parser.add_argument("--repair", action="store_true",
                        help="bestaande dataset: enkel beelden met een foute positie opnieuw maken")
    args = parser.parse_args()

    if (OUTPUT_DIR / "metadata.csv").exists() != args.repair:
        sys.exit(f"FOUT: {OUTPUT_DIR} bestaat al. Verwijder of hernoem die map eerst, zodat er geen oude "
                 "en nieuwe beelden door elkaar staan." if not args.repair else
                 f"FOUT: geen dataset gevonden in {OUTPUT_DIR} om te herstellen.")

    check_simulator_running()
    client = airsim.MultirotorClient(timeout_value=60)
    client.confirmConnection()
    pad = setup(client)
    print(f"Platform op NED x={pad.x_val:.2f}, y={pad.y_val:.2f}, z={pad.z_val:.2f} m")

    # Pauzeren: de tijd loopt enkel in de korte stap na elke teleportatie (zie capture).
    client.simPause(True)
    if args.repair:
        try:
            repair(client, pad)
        finally:
            client.simPause(False)
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(OUTPUT_DIR / "metadata.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=METADATA_COLUMNS)
            writer.writeheader()
            for split in args.splits:
                count = max(1, round(SPLIT_SIZES[split] * args.scale))
                print(f"Split {split}: {count} beelden")
                generate_split(client, pad, split, count, writer)
    finally:
        client.simPause(False)
    write_yaml_files()
    print(f"Klaar: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
