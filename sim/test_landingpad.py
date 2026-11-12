"""Test van het landingsplatform en de detectie-API van AirSim.

1. Zoek het platform (LandingPad) in de scène.
2. Vraag zijn positie op en vlieg de drone 10 m recht erboven.
3. Haal een beeld op van bottom_cam en vraag de detectie op.
4. Teken de box2D op het beeld en sla het op als results/landingpad_detection.png.

Gebruik (Unreal moet op Play staan, met het platform in de scène):
    python sim/test_landingpad.py
"""
import math
import sys
import time
from pathlib import Path

import airsim
import cv2

from airsim_utils import CAMERA_NAME, connect, get_bottom_image, land_and_release

PAD_NAME_REGEX = "LandingPad.*"     # regex voor simListSceneObjects
PAD_NAME_WILDCARD = "LandingPad*"   # wildcard voor de detectiefilter
ALTITUDE_ABOVE_PAD_M = 10.0
SPEED_MS = 3.0
SETTLE_TIME_S = 2.0                 # drone laten stilhangen, zodat de camera recht kijkt
DETECTION_RADIUS_CM = 200 * 100     # 200 m: ruim genoeg voor afstanden tot 30 m

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_DIR / "results" / "landingpad_detection.png"


def find_landing_pad(client):
    """Geef de naam en positie (NED, meter) van het platform terug."""
    names = client.simListSceneObjects(PAD_NAME_REGEX)
    if not names:
        sys.exit(
            "FOUT: geen object gevonden dat begint met 'LandingPad'.\n"
            "Controleer de actornaam in de World Outliner en sla het level op voor je op Play drukt."
        )
    if len(names) > 1:
        print(f"Let op: meerdere platformen gevonden {names}, ik gebruik het eerste.")
    name = names[0]

    # simGetObjectPose geeft NaN terug als het object niet gevonden wordt.
    position = client.simGetObjectPose(name).position
    if math.isnan(position.x_val):
        sys.exit(f"FOUT: positie van '{name}' kon niet opgevraagd worden.")
    return name, position


def get_pad_detections(client, retries=5):
    """Vraag de detecties van het platform op, met een paar nieuwe pogingen.

    De detectie wordt per frame berekend, dus het eerste frame na het
    instellen van de filter kan nog leeg zijn.
    """
    for _ in range(retries):
        detections = client.simGetDetections(CAMERA_NAME, airsim.ImageType.Scene)
        if detections:
            return detections
        time.sleep(0.2)
    return []


def to_yolo(box, image_width, image_height):
    """Zet een box2D (pixels) om naar YOLO-formaat: genormaliseerd x_c, y_c, w, h."""
    x_min, y_min = box.min.x_val, box.min.y_val
    x_max, y_max = box.max.x_val, box.max.y_val
    x_center = (x_min + x_max) / 2 / image_width
    y_center = (y_min + y_max) / 2 / image_height
    width = (x_max - x_min) / image_width
    height = (y_max - y_min) / image_height
    return x_center, y_center, width, height


def main():
    client = connect()

    pad_name, pad_pos = find_landing_pad(client)
    print(f"Platform gevonden: {pad_name} op NED "
          f"x={pad_pos.x_val:.2f}, y={pad_pos.y_val:.2f}, z={pad_pos.z_val:.2f} m")

    # Detectiefilter instellen: alleen objecten waarvan de naam op LandingPad* past.
    client.simSetDetectionFilterRadius(CAMERA_NAME, airsim.ImageType.Scene, DETECTION_RADIUS_CM)
    client.simClearDetectionMeshNames(CAMERA_NAME, airsim.ImageType.Scene)
    client.simAddDetectionFilterMeshName(CAMERA_NAME, airsim.ImageType.Scene, PAD_NAME_WILDCARD)

    print("Opstijgen...")
    client.takeoffAsync().join()
    # NED: z wijst naar beneden, dus 10 m boven het platform is z_platform - 10.
    target_z = pad_pos.z_val - ALTITUDE_ABOVE_PAD_M
    client.moveToPositionAsync(pad_pos.x_val, pad_pos.y_val, target_z, SPEED_MS).join()
    client.hoverAsync().join()
    time.sleep(SETTLE_TIME_S)
    print(f"Drone hangt {ALTITUDE_ABOVE_PAD_M:.0f} m boven het platform")

    image = get_bottom_image(client)
    detections = get_pad_detections(client)
    image_height, image_width = image.shape[:2]

    if not detections:
        print("GEEN detectie gevonden. Het beeld wordt toch opgeslagen om te controleren.")
    for detection in detections:
        box = detection.box2D
        top_left = (int(box.min.x_val), int(box.min.y_val))
        bottom_right = (int(box.max.x_val), int(box.max.y_val))
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(image, detection.name, (top_left[0], max(top_left[1] - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        x_c, y_c, w, h = to_yolo(box, image_width, image_height)
        print(f"Detectie '{detection.name}': box2D van {top_left} tot {bottom_right} px")
        print(f"  YOLO-label: 0 {x_c:.4f} {y_c:.4f} {w:.4f} {h:.4f}")

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(OUTPUT_PATH), image)
    print(f"Beeld opgeslagen: {OUTPUT_PATH}")

    land_and_release(client)


if __name__ == "__main__":
    main()
