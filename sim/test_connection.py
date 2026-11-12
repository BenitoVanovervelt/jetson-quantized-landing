"""Verbindingstest met AirSim.

Verbindt met de simulator, laat de drone opstijgen naar 10 m, haalt één beeld
op van bottom_cam en slaat het op als results/bottom_cam_test.png.

Gebruik (Unreal moet op Play staan):
    python sim/test_connection.py
"""
from pathlib import Path

import cv2

from airsim_utils import connect, get_bottom_image, land_and_release

ALTITUDE_M = 10.0
SPEED_MS = 3.0

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_DIR / "results" / "bottom_cam_test.png"


def main():
    client = connect()

    print("Opstijgen...")
    client.takeoffAsync().join()
    # NED: z wijst naar beneden, dus 10 m hoogte is z = -10.
    client.moveToPositionAsync(0, 0, -ALTITUDE_M, SPEED_MS).join()
    print(f"Drone staat op {ALTITUDE_M:.0f} m hoogte")

    image = get_bottom_image(client)
    height, width = image.shape[:2]
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(OUTPUT_PATH), image)   # BGR, zoals OpenCV verwacht
    print(f"Beeld opgeslagen: {OUTPUT_PATH}")
    print(f"Breedte: {width} px, hoogte: {height} px")

    land_and_release(client)


if __name__ == "__main__":
    main()
