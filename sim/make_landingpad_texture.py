"""Genereer de textuur van het landingsplatform: assets/landingpad_H.png.

Witte H in een witte cirkel op een donkergrijze achtergrond, zoals een helipad.
1024 x 1024 px: een macht van 2, zodat Unreal er mipmaps van kan maken
(scherp beeld op elke afstand).

Gebruik:
    python sim/make_landingpad_texture.py
"""
from pathlib import Path

import cv2
import numpy as np

SIZE = 1024
BACKGROUND = (60, 60, 60)    # donkergrijs (BGR)
WHITE = (255, 255, 255)

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_DIR / "assets" / "landingpad_H.png"


def main():
    image = np.full((SIZE, SIZE, 3), BACKGROUND, dtype=np.uint8)
    center = SIZE // 2

    # Cirkel: buitenrand op 42 % van de breedte, lijndikte 4 %.
    cv2.circle(image, (center, center), int(0.42 * SIZE), WHITE,
               thickness=int(0.04 * SIZE), lineType=cv2.LINE_AA)

    # H: 40 % hoog, 28 % breed, balken van 8 % dik.
    h_height = int(0.40 * SIZE)
    h_width = int(0.28 * SIZE)
    bar = int(0.08 * SIZE)
    top = center - h_height // 2
    bottom = center + h_height // 2
    left = center - h_width // 2
    right = center + h_width // 2
    cv2.rectangle(image, (left, top), (left + bar, bottom), WHITE, -1)            # linkerpoot
    cv2.rectangle(image, (right - bar, top), (right, bottom), WHITE, -1)          # rechterpoot
    cv2.rectangle(image, (left, center - bar // 2), (right, center + bar // 2), WHITE, -1)  # dwarsbalk

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(OUTPUT_PATH), image)
    print(f"Textuur opgeslagen: {OUTPUT_PATH} ({SIZE} x {SIZE} px)")


if __name__ == "__main__":
    main()
