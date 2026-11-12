"""Visuele controle van de labels: teken de YOLO-box op een aantal willekeurige beelden.

Automatische labels moet je toch met het oog nakijken: een verschoven of te
grote box valt in de cijfers niet op, maar het model leert hem wel.

Gebruik:
    python data_gen/check_labels.py --split train --count 25

Resultaat: results/label_check/<split>_grid.jpg (raster van 5 x 5 beelden).
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "landingpad"
TILE = 256


def draw(image_path, label_path):
    image = cv2.imread(str(image_path))
    h, w = image.shape[:2]
    lines = label_path.read_text().split("\n") if label_path.exists() else []
    for line in filter(None, lines):
        _, xc, yc, bw, bh = map(float, line.split())
        x1, y1 = int((xc - bw / 2) * w), int((yc - bh / 2) * h)
        x2, y2 = int((xc + bw / 2) * w), int((yc + bh / 2) * h)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if not any(lines):
        cv2.putText(image, "geen platform", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.putText(image, image_path.stem, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return cv2.resize(image, (TILE, TILE))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train")
    parser.add_argument("--count", type=int, default=25)
    args = parser.parse_args()

    images = sorted((DATA_DIR / "images" / args.split).glob("*.jpg"))
    picks = random.Random(0).sample(images, min(args.count, len(images)))
    tiles = [draw(p, DATA_DIR / "labels" / args.split / f"{p.stem}.txt") for p in picks]
    columns = 5
    while len(tiles) % columns:
        tiles.append(np.zeros((TILE, TILE, 3), dtype=np.uint8))
    grid = np.vstack([np.hstack(tiles[i:i + columns]) for i in range(0, len(tiles), columns)])

    out = PROJECT_DIR / "results" / "label_check" / f"{args.split}_grid.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), grid)
    print(f"Opgeslagen: {out}")


if __name__ == "__main__":
    main()
