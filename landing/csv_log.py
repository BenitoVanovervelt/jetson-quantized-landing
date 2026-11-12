"""CSV-logging met vaste kolommen uit het meetplan.

Alle metingen (ketentest en live landingen) schrijven naar hetzelfde formaat,
zodat de analyse één soort bestand moet lezen. Een kolom die voor een meting
niet van toepassing is, blijft leeg.

Let op de naam `precision_mode`: dat is de rekenprecisie (fp32/fp16/int8),
niet de metriek precision.
"""
import csv
import time
from pathlib import Path

COLUMNS = [
    "timestamp", "attempt_id", "model", "precision_mode",
    "theta_deg", "distance_m", "phi_deg", "frame_id",
    "detected", "confidence", "box_x1", "box_y1", "box_x2", "box_y2",
    "capture_ms", "inference_ms", "server_ms", "network_ms", "frame_ms", "fps",
    "decision", "drone_x", "drone_y", "drone_z", "altitude_m",
    "pad_present", "outcome", "outcome_reasons", "position_error_m", "landing_speed_ms", "tilt_deg",
]


class CsvLogger:
    """Schrijft één rij per frame (of per poging) en flusht meteen.

    Meteen flushen: crasht een lange meetreeks, dan staat alles tot dan toe al op schijf.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self.path.exists()
        self._file = open(self.path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=COLUMNS)
        if new_file:
            self._writer.writeheader()

    def log(self, **values):
        unknown = set(values) - set(COLUMNS)
        if unknown:
            raise ValueError(f"Onbekende kolommen: {sorted(unknown)}")
        values.setdefault("timestamp", f"{time.time():.6f}")
        self._writer.writerow(values)
        self._file.flush()

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
