"""Unittests voor de landingslogica, de uitkomstbepaling en de CSV-logging.

Gebruik (vanuit de projectmap, elke Python-omgeving):
    python -m unittest landing.test_logic -v
"""
import csv
import math
import tempfile
import unittest
from pathlib import Path

from landing.csv_log import COLUMNS, CsvLogger
from landing.decision import ABORT, DESCEND, HOLD, LandingDecider
from landing.outcome import classify_outcome, tilt_deg


class TestLandingDecider(unittest.TestCase):
    def run_frames(self, decider, frames):
        return [decider.update(det, conf) for det, conf in frames]

    def test_descends_after_five_consecutive_frames(self):
        decisions = self.run_frames(LandingDecider(), [(True, 0.9)] * 5)
        self.assertEqual(decisions, [HOLD, HOLD, HOLD, HOLD, DESCEND])

    def test_low_confidence_resets_streak(self):
        frames = [(True, 0.9)] * 4 + [(True, 0.3)] + [(True, 0.9)] * 4
        decisions = self.run_frames(LandingDecider(), frames)
        self.assertNotIn(DESCEND, decisions)

    def test_threshold_is_inclusive(self):
        decisions = self.run_frames(LandingDecider(conf_threshold=0.5), [(True, 0.5)] * 5)
        self.assertEqual(decisions[-1], DESCEND)

    def test_lost_detection_during_descent_aborts_after_three_frames(self):
        decider = LandingDecider()
        self.run_frames(decider, [(True, 0.9)] * 6)
        self.assertEqual(self.run_frames(decider, [(False, 0.0)] * 3), [DESCEND, DESCEND, ABORT])

    def test_single_missed_frame_during_descent_continues(self):
        decider = LandingDecider()
        frames = [(True, 0.9)] * 5 + [(False, 0.0)] * 2 + [(True, 0.9)] + [(False, 0.0)] * 2
        self.assertEqual(self.run_frames(decider, frames)[-1], DESCEND)

    def test_abort_is_final(self):
        decider = LandingDecider()
        self.run_frames(decider, [(True, 0.9)] * 5 + [(False, 0.0)] * 3)
        self.assertEqual(self.run_frames(decider, [(True, 0.9)] * 10)[-1], ABORT)

    def test_abort_when_platform_never_found(self):
        decisions = self.run_frames(LandingDecider(max_search_frames=20), [(False, 0.0)] * 20)
        self.assertEqual(decisions[-1], ABORT)
        self.assertEqual(decisions[-2], HOLD)


class TestOutcome(unittest.TestCase):
    def test_four_cells_of_the_table(self):
        self.assertEqual(classify_outcome(True, False)[0], "onterecht_afgebroken")
        self.assertEqual(classify_outcome(False, False)[0], "terecht_afgebroken")
        self.assertEqual(classify_outcome(False, True)[0], "foute_landing")
        self.assertEqual(classify_outcome(True, True, 0.1, 0.2, 2.0)[0], "goede_landing")

    def test_bad_landing_reasons(self):
        outcome, reasons = classify_outcome(True, True, 0.30, 0.8, 12.0)
        self.assertEqual(outcome, "slechte_landing")
        self.assertEqual(reasons, ["naast", "te_hard", "te_scheef"])

    def test_contact_with_other_object_is_bad_landing(self):
        outcome, reasons = classify_outcome(True, True, 0.0, 0.2, 0.0, on_pad=False)
        self.assertEqual((outcome, reasons), ("slechte_landing", ["ander_object"]))

    def test_limits_are_exclusive(self):
        # Grenzen uit het contractplan zijn strikt: < 0,25 m, < 0,5 m/s, < 10°.
        self.assertEqual(classify_outcome(True, True, 0.25, 0.1, 1.0)[1], ["naast"])

    def test_tilt_combines_roll_and_pitch(self):
        self.assertAlmostEqual(tilt_deg(0.0, 0.0), 0.0)
        self.assertAlmostEqual(tilt_deg(math.radians(10), 0.0), 10.0, places=6)
        self.assertGreater(tilt_deg(math.radians(7), math.radians(7)), 9.8)


class TestCsvLogger(unittest.TestCase):
    def test_writes_header_once_and_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.csv"
            with CsvLogger(path) as log:
                log.log(model="yolo26s", frame_id=1)
            with CsvLogger(path) as log:
                log.log(model="yolo26s", frame_id=2)
            with open(path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2)
            self.assertEqual(list(rows[0].keys()), COLUMNS)

    def test_rejects_unknown_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            with CsvLogger(Path(tmp) / "log.csv") as log:
                with self.assertRaises(ValueError):
                    log.log(precision=0.9)   # metriek, geen kolom


if __name__ == "__main__":
    unittest.main()
