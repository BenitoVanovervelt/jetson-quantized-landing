"""Uitkomst van een landingspoging bepalen, met de grenzen uit het contractplan.

                     | drone daalt                         | drone breekt af
  platform aanwezig  | goede_landing of slechte_landing    | onterecht_afgebroken
  platform afwezig   | foute_landing                       | terecht_afgebroken
"""
import math

MAX_POSITION_ERROR_M = 0.25
MAX_LANDING_SPEED_MS = 0.5
MAX_TILT_DEG = 10.0


def tilt_deg(roll_rad, pitch_rad):
    """Kanteling: hoek tussen de verticale as van de drone en de echte verticale lijn.

    Roll en pitch samen, niet elk apart: 7° roll + 7° pitch is samen bijna 10°.
    cos(kanteling) = cos(roll) * cos(pitch).
    """
    cos_tilt = math.cos(roll_rad) * math.cos(pitch_rad)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_tilt))))


def horizontal_error_m(drone_x, drone_y, pad_x, pad_y):
    """Horizontale afstand tussen drone en midden van het platform (NED, meter)."""
    return math.hypot(drone_x - pad_x, drone_y - pad_y)


def classify_outcome(pad_present, descended, position_error_m=None,
                     landing_speed_ms=None, tilt=None, on_pad=True):
    """Geef (uitkomst, redenen) terug. Redenen zijn enkel ingevuld bij slechte_landing.

    on_pad: het eerste contact was met het platform. Contact met iets anders (bv. een blok
    van de scène) is nooit een goede landing, ook al klopt de positie van de drone.
    """
    if not descended:
        return ("onterecht_afgebroken" if pad_present else "terecht_afgebroken"), []
    if not pad_present:
        return "foute_landing", []

    reasons = [] if on_pad else ["ander_object"]
    if position_error_m >= MAX_POSITION_ERROR_M:
        reasons.append("naast")
    if landing_speed_ms >= MAX_LANDING_SPEED_MS:
        reasons.append("te_hard")
    if tilt >= MAX_TILT_DEG:
        reasons.append("te_scheef")
    return ("slechte_landing" if reasons else "goede_landing"), reasons
