"""Landingslogica: beslist per frame of de drone mag dalen.

Regel uit het contractplan: dalen alleen als het platform in N opeenvolgende
frames gedetecteerd wordt boven een vaste confidence-drempel, anders afbreken.

Deze module kent geen AirSim en geen netwerk: ze krijgt per frame "gedetecteerd
ja/nee + confidence" en geeft een beslissing terug. Daardoor is ze los te testen
(landing/test_decision.py) en is de regel op één plek vastgelegd.
"""

HOLD = "hold"          # nog niet genoeg opeenvolgende detecties: hoogte houden
DESCEND = "descend"    # platform N frames na elkaar gezien: dalen
ABORT = "abort"        # landing afgebroken (definitief voor deze poging)

REQUIRED_FRAMES = 5    # N in FPS >= N * v / s_max
CONF_THRESHOLD = 0.5   # vaste drempel; confidence op de testset en in de ketentest ligt rond 0,9-0,98
MAX_MISSED_FRAMES = 3  # tijdens het dalen: afbreken pas bij 3 gemiste frames na elkaar


class LandingDecider:
    """Houdt de reeks opeenvolgende detecties bij en geeft per frame een beslissing.

    - Zolang de reeks korter is dan N: HOLD.
    - Reeks >= N: DESCEND.
    - Valt de detectie weg terwijl de drone al daalt: ABORT, maar pas na
      max_missed_frames gemiste frames na elkaar. Eén gemist frame is bij 20 FPS
      maar 5 cm dalen; meteen afbreken zou de logica gevoelig maken voor ruis.
    - Wordt de reeks van N niet bereikt binnen max_search_frames: ABORT.
    - ABORT blijft ABORT: een afgebroken landing wordt niet hervat.
    """

    def __init__(self, conf_threshold=CONF_THRESHOLD, required_frames=REQUIRED_FRAMES,
                 max_search_frames=100, max_missed_frames=MAX_MISSED_FRAMES):
        self.conf_threshold = conf_threshold
        self.required_frames = required_frames
        self.max_search_frames = max_search_frames   # 100 frames = 5 s bij 20 FPS
        self.max_missed_frames = max_missed_frames
        self.streak = 0
        self.missed = 0
        self.frames_seen = 0
        self.decision = HOLD

    def update(self, detected, confidence):
        """Verwerk één frame en geef de nieuwe beslissing terug."""
        self.frames_seen += 1
        if self.decision == ABORT:
            return ABORT

        if detected and confidence >= self.conf_threshold:
            self.streak += 1
            self.missed = 0
        else:
            self.streak = 0
            self.missed += 1

        if self.decision == DESCEND:
            # Was aan het dalen: blijven dalen tot het platform te lang weg is.
            if self.missed >= self.max_missed_frames:
                self.decision = ABORT
        elif self.streak >= self.required_frames:
            self.decision = DESCEND
        elif self.frames_seen >= self.max_search_frames:
            self.decision = ABORT
        else:
            self.decision = HOLD
        return self.decision
