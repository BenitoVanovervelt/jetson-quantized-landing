"""Demo: één landing live in beeld, met box, confidence, FPS en beslissing; wisselen tussen engines.

Succescriterium uit het contractplan: "drone landt of breekt af met box en FPS in beeld,
wisselen tussen minstens 2 engines". Zelfde landingslogica en vlucht als landing/live_test.py
(LandingDecider, eerst horizontaal tot boven het platform, onder 3 m blind), maar met een
venster en toetsen in plaats van een vaste reeks pogingen. Er wordt niets gelogd: de metingen
komen uit live_test.py.

Waarom twee servers tegelijk (poort 5000 en 5001): een engine laden duurt 20-60 s. Met beide
engines al geladen gaat het wisselen meteen. De demo stuurt enkel beelden naar de actieve engine.

Gebruik (Unreal op Play, servers gestart met landing/demo_servers.sh, in airsim-env):
    python -m landing.demo --engines yolo26s_fp16@192.168.55.1:5000 rtdetr-l_fp32@192.168.55.1:5001
Zonder Jetson (plan B: server op de pc met PyTorch-model, zie jetson/server.py):
    python -m landing.demo --engines yolo26s_pt@127.0.0.1:5000

Toetsen (in het demovenster):
    spatie   nieuwe landing (starthoek wisselt: 0, 15, 30, 45 graden)
    1, 2     engine kiezen
    p        platform weg / terug (afbreken tonen)
    f        mist aan / uit (sterkte 0,5)
    q, Esc   stoppen
Handenvrij (bv. voor een reservevideo): --auto 8 doet 8 landingen na elkaar en stopt dan. In die modus
wisselt elke landing van engine en is de laatste landing zonder platform, zodat de video ook wisselen
en afbreken toont.
Reservevideo opnemen: --record presentation/video/demo.mp4 schrijft precies het demovenster weg
(H.264 via Windows Media Foundation, speelt af in de browser), in echte tijd aan 30 beelden per seconde.
Voor het beeld van de drone in Unreal: "ViewMode": "NoDisplay" uit settings.json en opnieuw Play.
"""
import argparse
import collections
import math
import random
import time

import airsim
import cv2

from landing.decision import ABORT, DESCEND, HOLD, LandingDecider
from landing.inference_client import InferenceClient, best_detection
from landing.live_test import (BLIND_ALTITUDE_M, COMMAND_S, DESCENT_SPEED_MS, FINAL_ALTITUDE_M, FINAL_SPEED_MS,
                               HORIZONTAL_GAIN, MAX_START_DRIFT_M, clear_phi, descent_velocity, place_pad,
                               scene_obstacles, start_position)
from landing.outcome import classify_outcome, horizontal_error_m, tilt_deg
from sim.airsim_utils import connect, get_bottom_image, teleport, unlock_frame_rate

WINDOW = "Landingsdemo"
ANGLES = [0, 15, 30, 45]
FOG_STRENGTH = 0.5
GREEN, ORANGE, RED, WHITE, GRAY = (60, 200, 60), (0, 165, 255), (60, 60, 230), (255, 255, 255), (170, 170, 170)
DECISION_STYLE = {HOLD: ("HOLD: wacht op 5 detecties", ORANGE), DESCEND: ("DESCEND: dalen", GREEN),
                  ABORT: ("ABORT: afgebroken, blijft hangen", RED)}
OUTCOME_COLORS = {"goede_landing": GREEN, "terecht_afgebroken": GREEN, "slechte_landing": ORANGE,
                  "onterecht_afgebroken": ORANGE, "foute_landing": RED}


class Engine:
    def __init__(self, spec):
        """spec = naam@host:poort, bv. yolo26s_fp16@192.168.55.1:5000"""
        self.name, address = spec.split("@")
        host, port = address.rsplit(":", 1)
        self.client = InferenceClient(host, int(port))


class Recorder:
    """Schrijft het demovenster naar een mp4 in echte tijd.

    De demo haalt ± 15-20 beelden per seconde en staat soms even stil (teleporteren naar de start).
    Een video heeft een vaste beeldsnelheid, dus het vorige beeld wordt herhaald tot de klok bij is.
    Zo loopt de video even snel als de demo zelf.
    """
    FPS = 30

    def __init__(self, path):
        # MSMF = Windows Media Foundation: de enige H.264-encoder die in de pip-versie van OpenCV werkt.
        self.writer = cv2.VideoWriter(path, cv2.CAP_MSMF, cv2.VideoWriter_fourcc(*"H264"), self.FPS, (640, 640))
        if not self.writer.isOpened():
            raise SystemExit(f"Kan {path} niet openen om te schrijven")
        self.start, self.written, self.last = None, 0, None

    def add(self, frame):
        now = time.perf_counter()
        if self.start is None:
            self.start = now
        due = int((now - self.start) * self.FPS)
        while self.last is not None and self.written < due:
            self.writer.write(self.last)
            self.written += 1
        self.last = frame.copy()
        self.writer.write(self.last)
        self.written += 1

    def close(self):
        self.writer.release()


def draw(image, best, state, engine, fps, inference_ms):
    """Teken box en statuspaneel op het camerabeeld (640 x 640)."""
    if best:
        x1, y1, x2, y2 = map(int, best["box"])
        color = GREEN if best["conf"] >= state["decider"].conf_threshold else ORANGE
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"platform {best['conf']:.2f}", (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)

    panel = image.copy()
    cv2.rectangle(panel, (0, 0), (640, 118), (25, 25, 25), -1)
    cv2.rectangle(panel, (0, 600), (640, 640), (25, 25, 25), -1)
    image[:] = cv2.addWeighted(panel, 0.65, image, 0.35, 0)

    fps_color = GREEN if fps >= 20 else ORANGE
    cv2.putText(image, engine.name, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
    cv2.putText(image, f"{fps:4.1f} FPS", (430, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, fps_color, 2)
    cv2.putText(image, f"inferentie {inference_ms:4.1f} ms", (430, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GRAY, 1)
    text, color = state["status"]
    cv2.putText(image, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    decider = state["decider"]
    cv2.putText(image, f"detecties op rij {min(decider.streak, 5)}/5   drempel {decider.conf_threshold}   "
                       f"hoogte {state['altitude']:4.1f} m", (10, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GRAY, 1)
    cv2.putText(image, f"start theta {state['theta']} graden   platform {'ja' if state['pad'] else 'NEE'}   "
                       f"mist {'AAN' if state['fog'] else 'uit'}", (10, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GRAY, 1)
    cv2.putText(image, "spatie: landing   1/2: engine   p: platform   f: mist   q: stop", (10, 626),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, GRAY, 1)
    return image


def go_to_start(client, pad, boxes, theta):
    """Teleporteer naar een vrije startpositie op 20 m (zelfde methode als live_test.py)."""
    phi = clear_phi(boxes, pad, theta, random.uniform(0, 360))
    target = start_position(pad, theta, phi)
    for _ in range(3):
        teleport(client, target)
        client.moveByVelocityAsync(0, 0, 0, 1.0)
        time.sleep(1.0)
        actual = client.simGetVehiclePose().position
        if math.dist(target, (actual.x_val, actual.y_val, actual.z_val)) < MAX_START_DRIFT_M:
            return
        client.moveByVelocityAsync(0, 0, -1.0, 1.0).join()   # loskomen van de grond na een landing


def judge_landing(client, contact_state, contact_object, pad, pad_present):
    k = contact_state.kinematics_estimated
    roll, pitch, _ = airsim.to_eularian_angles(k.orientation)
    error = horizontal_error_m(k.position.x_val, k.position.y_val, pad.x_val, pad.y_val)
    speed = abs(k.linear_velocity.z_val)
    tilt = tilt_deg(roll, pitch)
    outcome, _ = classify_outcome(pad_present, True, error, speed, tilt, contact_object.startswith("LandingPad"))
    return outcome, f"fout {error:.2f} m   snelheid {speed:.2f} m/s   kanteling {tilt:.1f} gr"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engines", nargs="+", required=True, help="naam@host:poort, maximaal 9")
    parser.add_argument("--auto", type=int, default=0, help="zoveel landingen automatisch, dan stoppen")
    parser.add_argument("--record", help="demovenster opnemen naar dit mp4-bestand")
    args = parser.parse_args()
    auto_total = args.auto
    recorder = Recorder(args.record) if args.record else None

    def show(frame):
        cv2.imshow(WINDOW, frame)
        if recorder:
            recorder.add(frame)

    engines = [Engine(spec) for spec in args.engines[:9]]
    active = engines[0]
    client = connect()
    unlock_frame_rate(client)
    pad_name = client.simListSceneObjects("LandingPad.*")[0]
    home_pose = client.simGetObjectPose(pad_name)
    pad = home_pose.position          # het "GPS-doel" blijft hier, ook als het platform weg is
    boxes = scene_obstacles(client)
    client.simEnableWeather(True)
    client.simSetWeatherParameter(airsim.WeatherParameter.Fog, 0)

    state = {"phase": "idle", "decider": LandingDecider(), "status": ("Druk op spatie voor een landing", WHITE),
             "altitude": 0.0, "theta": ANGLES[0], "pad": True, "fog": False, "angle_index": 0,
             "outcome_until": 0.0}
    frame_times = collections.deque(maxlen=20)
    frame_id, baseline_ns = 0, 0

    client.takeoffAsync().join()
    go_to_start(client, pad, boxes, state["theta"])
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            t_start = time.perf_counter()
            image = get_bottom_image(client)
            pos = client.getMultirotorState().kinematics_estimated.position
            state["altitude"] = pad.z_val - pos.z_val
            best, inference_ms = None, 0.0

            if state["phase"] == "blind":
                # Laatste meters zonder detectie (zoals live_test.py): beeld enkel tonen.
                collision = client.simGetCollisionInfo()
                if collision.has_collided and collision.time_stamp > baseline_ns:
                    outcome, detail = judge_landing(client, last_state, collision.object_name, pad, state["pad"])
                    state["status"] = (f"{outcome}   {detail}", OUTCOME_COLORS[outcome])
                    state["phase"] = "done"
                    print(f"[{active.name}] theta {state['theta']}: {outcome}   {detail}", flush=True)
                else:
                    last_state = client.getMultirotorState()
                    speed = DESCENT_SPEED_MS if state["altitude"] > FINAL_ALTITUDE_M else FINAL_SPEED_MS
                    client.moveByVelocityAsync(HORIZONTAL_GAIN * (pad.x_val - pos.x_val),
                                               HORIZONTAL_GAIN * (pad.y_val - pos.y_val), speed, COMMAND_S)
            else:
                reply = active.client.infer(frame_id, image)
                frame_id += 1
                best = best_detection(reply)
                inference_ms = reply["inference_ms"]
                if state["phase"] == "landing":
                    decision = state["decider"].update(best is not None, best["conf"] if best else 0.0)
                    state["status"] = DECISION_STYLE[decision]
                    if decision == DESCEND:
                        client.moveByVelocityAsync(*descent_velocity(pos, pad), COMMAND_S)
                        if state["altitude"] < BLIND_ALTITUDE_M:
                            state["phase"] = "blind"
                            state["status"] = ("Onder 3 m: blind landen", GREEN)
                            baseline_ns = client.getMultirotorState().timestamp
                            last_state = client.getMultirotorState()
                    else:
                        client.moveByVelocityAsync(0, 0, 0, COMMAND_S)   # HOLD en ABORT: hangen
                        if decision == ABORT:
                            outcome, _ = classify_outcome(state["pad"], descended=False)
                            state["status"] = (f"ABORT -> {outcome}", OUTCOME_COLORS[outcome])
                            state["phase"] = "done"
                            print(f"[{active.name}] theta {state['theta']}: afgebroken -> {outcome}", flush=True)
                elif state["phase"] == "idle":
                    client.moveByVelocityAsync(0, 0, 0, COMMAND_S)

            frame_times.append(time.perf_counter() - t_start)
            fps = len(frame_times) / sum(frame_times)
            show(draw(image, best, state, active, fps, inference_ms))

            key = cv2.waitKey(1) & 0xFF
            if args.auto and state["phase"] in ("idle", "done") and time.time() >= state["outcome_until"]:
                if state["phase"] == "done":
                    args.auto -= 1
                    if args.auto == 0:
                        break
                    state["phase"] = "pause"
                    state["outcome_until"] = time.time() + 3.0     # uitkomst 3 s in beeld laten
                else:
                    key = ord(" ")
            elif args.auto and state["phase"] == "pause" and time.time() >= state["outcome_until"]:
                key = ord(" ")
            if key in (ord("q"), 27):
                break
            if ord("1") <= key <= ord("9") and key - ord("1") < len(engines):
                active = engines[key - ord("1")]
                frame_times.clear()
            elif key == ord("p"):
                state["pad"] = not state["pad"]
                place_pad(client, pad_name, home_pose, state["pad"])
            elif key == ord("f"):
                state["fog"] = not state["fog"]
                client.simSetWeatherParameter(airsim.WeatherParameter.Fog, FOG_STRENGTH if state["fog"] else 0)
            elif key == ord(" "):
                if auto_total:
                    # Automatisch: engines om beurten, laatste landing zonder platform (afbreken tonen).
                    landing_index = state["angle_index"]
                    active = engines[landing_index % len(engines)]
                    want_pad = landing_index < auto_total - 1
                    if want_pad != state["pad"]:
                        state["pad"] = want_pad
                        place_pad(client, pad_name, home_pose, want_pad)
                state["theta"] = ANGLES[state["angle_index"] % len(ANGLES)]
                state["angle_index"] += 1
                state["status"] = (f"Naar de start: theta {state['theta']} graden, 20 m", WHITE)
                show(draw(image, None, state, active, fps, 0.0))
                cv2.waitKey(1)
                go_to_start(client, pad, boxes, state["theta"])
                state["decider"] = LandingDecider()
                state["phase"] = "landing"
                frame_times.clear()
    finally:
        cv2.destroyAllWindows()
        if recorder:
            recorder.close()
            print(f"Video opgeslagen: {args.record}")
        client.simSetWeatherParameter(airsim.WeatherParameter.Fog, 0)
        client.simEnableWeather(False)
        place_pad(client, pad_name, home_pose, True)
        for engine in engines:
            engine.client.close()


if __name__ == "__main__":
    main()
