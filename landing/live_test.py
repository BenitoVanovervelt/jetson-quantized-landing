"""Live landingen: per engine 50 pogingen, met uitkomst per poging (deelvragen 7 en 8).

Verloop van één poging:
  1. Platform op zijn plaats zetten, of wegzetten (pogingen zonder platform).
  2. Drone teleporteren naar de startpositie (theta, d = 20 m, phi) en laten hangen.
  3. Elke frame: beeld -> Jetson -> LandingDecider.
       HOLD     hoogte houden (wachten op 5 opeenvolgende detecties)
       DESCEND  vliegen op positie ("GPS"): eerst horizontaal tot boven het platform, dan
                verticaal dalen, telkens 1 m/s
       ABORT    stoppen en hangen: poging afgebroken
  4. Onder 3 m boven het platform: blind verticaal landen (1 m/s, de laatste meter 0,3 m/s).
     Getest: onder ± 2 m valt de schaduw van de drone over het platform en zakt de confidence
     (results/live/close_range_view.jpg); het model zag ook nooit beelden van dichter dan 4 m.
     Echte precisielandingen schakelen in de laatste meters ook over op een andere sensor.
  5. Bij contact (simGetCollisionInfo): positiefout, landingssnelheid en kanteling -> uitkomst.

De detectie beslist dus óf de drone daalt; de positie (zoals GPS) bepaalt waar hij heen vliegt.
Zo staat het in het contractplan.

Gebruik (vanuit de projectmap, in airsim-env; Unreal op Play, server draait op de Jetson):
    python -m landing.live_test --host 192.168.55.1 --model yolo26s --precision fp16
    python -m landing.live_test ... --limit 5          # enkel de eerste 5 pogingen (test)
    python -m landing.live_test ... --only 34 35       # enkel deze pogingen naspelen

Resultaat in results/live/: live_<model>_<prec>_frames.csv (één rij per frame) en
live_<model>_<prec>_attempts.csv (één rij per poging, met de uitkomst).
"""
import argparse
import math
import random
import time
from pathlib import Path

import airsim

from landing.csv_log import CsvLogger
from landing.decision import ABORT, DESCEND, HOLD, LandingDecider
from landing.inference_client import InferenceClient, best_detection
from landing.outcome import classify_outcome, horizontal_error_m, tilt_deg
from sim.airsim_utils import connect, get_bottom_image, teleport, unlock_frame_rate

PROJECT_DIR = Path(__file__).resolve().parent.parent

START_DISTANCE_M = 20.0     # afstand is geen startvariabele: elke landing daalt door alle afstanden
BLIND_ALTITUDE_M = 3.0      # hieronder landen zonder detectie (zie stap 4 hierboven)
FINAL_ALTITUDE_M = 1.0      # hieronder trager dalen
DESCENT_SPEED_MS = 1.0      # v uit de FPS-eis (FPS >= N * v / s_max)
FINAL_SPEED_MS = 0.3        # laatste meter: onder de grens van 0,5 m/s voor een goede landing
HORIZONTAL_GAIN = 1.0       # P-regelaar horizontaal: 1 m/s per meter fout (afremmen boven het platform)
ABOVE_PAD_M = 0.2           # horizontaal binnen deze afstand = boven het platform: verticaal dalen
COMMAND_S = 0.5             # elk snelheidscommando loopt hoogstens zo lang (veilig als de lus hapert)
MAX_START_DRIFT_M = 0.5     # startpositie mag na 1 s hangen hoogstens zoveel afwijken
FROZEN_FRAMES = 40          # zoveel frames exact dezelfde positie tijdens het dalen = vastgelopen
MAX_FRAMES = 1500          # veiligheidsgrens per poging (± 75 s bij 20 FPS)
PAD_HIDDEN_OFFSET_M = 500.0  # pogingen zonder platform: platform ver buiten beeld zetten
OBSTACLE_MARGIN_M = 1.0     # vrije ruimte rond de blokken van de scène (drone ± 0,5 m breed)
PHI_STEP_DEG = 10.0         # startrichting zoveel verdraaien als de route door een blok gaat


def plan_attempts(angles, per_angle, no_pad, seed):
    """Vaste lijst van pogingen; dezelfde seed geeft voor elke engine exact dezelfde pogingen."""
    rng = random.Random(seed)
    attempts = [(theta, rng.uniform(0, 360), True) for theta in angles for _ in range(per_angle)]
    attempts += [(rng.choice(angles), rng.uniform(0, 360), False) for _ in range(no_pad)]
    return attempts


def start_position(pad, theta_deg, phi_deg):
    """Positie t.o.v. het platform zoals in het contractplan (NED)."""
    theta, phi = math.radians(theta_deg), math.radians(phi_deg)
    return (pad.x_val + START_DISTANCE_M * math.sin(theta) * math.cos(phi),
            pad.y_val + START_DISTANCE_M * math.sin(theta) * math.sin(phi),
            pad.z_val - START_DISTANCE_M * math.cos(theta))


def scene_obstacles(client):
    """Blokken van de Blocks-scène als (xmin, xmax, ymin, ymax, zmin, zmax) in NED.

    De blokken zijn kubussen van 1 m met het draaipunt in het midden, geschaald per as.
    """
    boxes = []
    for name in client.simListSceneObjects("TemplateCube.*"):
        p, s = client.simGetObjectPose(name).position, client.simGetObjectScale(name)
        boxes.append((p.x_val - s.x_val / 2, p.x_val + s.x_val / 2, p.y_val - s.y_val / 2,
                      p.y_val + s.y_val / 2, p.z_val - s.z_val / 2, p.z_val + s.z_val / 2))
    return boxes


def route_is_clear(boxes, start, pad):
    """Startpositie en horizontale vlucht tot boven het platform raken geen blok."""
    m = OBSTACLE_MARGIN_M
    for k in range(101):
        t = k / 100
        x, y, z = start[0] + (pad.x_val - start[0]) * t, start[1] + (pad.y_val - start[1]) * t, start[2]
        for x0, x1, y0, y1, z0, z1 in boxes:
            if x0 - m < x < x1 + m and y0 - m < y < y1 + m and z0 - m < z < z1 + m:
                return False
    return True


def clear_phi(boxes, pad, theta, phi):
    """Eerste vrije startrichting vanaf phi, in stappen van PHI_STEP_DEG.

    Bij 45° (14 m hoog) lag één startpositie in een toren van de scène: de drone zat vast in
    het blok. Enkel die poging krijgt zo een andere richting; de rest van de lijst blijft gelijk.
    """
    for k in range(int(360 / PHI_STEP_DEG)):
        candidate = (phi + k * PHI_STEP_DEG) % 360
        if route_is_clear(boxes, start_position(pad, theta, candidate), pad):
            return candidate
    raise RuntimeError(f"Geen vrije startrichting voor theta = {theta}")


def descent_velocity(pos, pad):
    """Snelheid (NED) naar het platform, op basis van de positie (zoals GPS).

    Eerst horizontaal op dezelfde hoogte tot boven het platform, dan verticaal dalen.
    Getest: in een rechte schuine lijn daalt de drone in het begin sneller dan hij vooruit
    komt, waardoor de hoek naar het platform bij theta = 45° even boven 45° gaat en het
    platform uit beeld schuift. Horizontaal vliegen maakt de hoek meteen kleiner.
    Echte precisielandingen werken ook zo: met GPS tot boven het platform, dan dalen.
    """
    dx, dy = pad.x_val - pos.x_val, pad.y_val - pos.y_val
    distance = math.hypot(dx, dy)
    if distance > ABOVE_PAD_M:
        speed = min(DESCENT_SPEED_MS, HORIZONTAL_GAIN * distance)   # afremmen vlak boven het platform
        return dx / distance * speed, dy / distance * speed, 0.0
    return HORIZONTAL_GAIN * dx, HORIZONTAL_GAIN * dy, DESCENT_SPEED_MS


def place_pad(client, pad_name, home_pose, present):
    pose = airsim.Pose(airsim.Vector3r(home_pose.position.x_val, home_pose.position.y_val,
                                       home_pose.position.z_val), home_pose.orientation)
    if not present:
        pose.position.x_val += PAD_HIDDEN_OFFSET_M
    client.simSetObjectPose(pad_name, pose, True)


def blind_landing(client, pad, baseline_ns, timeout_s=15.0):
    """Laatste meters zonder detectie. Geeft de toestand vlak voor het contact terug, of None."""
    last = None
    t_end = time.time() + timeout_s
    while time.time() < t_end:
        state = client.getMultirotorState()
        collision = client.simGetCollisionInfo()
        if collision.has_collided and collision.time_stamp > baseline_ns:
            return last or state, collision.object_name
        last = state
        pos = state.kinematics_estimated.position
        speed = DESCENT_SPEED_MS if pad.z_val - pos.z_val > FINAL_ALTITUDE_M else FINAL_SPEED_MS
        client.moveByVelocityAsync(HORIZONTAL_GAIN * (pad.x_val - pos.x_val),
                                   HORIZONTAL_GAIN * (pad.y_val - pos.y_val), speed, COMMAND_S)
        time.sleep(0.01)
    return None, ""


def run_attempt(client, inference, frame_log, attempt_id, theta, phi, pad_present, pad, common):
    """Eén landingspoging. Geeft de rij voor de pogingen-CSV terug."""
    # Vlak na een landing bleef de drone soms aan de grond "vastzitten": het teleporteren
    # lukte, maar zodra de simulatie liep stond hij terug op het platform. Daarom de echte
    # positie controleren na 1 s hangen, en anders eerst opstijgen en opnieuw proberen.
    target = start_position(pad, theta, phi)
    for _ in range(3):
        teleport(client, target)
        client.moveByVelocityAsync(0, 0, 0, 1.0)
        time.sleep(1.0)   # even laten stabiliseren na het teleporteren
        actual = client.simGetVehiclePose().position
        drift = math.dist(target, (actual.x_val, actual.y_val, actual.z_val))
        if drift < MAX_START_DRIFT_M:
            break
        client.moveByVelocityAsync(0, 0, -1.0, 1.0).join()   # loskomen van de grond
    else:
        raise RuntimeError(f"Drone niet op de startpositie na 3 pogingen (afwijking {drift:.2f} m)")

    decider = LandingDecider()
    decision = HOLD
    frozen, last_position = 0, None
    row = dict(common, attempt_id=attempt_id, theta_deg=theta, distance_m=START_DISTANCE_M,
               phi_deg=f"{phi:.1f}", pad_present=int(pad_present))
    for frame_id in range(MAX_FRAMES):
        t_start = time.perf_counter()
        image = get_bottom_image(client)
        t_captured = time.perf_counter()
        reply = inference.infer(frame_id, image)
        best = best_detection(reply)
        decision = decider.update(best is not None, best["conf"] if best else 0.0)

        pos = client.getMultirotorState().kinematics_estimated.position
        altitude = pad.z_val - pos.z_val
        if decision == DESCEND:
            client.moveByVelocityAsync(*descent_velocity(pos, pad), COMMAND_S)
        else:
            client.moveByVelocityAsync(0, 0, 0, COMMAND_S)   # HOLD en ABORT: hangen
        t_end = time.perf_counter()

        ms = (t_end - t_start) * 1000
        box = best["box"] if best else ["", "", "", ""]
        frame_log.log(**row, frame_id=frame_id, detected=int(best is not None),
                      confidence=best["conf"] if best else "",
                      box_x1=box[0], box_y1=box[1], box_x2=box[2], box_y2=box[3],
                      capture_ms=f"{(t_captured - t_start) * 1000:.2f}",
                      inference_ms=f"{reply['inference_ms']:.2f}", server_ms=f"{reply['server_ms']:.2f}",
                      network_ms=f"{reply['network_ms']:.2f}", frame_ms=f"{ms:.2f}", fps=f"{1000 / ms:.2f}",
                      decision=decision, drone_x=f"{pos.x_val:.3f}", drone_y=f"{pos.y_val:.3f}",
                      drone_z=f"{pos.z_val:.3f}", altitude_m=f"{altitude:.3f}")
        if decision == ABORT or (decision == DESCEND and altitude < BLIND_ALTITUDE_M):
            break
        # Vangnet: een drone die moet dalen maar exact stilstaat, wijst op een vastgelopen
        # simulatie (geen meetresultaat). Dan stoppen in plaats van een uitkomst te loggen.
        position = (pos.x_val, pos.y_val, pos.z_val)
        frozen = frozen + 1 if decision == DESCEND and position == last_position else 0
        last_position = position
        if frozen >= FROZEN_FRAMES:
            raise RuntimeError(f"Drone staat {FROZEN_FRAMES} frames stil tijdens het dalen: simulatie vastgelopen?")
    frames = frame_id + 1

    contact_state, contact_object = None, ""
    if decision == DESCEND:
        baseline_ns = client.getMultirotorState().timestamp
        contact_state, contact_object = blind_landing(client, pad, baseline_ns)

    if contact_state is None:
        # Afgebroken (of veiligheidsgrens bereikt / geen contact): niet geland.
        outcome, reasons = classify_outcome(pad_present, descended=False)
        row.update(outcome=outcome, outcome_reasons="", decision=decision, frame_id=frames)
        if decision != ABORT:
            row["outcome_reasons"] = "geen_contact" if decision == DESCEND else "max_frames"
        return row, drift, contact_object

    k = contact_state.kinematics_estimated
    roll, pitch, _ = airsim.to_eularian_angles(k.orientation)
    error = horizontal_error_m(k.position.x_val, k.position.y_val, pad.x_val, pad.y_val)
    speed = abs(k.linear_velocity.z_val)
    tilt = tilt_deg(roll, pitch)
    on_pad = contact_object.startswith("LandingPad")
    outcome, reasons = classify_outcome(pad_present, True, error, speed, tilt, on_pad)
    row.update(outcome=outcome, outcome_reasons=" ".join(reasons), decision="landed", frame_id=frames,
               drone_x=f"{k.position.x_val:.3f}", drone_y=f"{k.position.y_val:.3f}",
               drone_z=f"{k.position.z_val:.3f}", position_error_m=f"{error:.3f}",
               landing_speed_ms=f"{speed:.3f}", tilt_deg=f"{tilt:.2f}")
    return row, drift, contact_object


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--model", required=True, help="naam voor de log, bv. yolo26s of rtdetr-l")
    parser.add_argument("--precision", required=True, help="rekenprecisie voor de log: fp32, fp16, int8")
    parser.add_argument("--angles", type=float, nargs="+", default=[0, 15, 30, 45])
    parser.add_argument("--per-angle", type=int, default=10)
    parser.add_argument("--no-pad", type=int, default=10, help="pogingen zonder platform")
    parser.add_argument("--seed", type=int, default=0, help="zelfde seed = zelfde pogingen voor elke engine")
    parser.add_argument("--limit", type=int, help="enkel de eerste N pogingen (om te testen)")
    parser.add_argument("--only", type=int, nargs="+", help="enkel deze poging-id's (om één poging na te spelen)")
    args = parser.parse_args()

    client = connect()
    unlock_frame_rate(client)
    pad_name = client.simListSceneObjects("LandingPad.*")[0]
    home_pose = client.simGetObjectPose(pad_name)
    pad = home_pose.position     # het "GPS-doel" blijft hier, ook als het platform weg is
    inference = InferenceClient(args.host, args.port)

    attempts = plan_attempts(args.angles, args.per_angle, args.no_pad, args.seed)[:args.limit]
    boxes = scene_obstacles(client)
    for i, (theta, phi, pad_present) in enumerate(attempts):
        free_phi = clear_phi(boxes, pad, theta, phi)
        if free_phi != phi:
            print(f"Poging {i}: phi {phi:.1f} -> {free_phi:.1f} (route door een blok van de scène)")
            attempts[i] = (theta, free_phi, pad_present)
    out_dir = PROJECT_DIR / "results" / "live"
    stem = f"live_{args.model}_{args.precision}_{time.strftime('%Y%m%d_%H%M%S')}"
    common = {"model": args.model, "precision_mode": args.precision}
    print(f"{len(attempts)} pogingen, log: {out_dir / stem}_*.csv")

    client.takeoffAsync().join()
    try:
        with CsvLogger(out_dir / f"{stem}_frames.csv") as frame_log, \
                CsvLogger(out_dir / f"{stem}_attempts.csv") as attempt_log:
            for attempt_id, (theta, phi, pad_present) in enumerate(attempts):
                if args.only and attempt_id not in args.only:
                    continue
                place_pad(client, pad_name, home_pose, pad_present)
                t_start = time.time()
                row, drift, contact = run_attempt(client, inference, frame_log, attempt_id,
                                                  theta, phi, pad_present, pad, common)
                attempt_log.log(**row)
                print(f"#{attempt_id:2d} theta={theta:4.0f} phi={phi:5.1f} platform={'ja ' if pad_present else 'nee'} "
                      f"-> {row['outcome']:20s} {row.get('outcome_reasons', '')} "
                      f"fout={row.get('position_error_m', '-')} v={row.get('landing_speed_ms', '-')} "
                      f"kant={row.get('tilt_deg', '-')} contact={contact or '-'} "
                      f"({row['frame_id']} frames, {time.time() - t_start:.0f} s, startafwijking {drift * 100:.0f} cm)")
    finally:
        place_pad(client, pad_name, home_pose, True)
        inference.close()


if __name__ == "__main__":
    main()
