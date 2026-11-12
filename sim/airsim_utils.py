"""Gedeelde hulpfuncties voor de AirSim-scripts in deze map."""
import math
import socket
import sys
import time

import airsim
import numpy as np

AIRSIM_HOST = "127.0.0.1"
AIRSIM_PORT = 41451          # standaardpoort van de AirSim RPC-server
RPC_TIMEOUT_S = 60           # standaard is 3600 s: dan hangt een script een uur
CAMERA_NAME = "bottom_cam"


def check_simulator_running():
    """Stop met een duidelijke melding als AirSim niet luistert.

    De airsim-client blijft anders lang wachten. Een gewone TCP-verbinding
    met een korte timeout zegt meteen of Unreal op Play staat.
    """
    try:
        with socket.create_connection((AIRSIM_HOST, AIRSIM_PORT), timeout=2):
            pass
    except OSError:
        sys.exit(
            f"FOUT: geen AirSim-simulator gevonden op {AIRSIM_HOST}:{AIRSIM_PORT}.\n"
            "Open Blocks.uproject in Unreal en druk op Play, en voer dit script daarna opnieuw uit."
        )


def connect():
    """Controleer de simulator, verbind en neem de besturing over."""
    check_simulator_running()
    client = airsim.MultirotorClient(timeout_value=RPC_TIMEOUT_S)
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    return client


def unlock_frame_rate(client):
    """Haal de framelimiet van de Unreal-editor weg (geldt tot de volgende Play).

    simGetImages wacht op het volgende gerenderde frame. De editor stond op ± 30 FPS,
    waardoor elk beeld ± 33 ms kostte; zonder limiet ± 24 ms. Een echte camera op een
    drone heeft die limiet niet, dus zou ze de ketenmeting onterecht vertragen.
    """
    for command in ["t.MaxFPS 120", "r.VSync 0"]:
        client.simRunConsoleCommand(command)


def teleport(client, position, yaw_deg=0.0, max_drift_m=0.01, max_tries=5):
    """Zet de drone stil op een positie (NED, meter) en laat de simulatie daarna weer lopen.

    Zelfde methode als bij de datageneratie (zie data_gen/generate_dataset.py, capture):
    simSetKinematics zet positie én snelheid 0, in een gepauzeerde simulatie, en een stap
    van 10 ms past de positie toe. simSetVehiclePose zou de valsnelheid laten staan.
    Geeft de afwijking tot de doelpositie terug (m).
    """
    state = airsim.KinematicsState()   # snelheden en versnellingen standaard 0
    state.position = airsim.Vector3r(*position)
    state.orientation = airsim.to_quaternion(0, 0, math.radians(yaw_deg))
    client.simPause(True)
    for _ in range(max_tries):
        client.simSetKinematics(state, True)
        # Wachten tot de stap van 10 ms echt gebeurd is. Enkel wachten op simIsPause() volstaat
        # niet: de simulatie is vóór de stap al gepauzeerd, dus de lus stopte soms te vroeg.
        # Dan pauzeerde AirSim na simPause(False) alsnog, en hing de drone stil (live landingen).
        t_before = client.getMultirotorState().timestamp
        client.simContinueForTime(0.01)
        while client.getMultirotorState().timestamp == t_before or not client.simIsPause():
            time.sleep(0.005)
        actual = client.simGetVehiclePose().position
        drift = math.dist(position, (actual.x_val, actual.y_val, actual.z_val))
        if drift <= max_drift_m:
            break
    client.simPause(False)
    # Controleren dat de simulatie weer loopt.
    t_before = client.getMultirotorState().timestamp
    while client.getMultirotorState().timestamp == t_before:
        if client.simIsPause():
            client.simPause(False)
        time.sleep(0.005)
    return drift


def get_bottom_image(client):
    """Haal één onbewerkt (BGR) beeld op van de onderste camera."""
    request = airsim.ImageRequest(CAMERA_NAME, airsim.ImageType.Scene, False, False)
    # AirSim geeft heel af en toe één leeg beeld terug, ook als alles klopt (gezien tijdens
    # de live landingen). Daarom een paar keer opnieuw proberen voor we stoppen.
    for _ in range(5):
        response = client.simGetImages([request])[0]
        if response.width > 0 and response.height > 0:
            break

    # Een onbekende camera geeft altijd een leeg beeld (0 x 0) in plaats van een fout.
    if response.width == 0 or response.height == 0:
        sys.exit(
            f"FOUT: camera '{CAMERA_NAME}' gaf een leeg beeld.\n"
            "Controleer settings.json en start Play opnieuw (instellingen gelden pas na een nieuwe Play)."
        )

    # frombuffer geeft een alleen-lezen array (gedeeld geheugen met de buffer van AirSim).
    # Met .copy() kan OpenCV er later op tekenen.
    image = np.frombuffer(response.image_data_uint8, dtype=np.uint8).copy()
    return image.reshape(response.height, response.width, 3)


def land_and_release(client):
    """Netjes landen, zodat een script opnieuw kan draaien zonder Play te herstarten."""
    print("Landen...")
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)
