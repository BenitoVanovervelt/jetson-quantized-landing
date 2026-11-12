"""TCP-server: ontvangt JPEG-beelden, detecteert het platform en stuurt het resultaat terug.

Draait op de Jetson met een TensorRT-engine. Voor ontwikkeling kan hij ook op
de pc draaien met een .pt-model (plan B uit het overzicht): de client merkt
geen verschil.

Gebruik:
    python jetson/server.py --model engines/yolo26s_fp16.engine
    python jetson/server.py --model yolo26s.pt --host 127.0.0.1      # test op de pc

Eén client tegelijk: de keten heeft maar één drone, en zo deelt niets de GPU.
"""
import argparse
import socket
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detector import Detector                                  # noqa: E402
from protocol import (DEFAULT_PORT, configure_socket, recv_frame,  # noqa: E402
                      send_json)


def serve_client(conn, detector):
    while True:
        try:
            frame_id, jpeg = recv_frame(conn)
        except ConnectionError:
            return
        t_start = time.perf_counter()
        image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        t_decoded = time.perf_counter()
        detections, timings = detector.predict(image)
        t_end = time.perf_counter()

        send_json(conn, {
            "frame_id": frame_id,
            "detections": detections,
            "decode_ms": (t_decoded - t_start) * 1000,
            **timings,
            # Alles wat de server deed, van ontvangen beeld tot klaar antwoord.
            "server_ms": (t_end - t_start) * 1000,
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help=".engine, .onnx of .pt")
    parser.add_argument("--host", default="0.0.0.0", help="0.0.0.0 = luister op alle netwerkkaarten")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--conf", type=float, default=0.25,
                        help="minimale confidence om een detectie terug te sturen; "
                             "de beslissingsdrempel zelf ligt op de pc")
    args = parser.parse_args()

    print(f"Model laden: {args.model}")
    detector = Detector(args.model, conf=args.conf)
    detector.warmup()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(1)
        print(f"Wacht op verbinding op {args.host}:{args.port} (Ctrl+C om te stoppen)")
        while True:
            conn, address = server.accept()
            configure_socket(conn)
            print(f"Verbonden met {address[0]}")
            with conn:
                serve_client(conn, detector)
            print("Verbinding gesloten, wacht op de volgende")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Gestopt")
