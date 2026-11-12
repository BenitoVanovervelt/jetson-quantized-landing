"""Client voor de inferentieserver (jetson/server.py).

Stuurt één beeld als JPEG en meet de round trip. Wordt gebruikt door de
ketentest en later door de live landingen, zodat beide op dezelfde manier meten.
"""
import socket
import time

import cv2

from jetson.protocol import DEFAULT_PORT, configure_socket, recv_json, send_frame

JPEG_QUALITY = 90   # zelfde kwaliteit als de dataset, zodat het model hetzelfde soort beeld ziet


class InferenceClient:
    def __init__(self, host, port=DEFAULT_PORT, timeout_s=5.0):
        self.sock = socket.create_connection((host, port), timeout=timeout_s)
        configure_socket(self.sock)

    def infer(self, frame_id, image_bgr):
        """Geef het antwoord van de server terug, aangevuld met de tijden aan pc-kant (ms).

        encode_ms   JPEG-codering op de pc
        rtt_ms      versturen tot antwoord binnen
        network_ms  rtt_ms min de tijd die de server zelf werkte
        """
        t_start = time.perf_counter()
        ok, jpeg = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            raise RuntimeError("JPEG-codering mislukt")
        t_encoded = time.perf_counter()
        send_frame(self.sock, frame_id, jpeg.tobytes())
        reply = recv_json(self.sock)
        t_end = time.perf_counter()

        if reply["frame_id"] != frame_id:
            raise RuntimeError(f"Antwoord voor frame {reply['frame_id']}, verwacht {frame_id}")
        reply["encode_ms"] = (t_encoded - t_start) * 1000
        reply["rtt_ms"] = (t_end - t_encoded) * 1000
        reply["network_ms"] = reply["rtt_ms"] - reply["server_ms"]
        reply["jpeg_bytes"] = len(jpeg)
        return reply

    def close(self):
        self.sock.close()


def best_detection(reply):
    """De detectie met de hoogste confidence, of None. Er is maar één platform."""
    detections = reply["detections"]
    return max(detections, key=lambda d: d["conf"]) if detections else None
