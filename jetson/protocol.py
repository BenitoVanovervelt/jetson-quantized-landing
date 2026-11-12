"""Berichtformaat tussen pc (client) en Jetson (server) over TCP.

TCP is een stroom van bytes zonder grenzen tussen berichten. Daarom krijgt elk
deel een lengte van 4 bytes vooraf (big-endian), zodat de ontvanger weet hoeveel
bytes bij één bericht horen.

  pc -> Jetson:  [lengte][JSON-header]  [lengte][JPEG-bytes]
                 header = {"frame_id": int}
  Jetson -> pc:  [lengte][JSON-antwoord]
                 antwoord = {"frame_id", "detections": [{"box": [x1, y1, x2, y2], "conf"}],
                             "decode_ms", "preprocess_ms", "inference_ms", "postprocess_ms", "server_ms"}

De netwerktijd wordt op de pc berekend: round trip min server_ms. Zo zijn er
geen gesynchroniseerde klokken tussen pc en Jetson nodig.

Werkt met Python 3.8+ (pc en Jetson), alleen standaardbibliotheek.
"""
import json
import socket
import struct

DEFAULT_PORT = 5000
_LENGTH = struct.Struct(">I")   # 4 bytes, unsigned, big-endian


def configure_socket(sock):
    """Schakel Nagle uit: kleine berichten worden dan meteen verstuurd i.p.v. gebundeld.

    Met Nagle aan kan een klein antwoord tientallen ms blijven wachten, wat de
    gemeten netwerktijd vervalst.
    """
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


def _recv_exact(sock, size):
    buffer = bytearray()
    while len(buffer) < size:
        chunk = sock.recv(size - len(buffer))
        if not chunk:
            raise ConnectionError("Verbinding gesloten door de andere kant")
        buffer.extend(chunk)
    return bytes(buffer)


def send_blob(sock, data):
    sock.sendall(_LENGTH.pack(len(data)) + data)


def recv_blob(sock):
    (size,) = _LENGTH.unpack(_recv_exact(sock, _LENGTH.size))
    return _recv_exact(sock, size)


def send_json(sock, obj):
    send_blob(sock, json.dumps(obj).encode("utf-8"))


def recv_json(sock):
    return json.loads(recv_blob(sock).decode("utf-8"))


def send_frame(sock, frame_id, jpeg_bytes):
    send_json(sock, {"frame_id": frame_id})
    send_blob(sock, jpeg_bytes)


def recv_frame(sock):
    header = recv_json(sock)
    return header["frame_id"], recv_blob(sock)
