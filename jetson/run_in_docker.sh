#!/usr/bin/env bash
# Voer een commando uit in de Ultralytics-container op de Jetson.
#
# Waarom Docker: de Jetson is gedeeld (robotsoftware van het AI Lab). In de container
# zitten PyTorch, TensorRT en Ultralytics, zonder iets op het systeem zelf te installeren.
#
# Gebruik (op de Jetson, vanuit ~/landing-project):
#   jetson/run_in_docker.sh python3 jetson/build_engines.py --weights models/yolo26s.pt ...
#   DETACH=1 NAME=landing-project-build jetson/run_in_docker.sh bash -c "..."   # op de achtergrond
#
# Keuzes:
# - Vaste versie van het image (geen 'latest'): zelfde Ultralytics als op de pc, reproduceerbaar.
# - --user: bestanden die de container maakt, zijn van de gewone gebruiker, niet van root.
#   HOME en YOLO_CONFIG_DIR wijzen dan naar /tmp, omdat /root niet beschrijfbaar is.
#   USER moet gezet zijn: de container kent de gebruiker niet, en PyTorch vraagt de naam op.
# - --network host: de inferentieserver is dan rechtstreeks bereikbaar op poort 5000.
# - POWER_MODE: nvpmodel bestaat niet in de container; de modus wordt mee gelogd.
set -euo pipefail

IMAGE="ultralytics/ultralytics:8.4.182-jetson-jetpack6"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
POWER_MODE="$(nvpmodel -q 2>/dev/null | head -1 || echo onbekend)"

MODE_ARGS=(--rm -it)
if [[ "${DETACH:-0}" == "1" ]]; then
  MODE_ARGS=(-d)
fi

docker run "${MODE_ARGS[@]}" \
  --name "${NAME:-landing-project}" \
  --runtime nvidia --ipc=host --network host \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e USER="$(id -un)" -e YOLO_CONFIG_DIR=/tmp/ultralytics -e POWER_MODE="$POWER_MODE" \
  -v "$PROJECT_DIR":/workspace/landing-project -w /workspace/landing-project \
  "$IMAGE" "$@"
