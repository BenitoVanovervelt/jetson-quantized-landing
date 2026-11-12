#!/usr/bin/env bash
# Live landingen voor meerdere engines na elkaar (op de pc, in Git Bash, vanuit de projectmap).
# Per engine: server op de Jetson starten, wachten tot hij luistert, 50 landingen, server stoppen.
#
# Gebruik:  bash landing/live_all.sh yolo26s_fp32 yolo26s_fp16 rtdetr-l_int8 ...
# Vereist: Unreal op Play, Jetson via USB-C, SSH-sleutel voor de Jetson (gebruiker in jetson.env, zie docs/INSTALL.md).
set -u
# Gebruiker en adres van de Jetson staan in jetson.env (niet in Git), zodat ze niet publiek komen.
[ -f jetson.env ] && . ./jetson.env
JETSON=${JETSON:-user@192.168.55.1}
HOST=192.168.55.1
LOCK=results/live/live_all.lock

# Nooit twee reeksen tegelijk: elke reeks herstart de server op de Jetson en zou de
# verbinding van de andere afbreken (zo mislukten eerder drie YOLO-runs).
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "FOUT: er loopt al een reeks ($LOCK bestaat). Verwijder de map als dat niet klopt."
  exit 1
fi
trap 'rmdir "$LOCK"' EXIT

for engine in "$@"; do
  model="${engine%_*}"
  precision="${engine##*_}"
  echo "=== $engine ==="
  ssh "$JETSON" "docker rm -f landing-project-server >/dev/null 2>&1; cd ~/landing-project && \
    DETACH=1 NAME=landing-project-server jetson/run_in_docker.sh \
    python3 jetson/server.py --model engines/${engine}.engine >/dev/null"
  # Wachten tot de engine geladen is en de server luistert (max. 3 minuten).
  for _ in $(seq 90); do
    ssh "$JETSON" "docker logs landing-project-server 2>&1" | grep -q "Wacht op verbinding" && break
    sleep 2
  done
  airsim-env/Scripts/python.exe -m landing.live_test --host "$HOST" --model "$model" --precision "$precision"
done
ssh "$JETSON" "docker rm -f landing-project-server >/dev/null 2>&1"
echo "Klaar"
