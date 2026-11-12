#!/usr/bin/env bash
# Ketentest voor meerdere engines na elkaar (op de pc, in Git Bash, vanuit de projectmap).
# Per engine: server op de Jetson starten, wachten tot hij luistert, 500 frames meten, server stoppen.
#
# Gebruik:  bash landing/chain_all.sh yolo26s_fp32 yolo26s_fp16 rtdetr-l_int8 ...
# Vereist: Unreal op Play, Jetson via USB-C, SSH-sleutel voor de Jetson (gebruiker in jetson.env, zie docs/INSTALL.md).
set -u
# Gebruiker en adres van de Jetson staan in jetson.env (niet in Git), zodat ze niet publiek komen.
[ -f jetson.env ] && . ./jetson.env
JETSON=${JETSON:-user@192.168.55.1}
HOST=192.168.55.1

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
  airsim-env/Scripts/python.exe -m landing.chain_test --host "$HOST" \
    --model "$model" --precision "$precision" --source airsim --frames 500 --warmup 20
done
ssh "$JETSON" "docker rm -f landing-project-server >/dev/null 2>&1"
echo "Klaar"
