#!/usr/bin/env bash
# Start (of stop) de inferentieservers voor de demo op de Jetson: één per engine, poort 5000, 5001, ...
# Zo zijn alle engines al geladen en gaat wisselen in landing/demo.py meteen.
#
# Gebruik (op de pc, in Git Bash, vanuit de projectmap):
#   bash landing/demo_servers.sh start yolo26s_fp16 rtdetr-l_fp32
#   bash landing/demo_servers.sh stop
set -u
# Gebruiker en adres van de Jetson staan in jetson.env (niet in Git), zodat ze niet publiek komen.
[ -f jetson.env ] && . ./jetson.env
JETSON=${JETSON:-user@192.168.55.1}
ACTION="${1:-}"
shift || true

# Altijd eerst de servers van een vorige demo (en van live_all.sh) stoppen: die houden poorten bezet.
ssh "$JETSON" 'docker rm -f $(docker ps -aq --filter "name=landing-project-demo") landing-project-server >/dev/null 2>&1'
[[ "$ACTION" == "stop" ]] && { echo "Servers gestopt"; exit 0; }
[[ "$ACTION" != "start" || $# -eq 0 ]] && { echo "Gebruik: $0 start <engine> [<engine> ...] | stop"; exit 1; }

port=5000
specs=()
for engine in "$@"; do
  ssh "$JETSON" "cd ~/landing-project && DETACH=1 NAME=landing-project-demo-$port jetson/run_in_docker.sh \
    python3 jetson/server.py --model engines/${engine}.engine --port $port >/dev/null"
  specs+=("${engine}@192.168.55.1:${port}")
  port=$((port + 1))
done

# Wachten tot elke server zijn engine geladen heeft (max. 3 minuten per server).
port=5000
for engine in "$@"; do
  for _ in $(seq 90); do
    ssh "$JETSON" "docker logs landing-project-demo-$port 2>&1" | grep -q "Wacht op verbinding" && break
    sleep 2
  done
  ssh "$JETSON" "docker logs landing-project-demo-$port 2>&1" | grep -q "Wacht op verbinding" \
    && echo "Klaar: $engine op poort $port" || echo "FOUT: $engine start niet (docker logs landing-project-demo-$port)"
  port=$((port + 1))
done
echo
echo "Start de demo (PowerShell, airsim-env, Unreal op Play):"
echo "  python -m landing.demo --engines ${specs[*]}"
