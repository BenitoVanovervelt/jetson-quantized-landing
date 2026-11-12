#!/usr/bin/env bash
# Offline evaluatie van alle engines op de testset, elke engine in een eigen Python-proces
# (zelfde reden als build_all_engines.sh: gelijke hoeveelheid vrij geheugen per engine).
#
# Gebruik (op de Jetson, vanuit ~/landing-project; draait in de container):
#   DETACH=1 NAME=landing-project-eval jetson/run_in_docker.sh bash jetson/evaluate_all.sh
set -u

mkdir -p results/offline
for engine in engines/*.engine; do
  precision="$(basename "$engine" .engine)"
  precision="${precision##*_}"
  python3 jetson/evaluate.py --model "$engine" --precision "$precision" \
    >> results/offline/evaluate.log 2>&1
done
echo "Klaar" >> results/offline/evaluate_done.txt
