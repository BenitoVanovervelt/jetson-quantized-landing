#!/usr/bin/env bash
# Evaluatie van alle engines op de moeilijkere testsets (data/hard/<conditie>, gemaakt met
# data_gen/generate_hard_testset.py). Enkel totaal en per afstand: zo blijft de looptijd
# beperkt (± 2000 beelden per engine en conditie i.p.v. ± 4000).
# Elke engine in een eigen Python-proces (zelfde reden als evaluate_all.sh).
#
# Gebruik (op de Jetson, vanuit ~/landing-project; draait in de container):
#   DETACH=1 NAME=landing-project-hard jetson/run_in_docker.sh bash jetson/evaluate_hard_all.sh
#   ... bash jetson/evaluate_hard_all.sh fog_10 dust_05     # enkel deze condities
set -u

OUT=results/robustness
mkdir -p "$OUT"
if [[ $# -gt 0 ]]; then
  conditions=("$@")
else
  conditions=($(ls data/hard))
fi
for condition in "${conditions[@]}"; do
  condition_dir="data/hard/$condition/"
  for engine in engines/*.engine; do
    precision="$(basename "$engine" .engine)"
    precision="${precision##*_}"
    echo "=== $condition $engine $(date +%T)" >> "$OUT/evaluate_hard.log"
    python3 jetson/evaluate.py --model "$engine" --precision "$precision" \
      --data "$condition_dir" --condition "$condition" --groups all distance \
      --out "$OUT/robustness_metrics.csv" >> "$OUT/evaluate_hard.log" 2>&1
  done
done
echo "Klaar $(date +%T)" >> "$OUT/evaluate_hard_done.txt"
