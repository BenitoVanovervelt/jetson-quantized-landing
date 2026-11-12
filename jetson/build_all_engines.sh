#!/usr/bin/env bash
# Bouw de 6 engines (2 modellen x FP32/FP16/INT8), elk in een eigen Python-proces.
#
# Waarom een apart proces per engine: op de Jetson delen CPU en GPU 8 GB geheugen.
# Binnen één proces wordt het geheugen van de vorige engine niet volledig vrijgegeven
# (gemeten: 4,7 GB vrij voor FP32, nog 1,3 GB voor FP16). Een nieuw proces start telkens
# met hetzelfde vrije geheugen, zodat de engines onder gelijke omstandigheden gebouwd worden.
#
# Gebruik (op de Jetson, vanuit ~/landing-project; draait in de container):
#   DETACH=1 NAME=landing-project-build jetson/run_in_docker.sh bash jetson/build_all_engines.sh
set -u

for model in yolo26s rtdetr-l; do
  for precision in fp32 fp16 int8; do
    python3 jetson/build_engines.py \
      --weights "models/${model}.pt" --name "$model" --precisions "$precision" \
      --calib-data data/landingpad/calib.yaml --bench-images data/landingpad/images/test \
      >> "results/engines/build_${model}.log" 2>&1
  done
done
echo "Klaar" >> results/engines/build_done.txt
