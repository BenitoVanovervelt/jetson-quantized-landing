# Precisielanding met gekwantiseerde objectdetectie op een Jetson

Researchproject Howest MCT (AI Engineer) — Benito Vanovervelt.

Een gesimuleerde drone in AirSim nadert een landingsplatform met een H-marker. De beelden van de onderste camera gaan via TCP naar een NVIDIA Jetson Orin Nano, die het platform detecteert met YOLO26s of RT-DETR-l als TensorRT-engine in FP32, FP16 of INT8. Een landingslogica op de pc beslist per frame om te dalen of de landing af te breken.

**Onderzoeksvraag:** hoe kan een TensorRT-geoptimaliseerde objectdetector op een Jetson Orin Nano een landingsplatform in AirSim realtime (≥ 20 FPS) detecteren, zodat een landingslogica bij verschillende naderingshoeken en -afstanden correct beslist om te dalen of af te breken?

Achtergrond, meetplan en beslissingen: [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md). Installatie vanaf nul: [docs/INSTALL.md](docs/INSTALL.md). Gebruik: [docs/USER_GUIDE.md](docs/USER_GUIDE.md). Afwijkingen tegenover het contractplan: [docs/AFWIJKINGEN.md](docs/AFWIJKINGEN.md).

## Opstelling

```
Windows-pc                                   Jetson Orin Nano
Unreal + AirSim  ── beeld (JPEG, TCP) ──►    jetson/server.py
landing/*        ◄── box, conf, tijden ──    TensorRT-engine
```

## Mappen

| Map | Inhoud | Python-omgeving |
|---|---|---|
| `sim/` | Verbinding met AirSim, camera, testscripts | `airsim-env` (3.9) |
| `data_gen/` | Dataset genereren en labels controleren | `airsim-env` |
| `training/` | Fine-tuning en ONNX-export | `train-env` (3.12) |
| `jetson/` | Protocol, inferentieserver, engines bouwen, offline evaluatie | Jetson (of `train-env` om te testen) |
| `landing/` | Landingslogica, uitkomsten, CSV-logging, ketentest | `airsim-env` |
| `analysis/` | Grafieken en samenvattende tabel uit de CSV's | `train-env` (heeft matplotlib) |
| `results/` | Meetresultaten en grafieken | |
| `presentation/` | Presentatiewebsite voor de verdediging (offline, cockpitstijl) | `build_data.py`: elke Python; `make_gallery.py`: `train-env` |

## Werkstroom

| Stap | Commando | Waar |
|---|---|---|
| 1. Dataset genereren | `python data_gen/generate_dataset.py` | pc, Unreal op Play |
| 2. Labels nakijken | `python data_gen/check_labels.py --split train` | pc |
| 3. Fine-tunen | `train-env\Scripts\python training/train.py --model yolo26s` (en `rtdetr-l`) | pc |
| 4. Engines bouwen | `python3 jetson/build_engines.py --weights models/yolo26s.pt --name yolo26s --calib-data data/landingpad/calib.yaml` | Jetson |
| 5. Offline evaluatie | `python3 jetson/evaluate.py --model engines/yolo26s_fp16.engine --precision fp16` | Jetson |
| 6. Server starten | `python3 jetson/server.py --model engines/yolo26s_fp16.engine` | Jetson |
| 7. Ketentest (FPS) | `bash landing/chain_all.sh yolo26s_int8 ...` (alle engines) of `python -m landing.chain_test --host 192.168.55.1 --model yolo26s --precision int8` | pc, Unreal op Play |
| 8. Live landingen | `bash landing/live_all.sh yolo26s_fp16 ...` (50 pogingen per engine) | pc, Unreal op Play |
| 8b. Moeilijkere testsets | `python data_gen/generate_hard_testset.py --conditions ...` en `jetson/evaluate.py --data data/hard/<conditie>` | pc (weer: Unreal op Play), Jetson |
| Demo | `bash landing/demo_servers.sh start yolo26s_fp16 rtdetr-l_fp32` en `python -m landing.demo --engines ...` (zie [USER_GUIDE](docs/USER_GUIDE.md)) | pc + Jetson, Unreal op Play |
| 9. Analyse | `train-env\Scripts\python analysis/make_figures.py` → `results/figures/*.png`, `results/summary.csv` | pc |
| 10. Presentatie | `analysis/make_figures.py --dark`, `presentation/make_gallery.py`, `python presentation/build_data.py`, daarna `presentation/index.html` openen (zie [USER_GUIDE](docs/USER_GUIDE.md), 3c) | pc |

Scripts in `landing/` starten met `python -m landing.<script>` vanuit de projectmap, omdat ze code uit `jetson/` en `sim/` gebruiken.

## Tests

```powershell
python -m unittest landing.test_logic -v
```

## Licentie

- Code, documentatie en resultaten: **AGPL-3.0-only** ([LICENSE](LICENSE)). Het project gebruikt Ultralytics, dat zelf AGPL-3.0 is; code en modellen die erop verder bouwen vallen daar ook onder.
- Gefinetunede modellen: AGPL-3.0-only. Dataset: **CC BY 4.0** (vrij te gebruiken met naamsvermelding).
- Gebruikte software van derden (AirSim, Unreal Engine, TensorRT, ...): zie [NOTICE.md](NOTICE.md).
