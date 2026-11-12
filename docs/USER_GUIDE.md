# Gebruikershandleiding

Hoe je de opstelling start, landingen laat uitvoeren, metingen herhaalt en de resultaten leest.
De installatie vanaf nul staat in [INSTALL.md](INSTALL.md); de achtergrond en keuzes in
[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md).

## 0. Wat kan je ermee?

| Wat | Hoe | Sectie |
|---|---|---|
| Een drone autonoom laten landen op een platform, of de landing laten afbreken als er geen platform is | `landing/live_test.py` of `landing/live_all.sh` | 3 |
| Live demo: camerabeeld met box, confidence, FPS en beslissing; wisselen tussen engines, platform weghalen, mist aanzetten | `landing/demo.py` | 3b |
| Het onderzoek presenteren (15 hoofdstukken, offline, met grafieken en demovideo) | `presentation/index.html` | 3c |
| Alle metingen herhalen: offline nauwkeurigheid, FPS van de keten, 300 live landingen, robuustheid, drempel-sweep | scripts in `jetson/`, `landing/`, `analysis/` | 4 |
| De resultaten lezen en grafieken maken | `results/`, `analysis/make_figures.py` | 5 |

## 1. Wat je nodig hebt

| Onderdeel | Toestand |
|---|---|
| Windows-pc | Unreal Engine 4.27, `airsim-env` en `train-env` geïnstalleerd (INSTALL, sectie 1) |
| Scène | `unreal/Blocks/Blocks.uproject` met het platform `LandingPad` (INSTALL, sectie 3) |
| Jetson Orin Nano | via USB-C verbonden, SSH-sleutel ingesteld, engines gebouwd in `~/landing-project/engines/` (INSTALL, sectie 2) |
| Terminals | PowerShell voor Python-scripts, **Git Bash** voor de `.sh`-scripts |

Alle commando's starten vanuit de projectmap `C:\dev\landing-project`.

## 2. Opstarten (elke keer)

1. **Unreal.** Open `unreal\Blocks\Blocks.uproject` met `UE4Editor.exe` en druk op **Play**.
   Zonder Play geeft elk script `ConnectionRefusedError` of de melding "geen AirSim-simulator gevonden".
2. **Jetson.** Sluit de USB-C-kabel aan en controleer de verbinding:
   ```powershell
   ssh <gebruiker>@192.168.55.1 "nvpmodel -q | head -1; ls ~/landing-project/engines"
   ```
   Verwacht: de vermogensmodus (15W) en de 6 engines.
3. **Python-omgeving voor AirSim** (in PowerShell):
   ```powershell
   airsim-env\Scripts\activate
   python sim/test_connection.py      # moet "Connected!" en de positie van de drone tonen
   ```

## 3. Een landing laten uitvoeren

De Jetson moet de inferentieserver draaien met de gekozen engine. Het eenvoudigst is het script dat dat
zelf doet (Git Bash):

```bash
bash landing/live_all.sh yolo26s_fp16
```

Dat start de server op de Jetson, wacht tot de engine geladen is, voert 50 pogingen uit (10 per hoek θ ∈ {0, 15, 30, 45}°
en 10 zonder platform) en stopt de server. Duur: ± 25 minuten per engine.

Om eerst kort te testen, start je de server zelf en beperk je het aantal pogingen:

```bash
# Git Bash: server starten op de Jetson (blijft draaien op de achtergrond)
ssh <gebruiker>@192.168.55.1 "cd ~/landing-project && DETACH=1 NAME=landing-project-server jetson/run_in_docker.sh python3 jetson/server.py --model engines/yolo26s_fp16.engine"
ssh <gebruiker>@192.168.55.1 "docker logs -f landing-project-server"     # wacht op "Wacht op verbinding", dan Ctrl+C
```

```powershell
# PowerShell, airsim-env actief
python -m landing.live_test --host 192.168.55.1 --model yolo26s --precision fp16 --limit 5
python -m landing.live_test --host 192.168.55.1 --model yolo26s --precision fp16 --only 34 35   # één poging naspelen
```

```bash
ssh <gebruiker>@192.168.55.1 "docker rm -f landing-project-server"     # server stoppen
```

**Wat je ziet.** Per poging één regel in de terminal, bv.:

```
#12 theta=  30 phi=145.0 platform=ja  -> goede_landing  fout=0.04 v=0.29 kant=1.2 contact=LandingPad (412 frames, 38 s, startafwijking 0 cm)
```

| Veld | Betekenis |
|---|---|
| `theta`, `phi` | naderingshoek en richting van de startpositie (afstand altijd 20 m) |
| `platform` | `nee` = poging zonder platform (het platform staat 500 m verder) |
| uitkomst | zie tabel hieronder |
| `fout`, `v`, `kant` | positiefout (m), verticale snelheid bij contact (m/s), kanteling (°) |
| `contact` | object waarop de drone landde: moet `LandingPad` zijn |

| Uitkomst | Betekenis | Correct? |
|---|---|---|
| `goede_landing` | geland op het platform: fout < 0,25 m, snelheid < 0,5 m/s, kanteling < 10° | ja |
| `terecht_afgebroken` | geen platform, drone bleef hangen | ja |
| `slechte_landing` | geland, maar naast, te hard, te scheef of op een ander object | nee |
| `onterecht_afgebroken` | platform aanwezig, maar de drone daalde niet | nee |
| `foute_landing` | geland zonder platform (gevaarlijkste fout) | nee |

**Hoe de drone beslist** (`landing/decision.py`): dalen pas na **5 opeenvolgende frames** met een detectie boven
confidence 0,5. Afbreken na 3 gemiste frames tijdens het dalen, of na 100 frames zonder 5 op een rij. Afbreken =
blijven hangen. De drone vliegt op zijn positie (zoals GPS): eerst horizontaal tot boven het platform, dan verticaal
aan 1 m/s. Onder 3 m landt hij blind (schaduw van de drone over het platform; het model zag nooit beelden van
dichter dan 4 m), de laatste meter aan 0,3 m/s.

## 3b. Demo

Eén landing per keer, live in beeld: camerabeeld met box en confidence, FPS, inferentietijd, beslissing
(HOLD / DESCEND / ABORT), detecties op rij en hoogte. Zelfde landingslogica als de metingen; er wordt niets gelogd.

1. Voor een beeld van de drone in Unreal: haal `"ViewMode": "NoDisplay"` uit `settings.json` en druk opnieuw op **Play**
   (zet het terug voor metingen: het hoofdvenster renderen vertraagt de keten).
2. Servers starten op de Jetson, één per engine (Git Bash). Beide engines blijven geladen, zodat wisselen meteen gaat:
   ```bash
   bash landing/demo_servers.sh start yolo26s_fp16 rtdetr-l_fp32
   ```
3. Demo starten (PowerShell, `airsim-env`), het script toont het juiste commando:
   ```powershell
   python -m landing.demo --engines yolo26s_fp16@192.168.55.1:5000 rtdetr-l_fp32@192.168.55.1:5001
   ```
4. Toetsen in het demovenster:

   | Toets | Actie |
   |---|---|
   | spatie | nieuwe landing (starthoek wisselt: 0, 15, 30, 45°) |
   | 1, 2 | engine kiezen |
   | p | platform weg / terug: de drone breekt af |
   | f | mist aan / uit (sterkte 0,5): YOLO26s breekt af, RT-DETR-l ziet soms een vals platform |
   | q, Esc | stoppen |

5. Na afloop: `bash landing/demo_servers.sh stop`.

**Zonder Jetson** (plan B): de server draait ook op de pc met een PyTorch-model.
```powershell
train-env\Scripts\python jetson/server.py --model models/yolo26s.pt --host 127.0.0.1 --port 5000
python -m landing.demo --engines yolo26s_pt@127.0.0.1:5000
```
**Reservevideo**: `python -m landing.demo ... --auto 5 --record presentation/video/demo.mp4` doet 5 landingen na elkaar zonder toetsen (engines om beurten, starthoek 0/15/30/45°, laatste landing zonder platform) en schrijft het demovenster in echte tijd naar een mp4 (H.264, speelt af in de browser). Waarom zelf opnemen in plaats van een schermopname: exact het camerabeeld met box en FPS, zonder knippen.

## 3c. Presentatiewebsite

De website in `presentation/` vertelt het onderzoek in 15 hoofdstukken (één per scherm). Ze werkt volledig offline:
geen internet, geen installatie.

**Openen:** dubbelklik op `presentation/index.html` (Chrome of Edge), daarna **F11** voor volledig scherm.

| Toets | Actie |
|---|---|
| → of spatie / ← | volgend / vorig hoofdstuk |
| 1–9, 0 | rechtstreeks naar hoofdstuk 1–9, 0 = hoofdstuk 10 (demo) |
| W | lade "waarom" met de beslissingen en redenen bij dit hoofdstuk |
| N | spreeknotities en een timer (T zet de timer op 0) |
| B | bijlagen: bronnen, levenslang leren, versies, begrippen |
| V | demovideo volledig scherm (back-up voor de live demo) |
| L | licht / donker (als de projector het donkere thema slecht toont) |
| Esc | alles sluiten |

Bij hoofdstuk 10 (demo) wissel je met Alt + Tab naar het venster van `landing/demo.py` (sectie 3b) en terug.

**Zelf aan te vullen** (de website toont een duidelijk vak zolang ze ontbreken):
- in `presentation/index.html`, deel "Bijlagen": de bronnen die je echt gebruikte en wat je deed aan levenslang leren
  (gemarkeerd met "IN TE VULLEN"). Neem voor levenslang leren enkel over wat in `docs/community/community_log.md`
  staat, dus wat echt gebeurd is: het Medium-artikel en de forumbijdragen pas als ze gepost zijn.

**Cijfers bijwerken** na een nieuwe meting (vanuit de projectmap):
```powershell
train-env\Scripts\python analysis\make_figures.py --dark   # donkere grafieken in results/figures_dark/
train-env\Scripts\python presentation\make_gallery.py      # galerij slecht zicht (beide modellen op de pc)
python presentation\build_data.py                          # js/data.js + beelden kopiëren
```
Alle getallen op de website komen uit `js/data.js` en `js/gallery_data.js`; die worden door deze scripts gemaakt uit
`results/`, niets is met de hand overgetypt.

## 4. Metingen herhalen

| Meting | Commando | Waar | Duur | Resultaat |
|---|---|---|---|---|
| Offline evaluatie (6 engines) | `DETACH=1 NAME=landing-project-eval jetson/run_in_docker.sh bash jetson/evaluate_all.sh` | Jetson, in `~/landing-project` | ± 15 min | `results/offline/offline_metrics.csv` |
| Ketentest (FPS, 500 frames per engine) | `bash landing/chain_all.sh yolo26s_fp32 yolo26s_fp16 yolo26s_int8 rtdetr-l_fp32 rtdetr-l_fp16 rtdetr-l_int8` | pc, Git Bash, Unreal op Play | ± 15 min | `results/chain/` |
| Live landingen (50 per engine) | `bash landing/live_all.sh yolo26s_fp32 ... rtdetr-l_int8` | pc, Git Bash, Unreal op Play | ± 2,5 uur | `results/live/` |
| Moeilijkere testsets maken | `python data_gen/generate_hard_testset.py --conditions ...` (INSTALL 1.8) | pc, `airsim-env` | ± 30 min | `data/hard/` |
| Moeilijkere testsets evalueren | `DETACH=1 NAME=landing-project-hard jetson/run_in_docker.sh bash jetson/evaluate_hard_all.sh` | Jetson | ± 1,5 uur (8 condities) | `results/robustness/` |
| Drempel-sweep | `train-env\Scripts\python analysis/threshold_sweep.py` | pc | ± 30 min | `results/threshold_sweep/` |
| Grafieken en samenvatting | `train-env\Scripts\python analysis/make_figures.py` | pc | < 1 min | `results/figures/`, `results/summary.csv` |
| Unittests landingslogica | `python -m unittest landing.test_logic -v` | pc, `airsim-env` | < 1 s | 14 tests OK |

Resultaten van de Jetson naar de pc halen (Git Bash):

```bash
scp -r <gebruiker>@192.168.55.1:~/landing-project/results/offline results/offline_jetson
scp -r <gebruiker>@192.168.55.1:~/landing-project/results/robustness results/
```

**Tips**
- Laat de pc niet in slaapstand gaan tijdens lange runs (Windows: Instellingen → Systeem → Energie → Slaapstand: Nooit).
  Na een onderbroken run: verwijder `results/live/live_all.lock` en zet de halve CSV's in een submap.
- Draai nooit twee reeksen tegelijk: elke reeks herstart de server op de Jetson (het lock in `live_all.sh` bewaakt dat).
- Laat in Unreal "Use Less CPU when in Background" uit, anders vertraagt de simulatie als VS Code actief is.

## 5. Resultaten lezen

| Bestand | Inhoud |
|---|---|
| `results/summary.csv` | één rij per engine: precision, recall, mAP50, mAP50-95, inferentietijd, keten-FPS (gemiddelde en 5e percentiel), live FPS, % correct met 95 %-betrouwbaarheidsinterval, aantallen per uitkomst |
| `results/figures/offline_map50_95.png` | mAP50-95 per afstand en per hoek, per engine |
| `results/figures/offline_map50_95_heatmap.png` | mAP50-95 per combinatie hoek × afstand |
| `results/figures/inference_ms.png` | inferentietijd per engine op de Jetson |
| `results/figures/chain_fps.png` | FPS van de volledige keten (ketentest en tijdens de live landingen), met de eis van 20 FPS |
| `results/figures/chain_breakdown.png` | waar de tijd per frame naartoe gaat: beeld ophalen, netwerk, server, inferentie |
| `results/figures/live_outcomes.png` | uitkomsten per engine en per starthoek |
| `results/figures/live_confidence.png` | mediane confidence tegenover de hoogte tijdens het dalen |
| `results/figures/robustness.png` | recall per afstand en precision/vals alarm onder mist, stof, weinig licht, onscherpte en ruis |
| `results/figures/robustness_precision.png` | recall onder moeilijke omstandigheden per rekenprecisie (Jetson, 6 engines) |
| `results/figures/threshold_sweep.png` | recall en vals alarm tegenover de confidence-drempel (waarom 0,5) |
| `results/figures/rtdetr_fog_false_detections.jpg` | voorbeelden van valse detecties van RT-DETR-l in mist |
| `results/live/live_<engine>_<tijd>_attempts.csv` | één rij per poging (uitkomst, positiefout, snelheid, kanteling) |
| `results/live/live_<engine>_<tijd>_frames.csv` | één rij per frame (detectie, confidence, box, tijden, beslissing, positie) |

**Betrouwbaarheidsinterval.** 50 op 50 correct betekent niet "100 % betrouwbaar": het exacte 95 %-interval
(Clopper-Pearson) is 92,9–100 %. Per starthoek (10 pogingen) is de ondergrens 69 %.

## 6. Problemen

| Probleem | Oorzaak en oplossing |
|---|---|
| `ConnectionRefusedError` / "geen AirSim-simulator gevonden" | Unreal staat niet op Play |
| Wijziging in `settings.json` heeft geen effect | pas actief na een nieuwe Play |
| `ssh: connect to host 192.168.55.1 ... timed out` | Jetson niet (volledig) opgestart of USB-C-kabel in de verkeerde poort. Windows moet een "Remote NDIS"-netwerkadapter tonen |
| Server start niet (`live_all.sh` blijft wachten) | `ssh <gebruiker>@192.168.55.1 "docker logs landing-project-server"`; bij `out of memory`: andere containers en cache (INSTALL 2.4) |
| Jetson: `Input/output error`, bestandssysteem `ro` | opslag (SD-kaart) gaf een fout; stroom onderbreken en herstarten, bij herhaling de eigenaar verwittigen |
| Klok van de Jetson loopt achter | geen internet = geen tijdsynchronisatie; tijdstempels van de Jetson niet vergelijken met de pc. Bij `tar`: `-m` gebruiken |
| Drone start in een blok van de scène | `live_test.py` controleert de route en draait de startrichting φ bij (melding "route door een blok") |
