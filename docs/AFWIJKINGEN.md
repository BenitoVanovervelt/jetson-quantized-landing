# Afwijkingen tegenover het contractplan

Overzicht voor het verslag en de coach: wat anders gelopen is dan in het goedgekeurde contractplan, waarom, en
wat het betekent voor de resultaten. Gerangschikt van meest naar minst belangrijk.

## 1. Inhoudelijke afwijkingen (beïnvloeden de metingen)

| # | Contractplan | Uitgevoerd | Waarom | Gevolg |
|---|---|---|---|---|
| 1 | Camera onderaan, pitch −90°, vast aan de drone | Camera met **gimbal** (stabilisatie aan, pitch −90°, roll en yaw 0) | Zonder gimbal kantelt de camera mee met de drone. Bij het vliegen naar het platform vanuit θ = 45° viel het platform uit beeld. Een echte precisielandingsdrone heeft ook een gestabiliseerde camera. | Training, engines, offline evaluatie en ketentest blijven geldig (beelden toen ook recht naar beneden). De gimbal houdt de camera ook in dezelfde **gierrichting**: opnieuw gerenderde beelden zijn gedraaid t.o.v. de oorspronkelijke dataset (zie 1.5). |
| 2 | Labels uit **segmentatiebeelden** | Labels uit de **detectie-API** van AirSim (`simGetDetections`, filter `LandingPad*`, `box2D`) | Geeft rechtstreeks een box in pixels, zonder segmentatiebeeld te verwerken. Beide zijn automatische labels uit AirSim. Visueel gecontroleerd (`data_gen/check_labels.py`, `results/label_check/`). | Geen. De segmentatieaanpak staat in het plan als alternatief; niet nodig gebleken. |
| 3 | Teleporteren met `simSetVehiclePose` | `simSetKinematics` (positie + snelheid 0) in een **gepauzeerde** simulatie, met `simContinueForTime(0.01)` | `simSetVehiclePose` zet de snelheid niet terug: de drone bleef vallen en versnelde over de beelden heen (0,5 tot 47 m naast de doelpositie). | Gemeten afwijking ± 0,3 mm. 20 van 4800 beelden mislukten toch (Unreal even trager) en zijn opnieuw gemaakt (`--repair`); het script controleert nu elke positie. |
| 4 | Keten-FPS ≥ 20 voor minstens één combinatie | Ketentest (drone hangend op 10 m): YOLO26s FP16 **22,6** en INT8 **22,0** FPS. Tijdens de live landingen: maximaal **19,7** FPS (YOLO26s INT8) | Tijdens het vliegen duurt het beeld ophalen uit AirSim langer (± 30 ms i.p.v. ± 24 ms). De Jetson zelf is even snel. | Het succescriterium is gehaald in de ketentest zoals het meetplan die beschrijft, niet tijdens de live landingen. Het knelpunt is de simulator op de pc, niet de Jetson. Vermelden als beperking. |
| 5 | Landingslogica: dalen na 5 frames, anders afbreken | Zelfde regel, aangevuld: afbreken na **3 gemiste frames** tijdens het dalen of **100 frames** zonder 5 op een rij; afbreken = blijven hangen; eerst horizontaal tot boven het platform, dan verticaal; **onder 3 m blind** landen, laatste meter 0,3 m/s | Onder ± 2 m valt de schaduw van de drone over het platform en zakt de confidence; het model zag nooit beelden van dichter dan 4 m. Echte precisielandingen schakelen in de laatste meters ook over op een andere sensor. | De detectie beslist óf de drone daalt (zoals in het plan); de laatste 3 m zijn niet door detectie bewaakt. |
| 6 | Uitkomst `slechte_landing` = naast, te hard, te scheef | Ook: contact met **een ander object** dan het platform | Bij één startpositie (poging 35, θ = 45°, φ 5,1°) begon de drone in een blok van de scène en "landde" hij op dat blok binnen de grenzen. | Startposities worden nu gecontroleerd tegen de blokken van de scène; φ wordt in stappen van 10° bijgedraaid (enkel poging 35: 5,1° → 35,1°). Mislukte runs bewaard in `results/live/run_*`. |

## 2. Extra werk (niet in het plan)

| # | Wat | Waarom | Resultaat |
|---|---|---|---|
| 7 | **Moeilijkere testsets**: mist, stof, weinig licht, bewegingsonscherpte, ruis (`data_gen/generate_hard_testset.py`) | Alle 6 engines scoorden (bijna) perfect, offline en live (300/300). Zonder moeilijkere test is niet te zien waar de detectie faalt (deelvragen 7 en 8). Domain randomization stond in het plan als uitbreiding; dit test het probleem dat ze oplost. | Recall zakt sterk (bv. weinig licht: YOLO26s 0,27, RT-DETR-l 0,02). YOLO26s faalt "veilig" (ziet niets, precision ≥ 0,97); RT-DETR-l ziet in mist dingen die er niet zijn (precision 0,02–0,13, vals alarm 4–5 %). `results/figures/robustness.png` |
| 7a | Moeilijkere testsets **per rekenprecisie op de Jetson** (`jetson/evaluate_hard_all.sh`) | Het effect van kwantisatie was op schone beelden onzichtbaar (plafond). | FP16 = FP32. YOLO26s INT8 verliest recall onder moeilijke omstandigheden (mist 0,5: 0,73 → 0,59), RT-DETR-l INT8 (mengvorm) niet. `results/figures/robustness_precision.png` |
| 7b | **Drempel-sweep** (`analysis/threshold_sweep.py`) | De drempel 0,5 was gekozen, niet onderbouwd. | Normaal: recall 1,00 en 0 % vals alarm voor elke drempel tussen 0,25 en 0,85: 0,5 zit veilig in het midden. RT-DETR-l ziet in mist een groot vals vlak met confidence ± 0,93: geen drempel houdt dat tegen. |
| 8 | **Betrouwbaarheidsintervallen** (Clopper-Pearson, 95 %) | 50/50 correct zegt niet "100 %". | Ondergrens 92,9 % per engine, 69 % per starthoek (n = 10). |

Bij 7 zijn de weercondities opnieuw gerenderd op dezelfde posities. Door de gimbal (1) keek de camera anders dan
bij de oorspronkelijke dataset, waardoor de eerste versie verkeerde labels had. Opgelost met nieuwe labels uit
dezelfde detectie-API en een referentieconditie `clean` (zelfde posities, geen weer, recall 1,00). De foute
resultaten staan apart in `results/robustness_pc/robustness_metrics_invalid_weather_old_labels.csv`.
Regen en sneeuw zijn weggelaten: in de Blocks-scène veranderen ze het beeld van een camera die recht naar beneden
kijkt bijna niet.

## 3. Praktische afwijkingen (geen invloed op de conclusies)

| # | Contractplan / verwachting | Uitgevoerd | Waarom |
|---|---|---|---|
| 9 | Bekabelde verbinding pc ↔ Jetson | **USB-C** (netwerk over USB, 192.168.55.x) | Bekabeld en zonder extra hardware. Netwerktijd gemeten: ± 1 ms per frame. |
| 10 | TensorRT op de Jetson | TensorRT **10.7 in een Docker-container** (`ultralytics/ultralytics:8.4.182-jetson-jetpack6`), niet de 10.3 van het systeem | De Jetson is gedeeld (AI Lab): niets op het systeem installeren. Zelfde Ultralytics-versie als op de pc. |
| 11 | Vermogensmodus niet bepaald | **15W** voor alle metingen | Ingesteld door de eigenaar; enkel wijzigen met toestemming. MAXN zou sneller zijn: de gemeten FPS is dus een ondergrens van wat de Jetson kan. |
| 12 | RT-DETR-l in INT8: mogelijk niet exporteerbaar | Gelukt, maar als **mengvorm**: sommige transformerlagen blijven in hogere precisie | Zo bouwt TensorRT de engine. Verklaart waarom RT-DETR-l INT8 nauwelijks nauwkeurigheid verliest (mAP50-95 0,981 → 0,979). De versnelling FP16 → INT8 is wel even groot als bij YOLO26s (1,37× tegenover 1,38×). |
| 13 | Scène | Eigen kopie van de Blocks-scène in `unreal/Blocks/` | De AirSim-map blijft onaangeroerd (afspraak). |
| 14 | Simulatie op de pc | `"ViewMode": "NoDisplay"` en framelimiet weg (`t.MaxFPS 120`, `r.VSync 0`) | Het hoofdvenster renderen en de standaardlimiet van ± 30 FPS van de editor vertraagden elk beeld (± 33 ms i.p.v. ± 24 ms). Voor de demo `NoDisplay` weglaten. |

## 4. Incidenten

| Wanneer | Wat | Gevolg |
|---|---|---|
| Live landingen | Eerste runs mislukt: teleportfout, start in een blok (6), onderbroken door slaapstand, twee reeksen tegelijk die elkaars server herstartten | Opgelost (obstakelcontrole, lock in `live_all.sh`); oude runs apart in `results/live/run_*`. De definitieve run (8 nov) liep zonder fouten. |
| 8 nov, tijdens het kopiëren van de moeilijkere testsets | Het bestandssysteem van de Jetson ging op read-only (I/O-fouten op de SD-kaart). Na een stroomonderbreking herstelde het zichzelf bij het opstarten ("recovery complete") | Geen verlies: alle eerdere metingen stonden al op de pc, de engines bleven intact. De evaluatie per rekenprecisie op de moeilijkere testsets liep daarna zonder fouten (8 condities × 6 engines). |
