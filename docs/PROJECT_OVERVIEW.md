# Projectoverzicht: precisielanding met gekwantiseerde objectdetectie op een Jetson

Volledige achtergrond van het project. `README.md` geeft het korte overzicht; dit document legt uit **wat**, **waarom** en **hoe**.

---

## 1. Context

| | |
|---|---|
| Opleiding | Howest, Bachelor Multimedia & Creatieve Technologie (MCT), keuzetraject AI Engineer |
| Module | Researchproject: 4 weken voltijds de onderzoeksvraag technisch uitwerken, op school |
| Student | Benito Vanovervelt (individueel) |
| Hardware via school | NVIDIA Jetson Orin Nano (ontleend van school) |
| Vervolg | Bachelorproef: reflectie op de resultaten met het werkveld |

Het researchproject is altijd een **concrete technische realisatie**, geen literatuurstudie. De student moet elke technische keuze kunnen verdedigen bij de coach.

### Wat er op het einde opgeleverd moet worden
1. **Broncode** (volledig).
2. **Installatiehandleiding**: een "cold start". Iemand anders moet op basis van de code en de handleiding alles opnieuw kunnen installeren en reproduceren.
3. **Gebruikershandleiding**: hoe gebruik je de opstelling en de demo.
4. **Bijlages**: technische schema's, screenshots, video's, grafieken.

Houd daarom vanaf het begin een `README.md` en een `docs/INSTALL.md` bij.

---

## 2. Onderzoeksvraag

> Hoe kan een TensorRT-geoptimaliseerde objectdetector op een NVIDIA Jetson Orin Nano een landingsplatform in een AirSim-simulatie in realtime (≥ 20 FPS) detecteren, zodat een landingslogica bij verschillende naderingshoeken en -afstanden correct beslist om te dalen of de landing af te breken?

### Deelvragen

**Uit de literatuur (fase 1):**
1. Hoe werkt het omzetten van een model van FP32 naar FP16 en INT8 in TensorRT, en waarom heeft INT8 representatieve kalibratiedata nodig?
2. Wat zijn de belangrijkste verschillen tussen een CNN-detector (YOLO26) en een transformerdetector (RT-DETR), en waarom zouden ze anders kunnen reageren op kwantisatie?
3. Welke eisen stelt een autonome dronelanding aan de snelheid en betrouwbaarheid van de detectie?

**Eigen onderzoek:**

4. Hoe kan ik in AirSim automatisch gelabelde beelden van een landingsplatform genereren vanuit verschillende hoeken en afstanden?
5. Wat is de invloed van FP32, FP16 en INT8 op de precision, recall, mAP50 en inferentietijd van YOLO26 en RT-DETR op de Jetson?
6. Hoeveel vertraging voegt de beeldstream tussen de pc en de Jetson toe, en haalt de volledige keten van beeld tot beslissing nog 20 FPS?
7. Bij welke naderingshoeken en -afstanden daalt de detectie zo sterk dat de landingslogica een foute beslissing neemt?
8. Welke combinatie van model en rekenprecisie haalt de FPS-eis met het kleinste aantal foute landingsbeslissingen?

### Relevantie
Drones die pakjes leveren of autonoom landen in een laadstation ("drone-in-a-box") vliegen met GPS tot boven het platform en gebruiken in de laatste fase een camera onderaan en een kleine boordcomputer zoals een Jetson. Daar is rekenkracht beperkt, dus kwantisatie is vaak nodig om realtime te blijven. Dit project toont welke detector en rekenprecisie het best werken en wat dat kost aan betrouwbaarheid, zonder echte drones te riskeren.

---

## 3. Afbakening

| Binnen scope | Buiten scope |
|---|---|
| Eén landingsplatform met H-marker in een zelfgebouwde AirSim-scène | Een echte drone of echte testvluchten |
| YOLO26s en RT-DETR-l uit Ultralytics, gefinetuned op synthetische data | Andere detectiemodellen of een model van nul trainen |
| FP32, FP16 en INT8 via TensorRT op één Jetson Orin Nano | Quantization-aware training, pruning, andere optimalisaties |
| Een landingslogica die beslist om te dalen of af te breken | Een volledige flight controller (PX4, ArduPilot), padplanning, precieze positiebepaling |
| Naderingshoek en afstand als testvariabelen | Wind, regen, bewegend platform |
| Gewone (asrechte) bounding boxes | Geroteerde bounding boxes (OBB): enkel als uitbreiding |

---

## 4. Architectuur

```
VOORBEREIDING (offline, op de pc)
  AirSim-scène (UE 4.27, LandingPad)
        │  script: drone op posities (θ, d, φ) zetten
        ▼
  Datageneratie  ──►  beelden + YOLO-labels + metadata (CSV)
        │
        ▼
  Fine-tuning (Ultralytics, pc-GPU): YOLO26s, RT-DETR-l
        │  export .pt ──► ONNX
        ▼
  Op de Jetson: ONNX ──► TensorRT-engines (FP32, FP16, INT8)  = 6 engines

LANDINGSTEST (live)
  Windows-pc                               Jetson Orin Nano
  ┌──────────────────────────┐   beeld     ┌──────────────────────────┐
  │ Unreal + AirSim (drone)  │ ──(TCP)──►  │ TensorRT-engine          │
  │ Python-client:           │             │ detectie + tijdmeting    │
  │  - beeld bottom_cam      │ ◄──(TCP)──  │ stuurt box, conf,        │
  │  - landingslogica        │  resultaat  │ latency terug            │
  │  - drone aansturen       │             └──────────────────────────┘
  │  - logging naar CSV      │
  └──────────────────────────┘
        │
        ▼
  Analyse (matplotlib): grafieken per model, rekenprecisie, hoek, afstand
```

De landingslogica draait op de pc (dicht bij de simulator); de Jetson doet alleen de inferentie. Zo meet je precies wat de Jetson bijdraagt.

---

## 5. Componenten in detail

### 5.1 Simulatie
- Microsoft AirSim, gebouwd uit broncode, Unreal Engine 4.27, Blocks-omgeving.
- Drone: `SimpleFlight` (ingebouwde vluchtbesturing).
- Camera `bottom_cam`: onderaan, pitch −90°, 640 × 640, FOV 90°. Zie `docs/INSTALL.md` voor `settings.json`.
- Landingsplatform: 1 m × 1 m, textuur `assets/landingpad_H.png`, actornaam `LandingPad`.
- Python-client met Python 3.9 (`airsim-env`).

### 5.2 Naderingsgeometrie
- **Naderingshoek θ**: hoek tussen de verticale lijn boven het platform en de lijn drone naar platform. 0° = recht erboven.
- **Afstand d**: rechte afstand drone tot midden platform.
- **Richting φ**: van welke kant de drone komt.
- Positie t.o.v. platform (NED, meter): `x = d·sinθ·cosφ`, `y = d·sinθ·sinφ`, `z = −d·cosθ`.
- θ ∈ {0°, 15°, 30°, 45°}, d ∈ {5, 10, 20, 30} m.
- **Waarom max 45°**: met FOV 90° ligt de beeldrand op 45° van de verticale lijn. Daarboven valt het platform buiten beeld.
- **Grootte in beeld**: brandpuntsafstand `f = 320 / tan(45°) = 320 px`. Platform van 1 m op d meter is ongeveer `320 / d` pixels breed (op 30 m ≈ 11 px). Daar wordt detectie moeilijk: precies daar verwacht je het effect van kwantisatie.

### 5.3 Data en labels
- Drone teleporteren met `simSetVehiclePose` (geen fysica, snel, herhaalbaar).
- Labels automatisch uit AirSim. **Eerste keuze**: detectie-API (`simGetDetections`, filter `LandingPad*`, gebruik `box2D`). **Alternatief**: segmentatiebeelden, of de 3D-positie projecteren op het beeld.
  - Let op: het contractplan noemt "labels uit segmentatiebeelden". Beide zijn automatische labels uit AirSim; gebruikt is de detectie-API. Die afwijking staat in `docs/AFWIJKINGEN.md`.
- YOLO-formaat: één regel per object: `0 x_center y_center width height`, genormaliseerd 0..1.
- Metadata per beeld (CSV): bestandsnaam, θ, d, φ, positie drone, positie platform, of het platform zichtbaar is.
- Splitsing: ± 3000 train, 500 val, 1000 test, 300 INT8-kalibratie.
  - Testset met **andere posities** dan train (bv. andere φ of tussenliggende θ/d), zodat er geen lekkage is.
  - Kalibratieset los van de testset.
- Ook beelden **zonder platform** of met het platform half in beeld, nodig om afbreken te testen.

### 5.4 Modellen
- Ultralytics, vertrekken van COCO-gewichten.
- **YOLO26s** (CNN, NMS-vrij) en **RT-DETR-l** (transformer, kleinste RT-DETR in Ultralytics).
- Eén klasse: `landingsplatform`.
- Training op de pc in een **aparte** Python-omgeving (nieuwere Python, PyTorch met CUDA). Niet in `airsim-env`.

### 5.5 Kwantisatie
- Export naar ONNX op de pc, engines bouwen **op de Jetson** (een engine is gebonden aan hardware en TensorRT-versie).
- Drie rekenprecisies per model: FP32 (referentie), FP16, INT8 (met kalibratieset).
- Zes engines in totaal. Lukt een export technisch niet (bekend risico: RT-DETR in INT8), bewaar dan de foutmelding: dat is een resultaat.
- Noteer versies van JetPack, TensorRT, CUDA en Ultralytics in de resultaten.

### 5.6 Communicatie pc en Jetson
- TCP-sockets in Python, over een bekabelde verbinding.
- Pc stuurt JPEG-gecodeerd beeld + framenummer + tijdstempel.
- Jetson stuurt terug: boxes, confidences, inferentietijd.
- Pc meet de volledige round trip. Netwerktijd en inferentietijd apart loggen.
- ROS 2 is bewust niet gekozen: te zwaar voor deze scope.

### 5.7 Landingslogica
1. De drone vliegt op basis van zijn positie (zoals GPS) naar de startpositie en daalt richting platform.
2. Elke frame: detectie opvragen.
3. Dalen alleen als het platform in **5 opeenvolgende frames** gedetecteerd wordt boven een vaste confidence-drempel.
4. Valt de detectie weg: landing afbreken (stijgen of hoogte houden).

**FPS-eis**: `FPS ≥ N · v / s_max`. Met N = 5, v = 1 m/s, s_max = 0,25 m geeft dat **20 FPS**. Bij 20 FPS legt de drone 5 cm per frame af.

### 5.8 Wat is een goede landing
| Grootheid | Meting in AirSim | Grens |
|---|---|---|
| Positiefout | Horizontale afstand drone tot platform (`simGetVehiclePose`, `simGetObjectPose`) | < 0,25 m |
| Landingssnelheid | Verticale snelheid bij contact (`getMultirotorState().kinematics_estimated.linear_velocity`) | < 0,5 m/s |
| Kanteling (landingshoek) | Roll en pitch bij contact (`airsim.to_eularian_angles`) | < 10° |
| Moment van contact | `simGetCollisionInfo()` | |

### 5.9 Uitkomstklassen per landingspoging
| | Drone daalt | Drone breekt af |
|---|---|---|
| Platform aanwezig | `goede_landing` (binnen grenzen) of `slechte_landing` (naast, te hard, te scheef) | `onterecht_afgebroken` |
| Platform afwezig | `foute_landing` (gevaarlijkste fout) | `terecht_afgebroken` |

---

## 6. Meetplan

### Offline (alleen beelden, snel)
- Alle 6 engines op de testset.
- Precision, recall, mAP50, inferentietijd.
- Uitgesplitst per θ en d.

### Keten
- FPS van beeld tot beslissing, minstens 500 frames per engine.
- Gemiddelde en 5e percentiel.

### Live landingen
- Per engine 50 pogingen: 10 per starthoek (4 hoeken) en 10 zonder platform.
- Totaal 300 landingen, ± 40 s per landing, ± 3,5 uur. Volledig geautomatiseerd.
- Afstand is geen startvariabele: een landing daalt door alle afstanden heen. Het effect van afstand komt uit de offline metingen.

### Logging (CSV)
Minstens: tijdstempel, poging-id, model, rekenprecisie, θ, d, φ, framenummer, gedetecteerd (ja/nee), confidence, box, inferentietijd, netwerktijd, FPS, beslissing, positie drone, uitkomst, positiefout, landingssnelheid, kanteling.

---

## 7. Succescriteria

| Criterium | Meting | Minimaal resultaat |
|---|---|---|
| Realtime detectie | FPS keten over ≥ 500 frames per engine | Minstens één combinatie gemiddeld ≥ 20 FPS |
| Betrouwbare detectie | Precision en recall op de testset | Beste combinatie ≥ 0,90 |
| Correcte beslissingen | 50 pogingen per engine | Beste combinatie ≥ 95% correct |
| Effect kwantisatie | FP32, FP16, INT8 voor beide modellen, per θ en d | Alle 6 gemeten; mislukte export gedocumenteerd |
| Werkende demo | Live tijdens eindpresentatie | Drone landt of breekt af met box en FPS in beeld, wisselen tussen minstens 2 engines |

---

## 8. Risico's en plan B

| Risico | Plan B |
|---|---|
| AirSim niet meer onderhouden, bouwproblemen | Colosseum (branch UE 4.27) of kant-en-klare Blocks |
| RT-DETR-export naar TensorRT (vooral INT8) mislukt | Eerst via ONNX + `trtexec`; anders FP32/FP16 meten en mislukking rapporteren |
| Automatische labeling werkt niet | Segmentatie, projectie van 3D-positie, of kleine set manueel labelen in CVAT |
| Jetson niet op tijd | Keten ontwikkelen met TensorRT op de pc-GPU, Jetson-metingen inhalen |
| Pc kan simulatie + stream niet aan, netwerk te traag | Lagere resolutie, JPEG, kabel, netwerktijd apart meten |
| Te weinig tijd | Eerst één model volledig; minder landingen (5 per hoek); geen extra lichtcondities. FP32/FP16/INT8 blijven |

**Uitbreidingen als er tijd over is:** domain randomization (licht, texturen), quantization-aware training, test op echte foto's van een geprint platform, YOLO26-OBB voor uitlijning op de richting van het platform.

---

## 9. Beslissingslog (waarom zo)

| Beslissing | Reden |
|---|---|
| AirSim + UE 4.27 i.p.v. UE5 | Officiële AirSim werkt enkel met 4.27; beste documentatie; staat zo in contractplan |
| Python 3.9 voor AirSim | Pakket `airsim` hangt af van oude `tornado`, die crasht vanaf Python 3.10 |
| `numpy<2` | `airsim` dateert van voor NumPy 2 |
| CMake 3.21+ (VS 2022-generator), CMake 4 met `CMAKE_POLICY_VERSION_MINIMUM=3.5` | AirSim-build vereist VS 2022-generator; CMake 4 weigert oude minimumversies |
| Camera onderaan, pitch −90° | Zoals bij echte precisielanding |
| 640 × 640 | Invoergrootte van beide modellen, geen herschaling |
| FOV 90° | Bepaalt max. naderingshoek 45° |
| Gewone bounding boxes, geen OBB | RT-DETR ondersteunt in Ultralytics geen OBB; eerlijke vergelijking. Landingshoek wordt gemeten als uitkomst in AirSim |
| YOLO26s en RT-DETR-l | RT-DETR-l is kleinste RT-DETR; YOLO26s snel genoeg zonder het verschil scheef te trekken |
| Altijd FP32 + FP16 + INT8 | Consequent over alle deelvragen (opmerking van de coach); FP32 is de referentie |
| Engines bouwen op de Jetson | TensorRT-engines zijn niet overdraagbaar tussen hardware |
| TCP-sockets i.p.v. ROS 2 | Eenvoudiger, volstaat voor deze scope |
| 50 landingen per engine i.p.v. per hoek × afstand | 1920 landingen was niet haalbaar en afstand is geen startvariabele |
| Teleporteren voor datageneratie | Exact herhaalbaar, snel, geen fysica nodig |

---

## 10. Begrippen

| Begrip | Betekenis |
|---|---|
| Rekenprecisie | FP32, FP16 of INT8: aantal bits per getal in het model |
| Precision | Van alle detecties: het deel dat echt het platform is |
| Recall | Van alle keren dat het platform in beeld is: het deel dat gedetecteerd wordt |
| mAP50 | Gemiddelde nauwkeurigheid, detectie telt bij ≥ 50% overlap (IoU) met het label |
| Kalibratie | Voorbeeldbeelden door het model sturen zodat TensorRT het INT8-bereik kan bepalen |
| Inferentietijd | Tijd voor één beeld op de Jetson |
| FPS van de keten | Beelden per seconde van camera tot beslissing, inclusief netwerk |
| NED | North-East-Down: AirSim-assenstelsel, z wijst naar beneden |
| Engine | TensorRT-model, geoptimaliseerd voor één specifieke GPU |

---

## 11. Bronnen (contractplan)

1. S. Shah et al., "AirSim: High-fidelity visual and physical simulation for autonomous vehicles," Field and Service Robotics, 2018.
2. Y. Zhao et al., "DETRs beat YOLOs on real-time object detection," CVPR 2024.
3. B. Jacob et al., "Quantization and training of neural networks for efficient integer-arithmetic-only inference," CVPR 2018.
4. H. Wu et al., "Integer quantization for deep learning inference: Principles and empirical evaluation," arXiv:2004.09602, 2020.
5. J. Tobin et al., "Domain randomization for transferring deep neural networks from simulation to the real world," IROS 2017.
6. Ultralytics, TensorRT export: https://docs.ultralytics.com/integrations/tensorrt
7. Ultralytics, NVIDIA Jetson guide: https://docs.ultralytics.com/guides/nvidia-jetson
8. NVIDIA TensorRT documentation: https://docs.nvidia.com/deeplearning/tensorrt/
9. Colosseum: https://github.com/CodexLabsLLC/Colosseum
10. Ultralytics, OBB: https://docs.ultralytics.com/tasks/obb
11. PX4, Precision landing: https://docs.px4.io/main/en/advanced_features/precland.html
12. AirSim object detection: https://microsoft.github.io/AirSim/object_detection/

