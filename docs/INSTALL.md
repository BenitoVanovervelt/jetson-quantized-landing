# Installatie vanaf nul (cold start)

Doel: iemand anders kan met deze handleiding en de code de volledige opstelling opnieuw installeren. Volg de stappen in volgorde. Achter elke keuze staat waarom.

## 1. Pc (Windows)

### 1.1 Software

| Onderdeel | Versie | Waarom |
|---|---|---|
| Windows | 11 | |
| Unreal Engine | 4.27 (via Epic Games Launcher) | De officiële AirSim ondersteunt enkel UE 4.27 |
| Visual Studio 2022 | met "Desktop development with C++" | Nodig om AirSim en de Unreal-plugin te compileren |
| CMake | ≥ 3.21 | De AirSim-build gebruikt de VS 2022-generator |
| Python | 3.9 en 3.12 (python.org) | 3.9 voor AirSim, 3.12 voor training (zie 1.4 en 1.5) |
| NVIDIA-driver | recent (getest: 592.00) | CUDA voor training |

Getest met een NVIDIA GeForce RTX 3060 Laptop GPU (6 GB).

### 1.2 AirSim bouwen

1. Clone AirSim: `git clone https://github.com/microsoft/AirSim.git`.
2. Open de "Developer Command Prompt for VS 2022" in de AirSim-map en voer `build.cmd` uit.
3. Met CMake 4 faalt de build omdat CMake 4 oude minimumversies weigert. Zet dan eerst:
   ```
   set CMAKE_POLICY_VERSION_MINIMUM=3.5
   ```
4. Controleer dat `Unreal\Plugins\AirSim` aangemaakt is.

### 1.3 Scène

De Blocks-omgeving wordt gekopieerd naar de projectmap, zodat de AirSim-map onaangeroerd blijft:

1. Kopieer `AirSim\Unreal\Environments\Blocks` naar `landing-project\unreal\Blocks`.
2. Open `unreal\Blocks\Blocks.uproject` met `UE4Editor.exe`. Kies "Yes" als Unreal vraagt om modules te bouwen.
3. In Editor Preferences (**Edit → Editor Preferences → General → Performance**): zet **Use Less CPU when in Background** uit. Deze instelling geldt per project, dus ook in deze kopie opnieuw doen. Anders zakt Unreal naar ± 3 FPS zodra een ander venster actief is, en kost elk beeld van `bottom_cam` ± 330 ms.
4. Plaats het landingsplatform: zie [sectie 3](#3-landingsplatform-in-de-scène).

### 1.4 AirSim-instellingen

Kopieer deze inhoud naar `%USERPROFILE%\Documents\AirSim\settings.json`:

```json
{
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "ViewMode": "NoDisplay",
  "Vehicles": {
    "Drone1": {
      "VehicleType": "SimpleFlight",
      "Cameras": {
        "bottom_cam": {
          "CaptureSettings": [
            { "ImageType": 0, "Width": 640, "Height": 640, "FOV_Degrees": 90 }
          ],
          "X": 0.0, "Y": 0.0, "Z": 0.1,
          "Pitch": -90.0, "Roll": 0.0, "Yaw": 0.0,
          "Gimbal": { "Stabilization": 1, "Pitch": -90.0, "Roll": 0.0, "Yaw": 0.0 }
        }
      }
    }
  }
}
```

640 × 640 is de invoergrootte van beide modellen (geen herschaling nodig). FOV 90° bepaalt de maximale naderingshoek van 45°.

`"Gimbal"`: de camera blijft recht naar beneden kijken (en draait niet), ook als de drone kantelt om te vliegen. Zonder gimbal viel het platform bij θ = 45° uit beeld zodra de drone kantelde (het ligt dan op de rand van het beeld). De trainingsbeelden zijn met een horizontale drone gemaakt, dus de gimbal geeft dezelfde soort beelden. Yaw ligt ook vast: met enkel pitch en roll draaide het beeld rond zijn middelpunt (pitch −90° is een speciaal geval voor de hoeken in Unreal).

`"ViewMode": "NoDisplay"`: het hoofdvenster van Unreal rendert niet meer (het beeld staat stil), de camera's voor de API werken gewoon. Dat ontlast de GPU tijdens de metingen. Voor de demo deze regel weglaten.

**Framelimiet.** De editor beperkt Unreal tot ± 30 FPS. Omdat `simGetImages` op het volgende gerenderde frame wacht, kostte elk beeld dan 33 ms. `sim/airsim_utils.unlock_frame_rate()` zet bij de start `t.MaxFPS 120` en `r.VSync 0` (geldt tot de volgende Play); daarna ± 24 ms per beeld.

### 1.5 Python-omgeving voor AirSim (`airsim-env`, Python 3.9)

Het pakket `airsim` hangt af van een oude `tornado` die vanaf Python 3.10 crasht. Daarom een aparte omgeving met 3.9.

```powershell
cd C:\dev\landing-project
py -3.9 -m venv airsim-env
airsim-env\Scripts\activate
python -m pip install "numpy<2" msgpack-rpc-python wheel
python -m pip install --no-build-isolation airsim "numpy<2" "opencv-contrib-python<4.12"
```

Waarom in twee stappen: de setup van `airsim` importeert numpy al tijdens de installatie, dus numpy moet er eerst staan. `--no-build-isolation` zorgt dat pip die numpy gebruikt. OpenCV ≥ 4.12 vraagt NumPy 2, dus `<4.12`. Installeer niet ook `opencv-python`: het botst met `opencv-contrib-python`. Zie ook `requirements-airsim.txt`.

Test (Unreal op Play):
```powershell
python sim/test_connection.py
```
Verwacht: `results/bottom_cam_test.png`, 640 × 640, recht naar beneden.

### 1.6 Python-omgeving voor training (`train-env`, Python 3.12)

```powershell
py -3.12 -m venv train-env
train-env\Scripts\python -m pip install --upgrade pip
train-env\Scripts\python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
train-env\Scripts\python -m pip install -r requirements-train.txt
```

PyTorch komt van de eigen index van PyTorch (`cu128`), omdat de gewone pip-versie op Windows geen CUDA heeft. `requirements-train.txt` bevat de exacte versies waarmee alle resultaten gemaakt zijn.

Test:
```powershell
train-env\Scripts\python -c "import torch; print(torch.cuda.is_available())"
```
Verwacht: `True`.

Getest met torch 2.11.0+cu128 en ultralytics 8.4.182.

### 1.7 Dataset genereren

Unreal op Play, platform in de scène (sectie 3), in `airsim-env`:

```powershell
python data_gen/generate_dataset.py --scale 0.01      # mini-run van ± 50 beelden om te controleren
python data_gen/check_labels.py --split train         # raster met boxes: results/label_check/
Remove-Item -Recurse data\landingpad                  # mini-run weg, het script overschrijft niets
python data_gen/generate_dataset.py                   # volledige dataset, ± 4800 beelden
```

Het script gebruikt vaste seeds: dezelfde scène geeft exact dezelfde dataset.

**Teleporteren.** `simSetVehiclePose` zet enkel de positie, niet de snelheid. De drone blijft dan vallen, en versnelt over de beelden heen (gemeten: 0,5 tot 47 m naast de doelpositie). Daarom pauzeert het script de simulatie, zet het met `simSetKinematics` de positie en snelheid 0 tegelijk, en laat het 10 ms simulatietijd lopen (`simContinueForTime(0.01)`) zodat de positie toegepast wordt. Gemeten afwijking: ± 0,3 mm. Met 2 ms werd de positie soms niet toegepast. Na elk beeld controleert het script de positie en waarschuwt het bij een afwijking boven 1 cm.

### 1.8 Moeilijkere testsets (robuustheid)

Dezelfde 1000 testposities en labels, maar onder slechtere omstandigheden (in `airsim-env`):

```powershell
python data_gen/generate_hard_testset.py --conditions dark blur noise                  # zonder Unreal, ± 2 min
python data_gen/generate_hard_testset.py --conditions clean fog_05 fog_10 dust_05 dust_10   # Unreal op Play, ± 25 min
```

Resultaat: `data/hard/<conditie>/` (zelfde opbouw als de testset). Mist en stof komen uit het weersysteem van AirSim (`simEnableWeather`, `simSetWeatherParameter`); regen en sneeuw zijn weggelaten omdat ze in de Blocks-scène het beeld van een camera die naar beneden kijkt bijna niet veranderen. `dark`, `blur` en `noise` zijn beeldvervormingen met een vaste seed. **Gimbal:** de gewone dataset is gemaakt zonder gimbal (camera draaide mee met de yaw). Opnieuw gerenderde beelden zijn daardoor gedraaid; de weercondities krijgen daarom nieuwe labels uit `simGetDetections`, en `clean` (zelfde posities, geen weer) is hun referentie.

Evalueren op de pc (PyTorch-modellen, `train-env`):

```powershell
train-env\Scripts\python jetson/evaluate.py --model models/yolo26s.pt --precision pt --data data/hard/dark --condition dark --groups all distance --out results/robustness_pc/robustness_metrics.csv
```

Op de Jetson (engines): `data/hard/` naar `~/landing-project/data/hard/` kopiëren (met `tar -xmf`: de klok van de Jetson loopt achter), dan `DETACH=1 NAME=landing-project-hard jetson/run_in_docker.sh bash jetson/evaluate_hard_all.sh` → `results/robustness/robustness_metrics.csv`.

## 2. Jetson Orin Nano

De Jetson is een **gedeeld toestel** van school, waarop ook robotsoftware in Docker draait. Daarom wordt er **niets op het systeem zelf geïnstalleerd**: alles draait in een Docker-container, en alle projectbestanden staan in `~/landing-project`.

| Onderdeel | Versie |
|---|---|
| Toestel | Jetson Orin Nano Super Developer Kit, 8 GB gedeeld geheugen (CPU + GPU op één chip) |
| JetPack (host) | 6.2, L4T R36.4.7 |
| CUDA (host) | 12.6 |
| TensorRT (host) | 10.3 (niet gebruikt, zie container) |
| Docker-image | `ultralytics/ultralytics:8.4.182-jetson-jetpack6` (15,2 GB) |
| In de container | Python 3.10, torch 2.10.0 (CUDA), **TensorRT 10.7.0**, ultralytics 8.4.182 |
| Vermogensmodus | 15W (`nvpmodel -q`, modus 0) |

De engines worden gebouwd **in de container**, dus met TensorRT 10.7. Ze werken enkel in dezelfde container op dezelfde Jetson. Het image heeft een vaste versie (geen `latest`), zodat het resultaat reproduceerbaar is: het is dezelfde Ultralytics-versie als in `train-env` op de pc.

### 2.1 Verbinding via USB-C

Via de USB-C-kabel maakt de Jetson een netwerk met de pc: Jetson `192.168.55.1`, pc `192.168.55.100`. Inloggen met een SSH-sleutel, zodat scripts geen wachtwoord nodig hebben:

```powershell
ssh-keygen -t ed25519
type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh <gebruiker>@192.168.55.1 "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
ssh <gebruiker>@192.168.55.1
```

De scripts in `landing/` (`live_all.sh`, `chain_all.sh`, `demo_servers.sh`) lezen de gebruiker uit een bestand `jetson.env` in de projectmap. Dat bestand staat in `.gitignore`, zodat de gebruikersnaam niet in een publieke repo terechtkomt. Maak het één keer aan:
```bash
echo "JETSON=<gebruiker>@192.168.55.1" > jetson.env
```
Zonder dat bestand gebruiken de scripts `user@192.168.55.1`.

### 2.2 Internet (enkel voor de installatie)

Nodig om het Docker-image te downloaden. Op de Jetson:
```bash
nmcli device wifi list
sudo nmcli --ask device wifi connect <SSID>
```
Op het einde van het project: `sudo nmcli connection delete <SSID>`.

### 2.3 Docker-image en projectbestanden

```bash
docker pull ultralytics/ultralytics:8.4.182-jetson-jetpack6
mkdir -p ~/landing-project
```

Projectbestanden kopiëren vanaf de pc (in Git Bash, vanuit de projectmap). Gebruik `scp` of `tar`, **geen zip van Windows**: `Compress-Archive` zet backslashes in de bestandsnamen, en Linux leest die als deel van de naam in plaats van als map.
```bash
scp -r jetson models <gebruiker>@192.168.55.1:~/landing-project/
tar -cf - data/landingpad/images/{calib,test} data/landingpad/labels/{calib,test} data/landingpad/*.yaml data/landingpad/metadata.csv | ssh <gebruiker>@192.168.55.1 "cd ~/landing-project && tar -xf -"
```

Test van de container (GPU zichtbaar?):
```bash
docker run --rm --runtime nvidia --ipc=host ultralytics/ultralytics:8.4.182-jetson-jetpack6 \
  python3 -c "import torch; print(torch.cuda.is_available())"
```
Verwacht: `True`.

### 2.4 Engines bouwen

```bash
cd ~/landing-project
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches      # cache leegmaken, zie hieronder
DETACH=1 NAME=landing-project-build jetson/run_in_docker.sh bash jetson/build_all_engines.sh
```
Resultaat: `engines/<model>_<fp32|fp16|int8>.engine`, `results/engines/summary.csv` (bouwtijd, grootte, inferentietijd, vrij geheugen, versies) en een log per model. Volgen met `docker logs` of `tail -f results/engines/build_yolo26s.log`.

**Geheugen.** CPU en GPU delen 7,4 GB. Linux houdt gelezen bestanden in een cache (na het downloaden van het image: ± 3 GB). Op de Jetson geeft Linux die cache niet altijd op tijd vrij voor de GPU: dan faalt het opstarten van de GPU (`CUDA error: out of memory`, `NvMapMemAllocInternalTagged ... error 12`), of slaat TensorRT rekenmethodes over (`Skipping tactic`), wat een tragere engine kan opleveren. Daarom:
- de cache leegmaken vlak voor het bouwen (`drop_caches`: verwijdert geen bestanden, enkel hun kopie in het geheugen);
- elke engine in een **eigen proces** (`build_all_engines.sh`): binnen één proces werd het geheugen niet volledig vrijgegeven (4,7 GB vrij voor FP32, 1,3 GB voor FP16);
- `summary.csv` noteert per engine het vrije geheugen bij de start; de logs worden nagekeken op `NvMap` en `Skipping tactic`.

**Container als gewone gebruiker.** `run_in_docker.sh` start de container met `--user`, zodat de bestanden van de gebruiker blijven en niet van root. Omdat de container die gebruiker niet kent, worden `HOME`, `USER` en `YOLO_CONFIG_DIR` meegegeven (anders faalt PyTorch met `getpwuid(): uid not found`).

### 2.5 Vermogensmodus

Alle metingen gebeuren in **dezelfde** modus, want de modus bepaalt de kloksnelheid en dus de inferentietijd. Opvragen met `nvpmodel -q`. Op dit gedeelde toestel enkel wijzigen met toestemming van de eigenaar.

### Netwerk pc ↔ Jetson voor de metingen

De ketentest en de live landingen gebruiken dezelfde USB-C-verbinding (`192.168.55.1`). De server luistert op TCP-poort 5000.

### Live landingen

Unreal op Play, Jetson via USB-C. Vanuit de projectmap op de pc, in Git Bash:
```bash
bash landing/live_all.sh yolo26s_fp32 yolo26s_fp16 yolo26s_int8 rtdetr-l_fp32 rtdetr-l_fp16 rtdetr-l_int8
```
Per engine 50 pogingen (10 per hoek θ ∈ {0, 15, 30, 45}°, plus 10 zonder platform), met vaste seed: elke engine krijgt exact dezelfde pogingen. ± 30 s per poging, ± 25 min per engine. Resultaat in `results/live/`: één CSV per frame en één per poging (uitkomst, positiefout, landingssnelheid, kanteling). Eén engine testen: `python -m landing.live_test --host 192.168.55.1 --model yolo26s --precision fp16 --per-angle 1 --no-pad 1`.

### Analyse

Op de pc, in `train-env` (matplotlib komt mee met Ultralytics):
```powershell
train-env\Scripts\python analysis/make_figures.py
```
Leest de offline evaluatie (`results/offline_jetson/`), de ketentest (`results/chain/`) en de live landingen (`results/live/`, per engine het nieuwste bestand; submappen met oude runs tellen niet mee). Wat nog ontbreekt wordt overgeslagen. Resultaat: grafieken in `results/figures/` en één rij per engine in `results/summary.csv`. Met `--dark` dezelfde grafieken in donkere kleuren in `results/figures_dark/` (voor de presentatiewebsite).

### Presentatiewebsite

Na de analyse, vanuit de projectmap:
```powershell
train-env\Scripts\python analysis\make_figures.py --dark
train-env\Scripts\python presentation\make_gallery.py
python presentation\build_data.py
```
`make_gallery.py` heeft `models/*.pt` en `data/hard/` nodig (sectie 1.8). `build_data.py` gebruikt alleen de standaardbibliotheek. Daarna `presentation/index.html` openen; gebruik: [USER_GUIDE.md](USER_GUIDE.md), sectie 3c.

Live landingen: de CSV's worden na elke rij weggeschreven, dus de voortgang staat in `results/live/*_attempts.csv`; `live_all.log` loopt achter (Python buffert de terminaluitvoer). Stopt een run onverwacht (slaapstand), verwijder dan de map `results/live/live_all.lock` en zet de halve CSV's apart voor je opnieuw start.

### Ketentest

Unreal op Play, Jetson via USB-C. Vanuit de projectmap op de pc, in Git Bash:
```bash
bash landing/chain_all.sh yolo26s_fp32 yolo26s_fp16 yolo26s_int8 rtdetr-l_fp32 rtdetr-l_fp16 rtdetr-l_int8
```
Per engine start het script de server in de container op de Jetson, wacht tot de engine geladen is, en meet 20 opwarmframes + 500 frames met de drone op 10 m boven het platform. Resultaat: één CSV per engine in `results/chain/` (tijd per stap: beeld ophalen, netwerk, server, inferentie).

## 3. Landingsplatform in de scène

De Unreal-editor in `unreal/Blocks/Blocks.uproject` (niet op Play):

1. **Textuur.** Content Browser → nieuwe map `LandingPad` → **Import** → `assets/landingpad_H.png` (genereren met `python sim/make_landingpad_texture.py`). 1024 × 1024 is een macht van 2: Unreal maakt er mipmaps van, zodat het platform ook op 30 m scherp blijft.
2. **Materiaal.** Rechtsklik op de textuur → **Create Material** → `M_LandingPad`. Textuur aan **Base Color**; een Constant `0.9` aan **Roughness** (mat oppervlak, geen schittering). Apply, Save.
3. **Platform.** Place Actors → Basic → **Cube** in de viewport.
   - Scale X = 1, Y = 1, Z = 0.05 → 1 m × 1 m × 5 cm (de basiskubus is 100 cm). Een kubus en geen Plane: een echte botsingsvorm, nodig om te landen en contact te meten.
   - Location X = PlayerStart-X + 1000, Y = PlayerStart-Y (hier: X = 6690, Y = −100), dan **End** om op de vloer te zetten. Het platform ligt zo 10 m voor de startplek (AirSim NED x = 10, y = 0).
   - Materials → Element 0 → `M_LandingPad`.
   - Mobility → **Movable**, zodat een script het platform later kan wegzetten voor de landingen zonder platform.
   - World Outliner → F2 → naam `LandingPad`.
4. **Ctrl + S**.

Controle via de API (Unreal op Play): `simGetObjectScale("LandingPad")` geeft `1, 1, 0.05`; `simGetObjectPose("LandingPad")` geeft x = 10, y = 0.
- Test: `python sim/test_landingpad.py` → `results/landingpad_detection.png` met een groene box rond het platform.
