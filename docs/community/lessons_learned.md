# Lessons learned: AirSim + Unreal 4.27 + TensorRT on a Jetson Orin Nano

Practical problems I ran into while building a simulated drone-landing pipeline (AirSim → TCP → Jetson Orin Nano → TensorRT YOLO26s / RT-DETR-l → landing logic), and how I fixed them. Each one cost me somewhere between ten minutes and a day, so maybe this saves you that time.

Format per lesson: **Symptom** (what you see) → **Cause** (why) → **Fix** (what worked for me). Code references point to this repository.

Setup these lessons apply to: Windows 11, Unreal Engine 4.27, Microsoft AirSim built from source (Blocks environment), Python 3.9 for AirSim, Ultralytics 8.4.182, Jetson Orin Nano Super 8 GB, JetPack 6.2, TensorRT 10.7 inside the `ultralytics/ultralytics:8.4.182-jetson-jetpack6` Docker image, 15 W power mode.

**Contents**
1. [Installing the AirSim Python client](#1-installing-the-airsim-python-client)
2. [Teleporting the drone for dataset generation](#2-teleporting-the-drone-for-dataset-generation)
3. [Simulator speed and image capture](#3-simulator-speed-and-image-capture)
4. [Camera and gimbal](#4-camera-and-gimbal)
5. [Automatic labels](#5-automatic-labels)
6. [Live landing runs](#6-live-landing-runs)
7. [Building TensorRT engines on the Jetson](#7-building-tensorrt-engines-on-the-jetson)
8. [Quantization results that surprised me](#8-quantization-results-that-surprised-me)
9. [Training and evaluation](#9-training-and-evaluation)

---

## 1. Installing the AirSim Python client

### 1.1 `airsim` crashes on Python 3.10+
- **Symptom:** import errors from `tornado` / `msgpackrpc` as soon as you `import airsim`.
- **Cause:** the `airsim` package (last release 1.8.1) depends on an old `tornado` that breaks from Python 3.10 onwards.
- **Fix:** a separate venv with **Python 3.9** only for AirSim, and a second venv (Python 3.12) for PyTorch/Ultralytics. Keep the two worlds apart; talk between them through files and TCP.

### 1.2 `pip install airsim` fails or pulls NumPy 2
- **Symptom:** `ModuleNotFoundError: numpy` during installation, or runtime errors with NumPy 2.
- **Cause:** `airsim`'s `setup.py` imports NumPy *while installing*, and pip builds in an empty isolated environment. The package also predates NumPy 2. On top of that, `opencv-contrib-python` ≥ 4.12 requires NumPy 2.
- **Fix:** install in two steps (see `requirements-airsim.txt`):
  ```powershell
  python -m pip install "numpy<2" msgpack-rpc-python wheel
  python -m pip install --no-build-isolation airsim "numpy<2" "opencv-contrib-python<4.12"
  ```
  Do **not** also install `opencv-python`: both write to the same `cv2` folder and clash.

### 1.3 AirSim build fails with CMake 4
- **Symptom:** `build.cmd` stops with "Compatibility with CMake < 3.5 has been removed".
- **Cause:** CMake 4 refuses the old minimum versions in AirSim's dependencies.
- **Fix:** `set CMAKE_POLICY_VERSION_MINIMUM=3.5` before `build.cmd`, and use CMake ≥ 3.21 for the VS 2022 generator.

### 1.4 Scripts hang for an hour when Unreal is not on Play
- **Symptom:** a script just sits there.
- **Cause:** the default RPC timeout of the AirSim client is 3600 s.
- **Fix:** check the port first with a plain socket (2 s timeout) and give a clear error, and use `MultirotorClient(timeout_value=60)` (`sim/airsim_utils.py`, `check_simulator_running`).

---

## 2. Teleporting the drone for dataset generation

### 2.1 `simSetVehiclePose` does not reset velocity
- **Symptom:** images are taken further and further from the intended position (I measured 0.5 m up to **47 m** off). Labels look fine, metadata is wrong.
- **Cause:** `simSetVehiclePose` only sets the pose. The drone keeps the velocity it had, so it keeps falling, and accelerates over consecutive teleports.
- **Fix:** pause the simulation, set position **and** zero velocity together with `simSetKinematics(state, ignore_collision=True)`, then let physics run for one short step with `simContinueForTime(0.01)` so the new pose is actually applied. Measured error afterwards: ±0.3 mm. See `data_gen/generate_dataset.py`, `capture()`.

### 2.2 A 2 ms step is sometimes too short (and 10 ms too, occasionally)
- **Symptom:** a few images still at the old position.
- **Cause:** in a paused sim the pose is only applied on a physics tick. 2 ms sometimes contained no tick; 10 ms failed in 20 of 4800 images when Unreal was briefly slow.
- **Fix:** after the step, read back the pose and **retry until the drift is < 1 cm** (max 5 tries). Also store the actual drone position in the metadata, and have a `--repair` mode that regenerates only the bad images with the same θ, d, φ and yaw.

### 2.3 Pause race: the drone freezes in mid-air after teleporting
- **Symptom:** in live tests the drone sometimes hung motionless after a teleport, as if the simulation was paused.
- **Cause:** waiting for `simIsPause()` to become true after `simContinueForTime()` is not enough: the sim is *already* paused before the step starts, so the loop can exit before the step happens. AirSim then pauses again *after* my `simPause(False)`.
- **Fix:** wait until the vehicle **timestamp has changed** *and* the sim is paused. After unpausing, check that the timestamp keeps moving, and unpause again if needed (`sim/airsim_utils.py`, `teleport()`).

---

## 3. Simulator speed and image capture

### 3.1 Unreal drops to ~3 FPS when another window has focus
- **Symptom:** each `simGetImages` call takes ~330 ms as soon as you click on VS Code / your terminal.
- **Cause:** *Use Less CPU when in Background* in the editor preferences. It is a **per-project** setting, so a copied environment has it on again.
- **Fix:** *Edit → Editor Preferences → General → Performance → Use Less CPU when in Background* → off.

### 3.2 The editor caps at ~30 FPS, so every image costs 33 ms
- **Symptom:** image capture never goes below ~33 ms, even with a small scene.
- **Cause:** `simGetImages` waits for the next rendered frame, and the editor is frame-limited.
- **Fix:** at the start of every run, send console commands through the API: `client.simRunConsoleCommand("t.MaxFPS 120")` and `"r.VSync 0"` (`unlock_frame_rate()`). They reset on every new Play. Capture went from 33 ms to ~24 ms. A real drone camera has no editor frame cap, so leaving it in would unfairly slow down the measured chain.

### 3.3 Don't render a view nobody looks at
- **Fix:** `"ViewMode": "NoDisplay"` in `settings.json` stops rendering the main viewport while API cameras keep working. Less GPU load during measurements. Remove it for demos.

### 3.4 Image capture is the real bottleneck of the whole chain
- **Symptom:** FP16 → INT8 makes inference 2.6 ms faster on the Jetson, but end-to-end FPS doesn't move.
- **Cause:** per frame, getting the image out of AirSim costs ~24–28 ms; network over USB-C ~1 ms; server overhead (JPEG decode + pre/post-processing) ~7–9 ms; inference 7–36 ms. Capture is the biggest single term.
- **Fix / lesson:** measure *every* stage separately (`capture_ms`, `network_ms`, `server_ms`, `inference_ms` in `results/chain/`) before optimizing the model. See also 8.2.

---

## 4. Camera and gimbal

### 4.1 Without a gimbal the platform leaves the image at 45°
- **Symptom:** starting at θ = 45°, the platform is visible while hovering but disappears as soon as the drone tilts to fly.
- **Cause:** with FOV 90° the edge of the image is exactly 45° from vertical. A rigidly mounted camera tilts with the drone.
- **Fix:** a stabilized camera in `settings.json`: `"Gimbal": { "Stabilization": 1, "Pitch": -90, "Roll": 0, "Yaw": 0 }`. Training images were rendered with a level drone, so this gives the same kind of images.

### 4.2 Fix Yaw too, or the image spins
- **Symptom:** with only pitch and roll in the gimbal, the image rotates around its center.
- **Cause:** pitch −90° is a gimbal-lock case for Euler angles: yaw and roll become the same axis, so the stabilizer "corrects" yaw in a strange way.
- **Fix:** set `"Yaw": 0` in the gimbal as well.

### 4.3 A wrong camera name returns an empty image, not an error
- **Symptom:** `response.width == 0`, reshape fails.
- **Cause:** unknown camera names return a 0 × 0 image. Separately, AirSim *very* occasionally returns one empty image even when everything is fine.
- **Fix:** retry up to 5 times, then exit with a message that points to `settings.json` and "restart Play" (settings only apply after a new Play).

### 4.4 Images are BGR
- `simGetImages` gives BGR (like OpenCV), not RGB. Also `np.frombuffer` returns a read-only array: `.copy()` it before drawing on it.

---

## 5. Automatic labels

### 5.1 The detection API works well as a label source
- `simSetDetectionFilterRadius` (in **centimeters**), `simAddDetectionFilterMeshName(cam, ImageType.Scene, "LandingPad*")`, then `simGetDetections()` → `box2D` in pixels. Clip to the image and convert to YOLO format.
- Gotchas: boxes can extend outside the image (clip them), and a platform that is barely visible gives a 1-pixel box, so I treat boxes < 2 px as "not visible". Check labels visually with a grid (`data_gen/check_labels.py`).
- Before relying on it: verify the actor name with `simListSceneObjects("LandingPad.*")`.

### 5.2 Use a cube, not a plane, for the platform
- A plane has no proper collision volume: you can't land on it or detect contact. A 1 × 1 × 0.05 m cube works. Set it to **Movable** so a script can move it away for "no platform" tests.

### 5.3 Use power-of-two textures
- A 1024 × 1024 texture gets mipmaps in Unreal, so the H-marker stays clean at 30 m instead of shimmering.

---

## 6. Live landing runs

### 6.1 The drone's own shadow below ~2 m
- **Symptom:** confidence drops in the last meters of the descent.
- **Cause:** below ~2 m the drone's shadow falls on the platform (`results/live/close_range_view.jpg`), and the model never saw images closer than 4 m.
- **Fix:** below 3 m, land "blind" on position (last meter at 0.3 m/s). Real precision-landing systems also switch to another sensor (or a smaller marker) for the final meters. This is a **limitation** of my setup, not a solved problem.

### 6.2 Start position inside a building block
- **Symptom:** attempt 35 of a run (θ = 45°, ~14 m high) started inside one of the towers of the Blocks scene; the drone was stuck in the block and the run was useless.
- **Cause:** random start direction φ + 20 m distance can put you inside geometry.
- **Fix:** read the scene blocks via the API (`simListSceneObjects("TemplateCube.*")`, `simGetObjectPose`, `simGetObjectScale`), and rotate φ in 10° steps until both the start position and the horizontal path to the platform have 1 m clearance (`landing/live_test.py`, `scene_obstacles()`).

### 6.3 The drone "sticks" to the ground after a landing
- **Symptom:** teleport reports success, but once the sim runs the drone is back on the platform.
- **Fix:** after teleporting, hover 1 s and check the real position; if it's wrong, take off first and teleport again.

### 6.4 Long unattended runs get killed by sleep mode
- **Symptom:** a 2.5 h run stopped halfway.
- **Fix:** disable sleep, write CSV rows with a flush after every row (so partial results survive), use a lock file so two runs can't overlap, and keep interrupted runs in a separate folder instead of mixing them. Note: the console log lags behind because Python buffers stdout when piped (`python -u` helps).

### 6.5 Use the same seed for every engine
- All 6 engines get *exactly* the same 50 attempts (θ, φ, with/without platform). Otherwise differences between engines can just be differences between starting positions.

---

## 7. Building TensorRT engines on the Jetson

### 7.1 Build the engine on the device itself
- A TensorRT engine is tied to the GPU **and** the TensorRT version. Export ONNX on the PC if you like, but build `.engine` files on the Jetson, inside the container you'll run them in.

### 7.2 Shared device? Use Docker and a pinned image
- I could not install anything on the shared lab Jetson. `ultralytics/ultralytics:8.4.182-jetson-jetpack6` (TensorRT 10.7, same Ultralytics version as on my PC) worked. Pin the tag, never `latest`, so results are reproducible.
- Running as your own user (`--user`) avoids root-owned files, but then you must pass `HOME`, `USER` and `YOLO_CONFIG_DIR`, or PyTorch fails with `getpwuid(): uid not found`.

### 7.3 "CUDA out of memory" on an 8 GB Jetson with GBs free
- **Symptom:** `CUDA error: out of memory`, `NvMapMemAllocInternalTagged ... error 12`, or `Skipping tactic` warnings (→ possibly slower engines).
- **Cause:** CPU and GPU share 7.4 GB. Linux keeps read files in page cache (≈3 GB after pulling the image) and doesn't always release it fast enough for the GPU. Within one Python process memory wasn't fully released between builds either (4.7 GB free before FP32, 1.3 GB before FP16).
- **Fix:** `sync; echo 3 | sudo tee /proc/sys/vm/drop_caches` before building, build **each engine in its own process**, log free memory per build, and grep the logs for `NvMap` and `Skipping tactic`.

### 7.4 Don't zip on Windows to copy to Linux
- `Compress-Archive` stores backslashes in file names; Linux treats them as part of the name. Use `scp -r` or `tar | ssh ... tar -x`.

### 7.5 Ultralytics calibrates INT8 on the `val` split of your yaml
- So make a separate `calib.yaml` with `val: images/calib` (300 images, separate from the test set). Otherwise you calibrate on validation data, or worse, test data.

### 7.6 Fix the power mode and write it down
- The power mode sets clock speeds, so it directly changes inference time. All my measurements are at 15 W, logged next to JetPack/TensorRT/Ultralytics versions in `results/engines/summary.csv`.

---

## 8. Quantization results that surprised me

### 8.1 RT-DETR-l INT8 builds, but as a mixed-precision engine
- **Symptom:** the INT8 build succeeds, but inference is only 36 → 26 ms (FP16 → INT8), and accuracy barely changes (mAP50-95 0.974 → 0.979).
- **Cause:** the build log has 126 warnings `Missing scale and zero-point for tensor ... expect fall back to non-int8 implementation` (transformer layers like LayerNorm), and Ultralytics also keeps 8 head layers out of INT8. Those layers stay in FP16/FP32. YOLO26s INT8 had no such fallbacks (only its 25 head layers kept out).
- **Lesson:** "it exported to INT8" doesn't mean the model *runs* in INT8. Read the build log. This is also why the expected "transformers suffer more from INT8" result didn't show up: the sensitive parts were simply not quantized.

### 8.2 In the full chain, INT8 gains nothing over FP16
| YOLO26s | inference only | full chain (mean FPS) |
|---|---|---|
| FP32 | 19.0 ms | 18.5 |
| FP16 | 9.4 ms | **22.6** |
| INT8 | 6.8 ms | 22.0 |
- **Cause:** see 3.4: ~25 ms image capture + ~8 ms server overhead dominate. Saving 2.6 ms of inference is within the noise.
- **Lesson:** decide on quantization from end-to-end numbers, not from a benchmark of the model alone. Here FP16 is the sweet spot: same accuracy as FP32, ~same speed as INT8.

### 8.3 FP16 is free; INT8 costs localization accuracy on small objects
- YOLO26s FP32 and FP16 gave **identical** mAP50-95 (0.984). INT8 dropped to 0.919 overall, and to **0.843 at 30 m** (platform ≈ 11 px wide). Precision, recall and mAP50 stayed ≈ 1.0.
- **Lesson:** INT8 hurts box *tightness* first, and on small objects most. A metric with a loose IoU threshold won't show it.

### 8.4 Watch out for a ceiling effect
- On a clean synthetic dataset, mAP50 is 0.995 for every engine. That number cannot tell the engines apart. Report mAP50-95 and break results down **per distance and angle**; that's where the differences are.

---

## 9. Training and evaluation

### 9.1 Avoid leakage between train and test in synthetic data
- Train/val/calib: θ uniform in [0°, 45°], d uniform in [4, 32] m, φ ∈ {0°, 10°, …, 350°}. Test: a fixed grid θ ∈ {0, 15, 30, 45}°, d ∈ {5, 10, 20, 30} m and φ ∈ {5°, 15°, …, 355°}: never the same viewing direction as in training. The grid makes per-θ/per-d reporting easy.
- Add ~10 % **negative** images (platform out of view), or you can't test "abort".

### 9.2 Flip augmentation is valid for top-down views
- A downward camera image flipped up/down is just as realistic, and the H-marker is symmetric, so `flipud=0.5, fliplr=0.5` is safe here (not in general).

### 9.3 RT-DETR-l on a 6 GB laptop GPU
- Batch 4 (vs 16 for YOLO26s). Training was interrupted at epoch 46 and resumed from `last.pt` with `resume=True`, which restores optimizer state and epoch.
- Inference on the PC: 36 ms (RT-DETR-l) vs 15 ms (YOLO26s).

### 9.4 Ultralytics and `.engine` files: tell it the model class and task
- Ultralytics can't always infer the task or model type from an `.onnx`/`.engine`. Load RT-DETR through `RTDETR(...)` (I pick it by "rtdetr" in the file name) and pass `task="detect"` for YOLO (`jetson/detector.py`).

---

*Found something wrong, or a better fix? Please open an issue — that's exactly the kind of feedback I'm looking for.*
