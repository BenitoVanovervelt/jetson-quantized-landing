# Notice

Copyright (C) 2026 Benito Vanovervelt

This program is free software: you can redistribute it and/or modify it under the terms of the
GNU Affero General Public License as published by the Free Software Foundation, version 3
(SPDX: `AGPL-3.0-only`). It is distributed WITHOUT ANY WARRANTY; see [LICENSE](LICENSE).

## What falls under which licence

| Part | Licence |
|---|---|
| Source code, documentation and results in this repository | AGPL-3.0-only |
| Fine-tuned model weights (YOLO26s, RT-DETR-l), published separately | AGPL-3.0-only (derived from Ultralytics models) |
| Dataset (4,800 rendered images with labels), published separately | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

## Third-party software (used, not included in this repository)

- **Ultralytics YOLO / RT-DETR**: AGPL-3.0, https://github.com/ultralytics/ultralytics
- **Microsoft AirSim**: MIT License, Copyright (c) Microsoft Corporation, https://github.com/microsoft/AirSim
- **Unreal Engine 4.27**: Epic Games EULA. The Unreal scene is not redistributed; `docs/INSTALL.md` explains how to rebuild it.
- **NVIDIA TensorRT / JetPack**: NVIDIA licence. TensorRT engines are not redistributed; they are built on the device itself.
- **PyTorch** (BSD-3-Clause) and **OpenCV** (Apache-2.0): installed via pip, not included.
