# Resident model serving

Start **Pick policy** from the operator UI, wait for the models to be ready, enter
an object description, and click **Run pick**. The managed group starts
`ModelServerNode` and `PickNode`. Both inference workers load and warm up once,
remain resident across tasks, and stop when the policy process stops. Starting
the process does not move the robot; changing a prompt does not reload weights.

`ModelClient` uses private Unix sockets to send numeric arrays, without pickle.
The existing HTTP console accepts task requests. Observations and robot commands
continue through `RealtimeEnvironment` and LCM, and `PickPolicy` implements the
same `Policy` interface as teleoperation. Perception runs outside the control tick.

## Jetson setup

Defaults are in `humanoid/config/serving.py`:

- `/opt/humanoid/.venv`: robot stack and policy.
- `~/humanoid-models/.venv`: ONNX segmentation and native TensorRT grasp workers.
- `~/humanoid-models/graspgenx-venv`: offline GraspGenX build/export environment.
- `~/humanoid-models/exports/yoloe26-s`: exported vision model, text encoder,
  tokenizer, and artifact manifest.

`./scripts/deploy` updates the robot stack and each prepared model environment
listed in `humanoid/config/deployment.py` with the same Humanoid wheel. Model
environments use **`--no-deps`**; missing environments are skipped. Keep these
paths synchronized with any custom worker paths in `humanoid/config/serving.py`.
Preserve the JetPack CUDA wheels:
PyTorch `2.14.1+cu132` and torchvision `0.29.2+cu132` on the tested JetPack 7.2.1
board. Do not apply the generic development lockfile over those CUDA environments.
The `model-serving` dependency group lists the lightweight runtime dependencies;
`perception` additionally lists offline segmentation exporter dependencies. Neither
serving worker imports PyTorch. The build environments can retain it for re-export.

Run this once in the prepared segmentation environment, from `~/humanoid-models`:

```bash
uv run --no-project --python .venv/bin/python python -m humanoid.serving.export_segmentation
```

Export is an offline setup step. It checks prompt tokenization, text embeddings,
and image-model outputs against the original checkpoint before writing the
manifest. Re-export after changing the checkpoint, revision, or image size.
The worker fails clearly for missing or mismatched artifacts.

YOLOE uses `openvision/yoloe26-s-seg` revision
`86f1684b97c4f14ae1434be945be7aa172fd6987`, Ultralytics `8.4.215`, and the CLIP tokenizer
from `ultralytics/CLIP` revision `7ffa84b3bfa40c42ecc1c77147a855e69cb2dd40` during
export. The exported tokenizer is checked against that tokenizer. YOLOE and
Ultralytics use AGPL-3.0.

GraspGenX uses checkout `44eb1d77d6d27bc97ee361e3afb717a2115e30a9` with its `serve`
extra and checkpoint revision `7c834043c11a11417e31d6d5ea9355801e40a2c1` from
`adithyamurali/GraspGenXModel`. Keep `release/gen` and `release/dis` under its
`ext/graspgenx_checkpoints` directory, or set `GRASPGENX_CHECKPOINT_DIR`.
Code is Apache-2.0; weights use the NVIDIA Open Model License. The `serve` extra
supplies upstream API imports; we do not launch its server.

Build native GraspGenX artifacts once **on the target Jetson**, in its prepared
GraspGenX environment (requires JetPack CUDA/TensorRT development libraries):

```bash
uv pip install --python graspgenx-venv/bin/python cmake ninja onnx onnxscript onnxruntime
uv run --no-project --python graspgenx-venv/bin/python python -m humanoid.serving.export_grasp \
  --source GraspGenX \
  --output GraspGenX/cpp_runtime/artifacts/ddpm20-p3500-2048-1024-512-512
```

The exporter builds NVIDIA's native library, four TensorRT engines, diffusion
constants, and a small ONNX gripper encoder. It bounds build workspace to 256 MiB,
uses memory-mapped checkpoints, and checks gripper encoder numerical parity.
A naming-only workaround handles Torch 2.14's encoder-output name collision.
Each export stage runs in a separate process to release build memory. Engine plans
are specific to the GPU/TensorRT version; rebuild after changing those or the grasp
count (default 100 diffusion candidates, zero analytic OBB candidates).

## Optimization and limits

- Segmentation runs entirely in ONNX Runtime on CPU, with no PyTorch import.
  The vision graph accepts a text embedding as an input, so arbitrary object
  prompts remain supported. Standard Ultralytics exports bake in the class names
  and cannot provide this behavior. Both ONNX sessions and a bounded prompt
  embedding cache remain resident. Masks return in original RGB coordinates.
- GraspGenX uses NVIDIA's native C++/CUDA/TensorRT runtime through its C ABI,
  with reusable CUDA buffers and an ONNX gripper-conditioning encoder. Runtime
  weights/engines remain resident; no training checkpoint or PyTorch model loads.
  Encoder overflow invalidates the result. Geometric processing resolves jaw width;
  there is no heuristic grasp-pose fallback or unconditional full-close command.
  `GraspRuntime.PYTORCH` remains available as a diagnostic reference backend and
  requires setting `grasp_python` to the prepared `graspgenx-venv` interpreter.
- Both workers warm up before readiness. Warmup output is discarded. Actual
  requests use camera observations and the robot's URDF gripper conditioning.
- Startup and RPC timeouts are bounded. Inference is serialized; health queries
  remain responsive. A 1 GiB available-host-memory reserve stops workers before
  severe memory pressure, without silently reloading models.
- Display meshes load on demand in `Robot`, avoiding duplicated high-resolution
  meshes in control processes that only need kinematics and collision geometry.

On the Jetson, the ONNX-only segmentation process used about 660 MiB RSS after
several prompts, versus about 2.5 GiB for the PyTorch implementation. Vision plus
text inference took about 0.57 s per image in the isolated test. Native TensorRT
grasp inference measured about 45 ms after warmup
(previous PyTorch run: 730 ms), with about 1.36 GiB process RSS in the standalone
validation. Repeated requests from different threads returned finite, valid grasps
without importing PyTorch. These are model timings, not complete pickup latency.
Physical pickup remains untested.

With the full Jetson simulation stack running, repeated model requests reused the
same worker PIDs: segmentation took 0.60–1.06 s and grasp inference took 55–73 ms
including local RPC. The lowest sampled available memory was 1.67 GiB. UI run,
cancel, return to Idle, repeated no-object requests and worker shutdown were
verified. The policy reuses a separate perception robot and strips failed-plan
tracebacks; repeated failed requests kept its RSS near 338 MiB. The simulator
ran below its 200 Hz target (roughly 130–150 Hz), while the controller and pick
policy maintained 30 Hz outside transient startup/load effects.

YOLOE stays on ONNX CPU: the installed JetPack 7.2.1 TensorRT 10.16.2 builder
failed inside its convolution tactic library for the YOLOE graph, at optimization
levels 0 and 2. The ONNX CPU path avoids that failure and saves memory. GraspGenX
uses its separately built and validated native TensorRT engines by default.

For model-only diagnostics, run `uv run python -m humanoid.serving` from an installed
Humanoid environment. Do not run it alongside the UI-managed model server: the
second process refuses to replace a live socket.
