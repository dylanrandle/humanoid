# Camera depth models

These FP16 TensorRT engines complete sparse, RGB-aligned depth using ByteDance's camera-specific CDM checkpoints. Select the engine for the camera that produced the depth frame. Both engines were built on Triskel's Jetson Orin Nano (SM 8.7), L4T R39.2.2, with TensorRT 11.4.1.31. Rebuild for a different GPU or incompatible TensorRT version.

| Engine | Original checkpoint |
| --- | --- |
| `d405.plan` | [CDM-D405 at 7564d13](https://huggingface.co/depth-anything/camera-depth-model-d405/tree/7564d13fa0a5a6e5450a0428ad9aeb53fbd6233b) |
| `d435.plan` | [CDM-D435 at e6aa46d](https://huggingface.co/depth-anything/camera-depth-model-d435/tree/e6aa46d86d2bc623558b7d2659e5ef907252801c) |

The model weights and the converted engines are licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). This project converted the published weights to FP16, precomputed the positional embeddings for the camera resolution, and compiled the graphs with TensorRT. [Upstream implementation and authors](https://github.com/ByteDance-Seed/manip-as-in-sim-suite/tree/757b2ea26bcf7992c92ac80485d82b905e961850/cdm).

## Inference on Triskel

Follow [Deployment](../deployment.md#deploy-the-stack) to fetch and deploy the model
engines. For a manual wheel installation, copy the materialized files to `models/cdm`
under the deployment data root, or pass their directory as `model_dir`.

`scripts/deploy` installs the locked dependencies and exposes JetPack's TensorRT bindings. For a separate Python 3.12 environment on Triskel, a `.pth` file in the virtual environment's `lib/python3.12/site-packages` containing the following line exposes the SDK bindings:

```text
/usr/lib/python3.12/dist-packages
```

```python
from humanoid.perception.depth import DepthCompleter

with DepthCompleter("d405") as model:
    model.warmup()
    metres = model.infer(rgb, depth_in_rgb, metres_per_unit=sensor_depth_scale)
```

Use synchronized RGB and depth registered to RGB, with the sensor's depth scale.
The result is estimated axial depth in metres. Keep a model instance open across
frames to reuse GPU resources; the [inference implementation](../../src/humanoid/perception/depth.py)
defines exact input shapes and invalid-value handling.

## Rebuilding

Use the pinned upstream implementation linked above and the checkpoint for the desired camera. The conversion environment needs CUDA-enabled PyTorch, torchvision, OpenCV and ONNX. The build used PyTorch 2.14.1 and ONNX 1.23.2. The inference environment only needs the project dependencies and the matching JetPack runtime.

Run from the repository root on the target Jetson, with the upstream checkout's `cdm` directory on `PYTHONPATH`. Set `CDM_PYTHON` to the conversion environment's interpreter, `CAMERA` to `d405` or `d435`, and `CHECKPOINT` to its original checkpoint path.

```bash
uv run --no-project --python "$CDM_PYTHON" python scripts/cdm/export_onnx.py \
    "$CHECKPOINT" "/tmp/cdm/$CAMERA"
trtexec --onnx="/tmp/cdm/$CAMERA/model.onnx" --saveEngine="models/cdm/$CAMERA.plan" \
    --skipInference --builderOptimizationLevel=0 --memPoolSize=workspace:256 \
    --timingCacheFile=/tmp/cdm/timing.cache
```

## Validating against PyTorch

```bash
uv run python scripts/cdm/validate.py "$CAMERA" models/cdm \
    "$CAPTURES" "$PYTORCH_REFERENCES" "$OBJECT_MASKS"
```

Each capture directory contains `scene-*/rgb.png`, `depth_in_rgb.png`, and
`metadata.json`. Supply matching PyTorch FP16 references (`scene-XXX-cdm.npy`, metres)
and object masks (`scene-XXX-mask.png`). The validator checks valid-pixel agreement,
full-frame relative error, and object depth error. Its code defines the acceptance
thresholds and reports per-scene results.

## See also

- [Documentation index](../README.md)
- [Agent perception](../agents.md#perception)
- [ONNX exporter](../../scripts/cdm/export_onnx.py)
- [Parity validator](../../scripts/cdm/validate.py)
