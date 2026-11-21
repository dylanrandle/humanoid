# RealSense Camera Depth Models

These FP16 TensorRT engines complete sparse, RGB-aligned depth using ByteDance's camera-specific CDM checkpoints. Select the engine for the camera that produced the depth frame. Both engines were built on Triskel's Jetson Orin Nano (SM 8.7), L4T R39.2.2, with TensorRT 11.4.1.31. Rebuild for a different GPU or incompatible TensorRT version.

| Engine | Original checkpoint |
| --- | --- |
| `d405.plan` | [CDM-D405 at 7564d13](https://huggingface.co/depth-anything/camera-depth-model-d405/tree/7564d13fa0a5a6e5450a0428ad9aeb53fbd6233b) |
| `d435.plan` | [CDM-D435 at e6aa46d](https://huggingface.co/depth-anything/camera-depth-model-d435/tree/e6aa46d86d2bc623558b7d2659e5ef907252801c) |

The model weights and the converted engines are licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). This project converted the published weights to FP16, precomputed the positional embeddings for the camera resolution, and compiled the graphs with TensorRT. [Upstream implementation and authors](https://github.com/ByteDance-Seed/manip-as-in-sim-suite/tree/757b2ea26bcf7992c92ac80485d82b905e961850/cdm).

## Inference on Triskel

From a checkout with Git LFS installed, run `git lfs pull --include="models/cdm/*.plan"`. The model directory is distributed separately from the Python wheel. When deploying a wheel, copy the materialized files to `models/cdm` under the deployment data root (normally `/opt/humanoid` on Triskel), or pass their directory as `model_dir`.

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

Use a synchronized 640 by 480 RGB uint8 frame and its depth frame aligned to RGB. Obtain the depth scale from the sensor or capture metadata. For input already in metres, pass `metres_per_unit=1.0`. Zero, negative, nonfinite and over-25-metre inputs are treated as holes. Output is a new 480 by 640 float32 array of axial depth in metres, with NaN for invalid predictions. Keep an instance open across frames to reuse the execution context and buffers.

## Rebuilding

Use the pinned upstream implementation linked above and the checkpoint for the desired camera. The conversion environment needs CUDA-enabled PyTorch, torchvision, OpenCV and ONNX. The build used PyTorch 2.14.1 and ONNX 1.23.2. The inference environment only needs the project dependencies and the matching JetPack runtime.

Run from the repository root on the target Jetson, with the upstream checkout's `cdm` directory on `PYTHONPATH`. Set `CDM_PYTHON` to the conversion environment's interpreter, `CAMERA` to `d405` or `d435`, and `CHECKPOINT` to its original checkpoint path.

```sh
uv run --no-project --python "$CDM_PYTHON" python scripts/cdm/export_onnx.py \
    "$CHECKPOINT" "/tmp/cdm/$CAMERA"
trtexec --onnx="/tmp/cdm/$CAMERA/model.onnx" --saveEngine="models/cdm/$CAMERA.plan" \
    --skipInference --builderOptimizationLevel=0 --memPoolSize=workspace:256 \
    --timingCacheFile=/tmp/cdm/timing.cache
```

## Validating against PyTorch

```sh
uv run python scripts/cdm/validate.py "$CAMERA" models/cdm \
    "$CAPTURES" "$PYTORCH_REFERENCES" "$OBJECT_MASKS"
```

Each capture directory contains `scene-*/rgb.png`, `depth_in_rgb.png` and `metadata.json`. The validator compares the public inference API against full-frame PyTorch FP16 reference depths (`scene-XXX-cdm.npy`, metres) and uses the saved `scene-XXX-mask.png` to isolate the car or cube. Per-scene parity requires at least 99.9% agreement on valid pixels, mean relative error at most 1%, 99th-percentile relative error at most 3%, and median and 95th-percentile absolute differences on the object at most 2 mm and 5 mm respectively.

All eleven Triskel captures passed. The error columns below report the worst scene for each metric. Latency is the median warm call to `infer`, including preprocessing, GPU execution, and returning a full-resolution array. These measurements compare the conversion to PyTorch FP16 on the same inputs.

| Engine | Captures passed | Valid-pixel agreement | Full-frame mean relative error | Full-frame p99 relative error | Object median absolute difference | Object p95 absolute difference | Warm inference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CDM-D405 | 6/6 | 100% | 0.181% | 1.473% | 0.327 mm | 0.754 mm | 704 ms |
| CDM-D435 | 5/5 | 100% | 0.244% | 2.071% | 0.567 mm | 1.188 mm | 698 ms |
