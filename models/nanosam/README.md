# NanoSAM on Triskel

These TensorRT engines run NVIDIA NanoSAM on the Jetson Orin Nano Super
(SM 8.7, L4T R39.2.2, TensorRT 11.4.1.31). The ResNet-18 encoder uses FP16
internally with float32 bindings. The MobileSAM decoder accepts point and box
prompts and a previous mask's logits. The decoder profile supports 2 through
16 coordinate tokens, including padding for a single point. [NVIDIA NanoSAM](https://github.com/NVIDIA-AI-IOT/nanosam)

Fetch the materialized engines with `git lfs pull --include="models/nanosam/*.plan,models/cdm/*.plan"`.
When deploying the Python wheel, copy both model directories under the dashboard
host's data root (normally `/opt/humanoid/models`). RGB segmentation uses
`nanosam/encoder.plan` and `nanosam/decoder.plan`. The optional `complete_depth`
tool uses the matching `cdm/d435.plan` or `cdm/d405.plan`.

On a compatible Jetson, run the encoder and decoder smoke test with
`HUMANOID_RUN_NANOSAM_TESTS=1 uv run pytest -q tests/perception/test_segmenter.py`.

## Build on a compatible Jetson

| Input | SHA-256 of source ONNX |
| --- | --- |
| [ResNet-18 image encoder](https://raw.githubusercontent.com/johnnynunez/nanosam/main/data/resnet18_image_encoder.onnx) | `d266fcdc9e4f0182b59946cd3cf1be331641f1d0d1de2415c4fd94e7a1c9cf0a` |
| [MobileSAM mask decoder](https://huggingface.co/dragonSwing/nanosam/resolve/main/mobile_sam_mask_decoder.onnx) | `41e49a298099048186ce109a4518286b8972959898a02577414405efa5c3b247` |

Convert the encoder ONNX with
`onnxruntime.transformers.float16.convert_float_to_float16(model, keep_io_types=True)`
and save it as `encoder-fp16.onnx`. Compile the encoder and original decoder on
the target GPU with the installed TensorRT runtime.

```sh
trtexec --onnx=encoder-fp16.onnx --saveEngine=encoder.plan --skipInference --memPoolSize=workspace:512MiB
trtexec --onnx=mobile_sam_mask_decoder.onnx --saveEngine=decoder.plan --skipInference \
  --minShapes=point_coords:1x2x2,point_labels:1x2 \
  --optShapes=point_coords:1x2x2,point_labels:1x2 \
  --maxShapes=point_coords:1x16x2,point_labels:1x16
```
