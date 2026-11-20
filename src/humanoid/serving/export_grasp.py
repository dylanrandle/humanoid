"""Build official native GraspGenX artifacts with bounded export memory on Jetson.

Run offline in the prepared GraspGenX build environment. No robot commands.
"""

# ruff: noqa: PLC0415 -- heavyweight dependencies are offline-only.

import argparse
import json
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

STAGES = ("dense", "encoders", "weights", "gripper", "engines")


def _encoder_export(export, module, inputs, path, **kwargs):
    """Avoid Torch 2.14's collision between the Gather and graph output names."""
    import onnx

    # Both the first Gather and final output otherwise get named "embedding".
    # Export under a unique output name, then preserve the native ABI's name
    # while renaming the internal value. This changes no operators or weights.
    kwargs["output_names"] = ("humanoid_encoder_output",)
    export(module, inputs, path, **kwargs)
    graph = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(graph)
    names = {"embedding": "humanoid_input_gather", "humanoid_encoder_output": "embedding"}
    for node in graph.graph.node:
        for values in (node.input, node.output):
            for index, name in enumerate(values):
                values[index] = names.get(name, name)
    for value in (*graph.graph.input, *graph.graph.output, *graph.graph.value_info):
        value.name = names.get(value.name, value.name)
    onnx.checker.check_model(graph)
    onnx.save(graph, path)


def _load_model_weights(self, generator: str, discriminator: str) -> None:
    import torch

    for component, path in (
        (self.grasp_generator, generator),
        (self.grasp_discriminator, discriminator),
    ):
        weights = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        component.load_state_dict(weights["model"], strict=True, assign=True)


def _export_gripper(source: Path, output: Path) -> None:
    import numpy as np
    import onnxruntime as ort
    import torch
    from graspgenx.models.grasp_gen import GraspGen
    from graspgenx.utils.checkpoint_io import load_model_cfg

    release = source / "ext/graspgenx_checkpoints/release"
    config = load_model_cfg(str(release / "gen"), str(release / "dis"))
    model = GraspGen.from_config(config.diffusion, config.discriminator)
    _load_model_weights(model, config.eval.gen_checkpoint, config.eval.dis_checkpoint)
    model.eval()

    class Gripper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.generator = model.grasp_generator.gripper_encoder
            self.discriminator = model.grasp_discriminator.gripper_encoder

        def forward(self, conditioning):
            return self.generator(conditioning), self.discriminator(conditioning)

    wrapper = Gripper().eval()
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            torch.ones(1, 12),
            str(output / "gripper.onnx"),
            input_names=["conditioning"],
            output_names=["generator", "discriminator"],
            opset_version=17,
            dynamo=False,
        )
        session = ort.InferenceSession(
            str(output / "gripper.onnx"), providers=["CPUExecutionProvider"]
        )
        for scale in (0.02, 0.1, 0.2):
            conditioning = torch.ones(1, 12) * scale
            expected = wrapper(conditioning)
            actual = session.run(None, {"conditioning": conditioning.numpy()})
            for reference, exported in zip(expected, actual, strict=True):
                np.testing.assert_allclose(reference.numpy(), exported, atol=1e-5, rtol=1e-4)
    print("Gripper encoder ONNX numerical parity passed", flush=True)


def _run_stage(stage: str, source: Path, output: Path, count: int) -> None:
    tools = source / "cpp_runtime/tools"
    sys.path.insert(0, str(tools))
    release = str(source / "ext/graspgenx_checkpoints/release")
    counts = ["--diffusion-grasps", str(count), "--obb-grasps", "0"]
    commands = {
        "dense": [
            "export_onnx.py",
            "--release-root",
            release,
            "--output-dir",
            str(output / "onnx"),
            *counts,
        ],
        "encoders": [
            "export_ptv3_encoders.py",
            "--release-root",
            release,
            "--output-dir",
            str(output / "onnx/ptv3"),
        ],
        "weights": [
            "export_runtime_weights.py",
            "--release-root",
            release,
            "--output-dir",
            str(output / "weights"),
            "--diffusion-grasps",
            str(count),
        ],
        "engines": [
            "build_engines.py",
            "--plugin-library",
            str(source / "graspgenx/fast_inference/lib/libgraspgenx_runtime.so"),
            "--onnx-dir",
            str(output / "onnx"),
            "--output-dir",
            str(output / "engines"),
            *counts,
        ],
    }
    if stage != "engines":
        import torch
        from graspgenx.models.grasp_gen import GraspGen

        torch.set_num_threads(2)
        # Patch only this short-lived export process, not the upstream checkout.
        GraspGen.load_state_dict = _load_model_weights
    if stage == "gripper":
        _export_gripper(source, output)
        return
    if stage == "encoders":
        from functools import partial

        torch.onnx.export = partial(_encoder_export, torch.onnx.export)
    script, *arguments = commands[stage]
    sys.argv = [str(tools / script), *arguments]
    if stage == "engines":
        namespace = runpy.run_path(sys.argv[0])
        # Upstream's 8 GiB builder workspace exceeds this board's total RAM.
        namespace["main"].__globals__["WORKSPACE_BYTES"] = 256 * 1024**2
        namespace["main"]()
    else:
        runpy.run_path(sys.argv[0], run_name="__main__")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Pinned GraspGenX checkout")
    parser.add_argument("--output", type=Path, required=True, help="Target artifact directory")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--stage", choices=STAGES)
    args = parser.parse_args()
    source, output = args.source.expanduser().resolve(), args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.stage:
        _run_stage(args.stage, source, output, args.count)
        return
    cmake = shutil.which("cmake")
    ninja = shutil.which("ninja")
    uv = shutil.which("uv")
    if cmake is None or ninja is None or uv is None:
        raise RuntimeError("Install cmake, ninja and uv in the prepared build environment")
    build = source / "cpp_runtime/build-native"
    subprocess.run(
        [
            cmake,
            "-G",
            "Ninja",
            "-S",
            str(source / "cpp_runtime"),
            "-B",
            str(build),
            f"-DCMAKE_MAKE_PROGRAM={ninja}",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={source / 'graspgenx/fast_inference/lib'}",
            "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc",
            "-DCUDAToolkit_ROOT=/usr/local/cuda",
        ],
        check=True,
    )
    subprocess.run([cmake, "--build", str(build), "--parallel", "1"], check=True)
    for stage in STAGES:
        print(f"Exporting native GraspGenX: {stage}", flush=True)
        subprocess.run(
            [
                uv,
                "run",
                "--no-project",
                "--python",
                sys.executable,
                "python",
                "-m",
                "humanoid.serving.export_grasp",
                "--source",
                str(source),
                "--output",
                str(output),
                "--count",
                str(args.count),
                "--stage",
                stage,
            ],
            check=True,
        )
    (output / "humanoid-export.json").write_text(
        json.dumps(
            {
                "diffusion_grasps": args.count,
                "obb_grasps": 0,
                "source_commit": subprocess.check_output(
                    ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
                ).strip(),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
