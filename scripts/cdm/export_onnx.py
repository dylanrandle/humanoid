"""Export a fixed-resolution FP16 CDM checkpoint with external ONNX weights on a Jetson."""

import argparse
from functools import partial
from pathlib import Path
from unittest.mock import patch

import onnx  # ty: ignore[unresolved-import] -- Jetson export environment
import torch  # ty: ignore[unresolved-import] -- Jetson export environment
from onnx import (  # ty: ignore[unresolved-import] -- Jetson export environment
    TensorProto,
    external_data_helper,
    helper,
)
from rgbddepth.dpt import RGBDDepth  # ty: ignore[unresolved-import] -- Jetson export environment


def load(checkpoint: Path):
    # On Orin, allocating both complete FP32 encoders before loading exceeds available RAM.
    with (
        torch.device("meta"),
        patch.object(torch, "linspace", partial(torch.linspace, device="cpu")),
    ):
        model = RGBDDepth()
    state = torch.load(checkpoint, weights_only=True, mmap=True, map_location="cpu")["state_dict"]
    weights = {
        name: state[f"pipeline.{name}"].to(
            device="cuda", dtype=torch.float16 if param.is_floating_point() else param.dtype
        )
        for name, param in model.state_dict().items()
    }
    model.load_state_dict(weights, assign=True, strict=True)
    return model.eval()


def attach_weights(graph, model, output: Path) -> None:
    # Stream each initializer to disk. A single protobuf would require several extra GB of RAM.
    params = model.state_dict()
    with (output / "weights.bin").open("wb") as stream:
        for value in list(graph.graph.input)[1:]:
            array = params[value.name].detach().cpu().contiguous().numpy()
            tensor = TensorProto()
            tensor.name = value.name
            tensor.data_type = helper.np_dtype_to_tensor_dtype(array.dtype)
            tensor.dims.extend(array.shape)
            tensor.raw_data = b"0"  # ONNX requires raw_data before setting the external location.
            external_data_helper.set_external_data(
                tensor, location="weights.bin", offset=stream.tell(), length=array.nbytes
            )
            tensor.ClearField("raw_data")
            stream.write(memoryview(array).cast("B"))
            graph.graph.initializer.append(tensor)
            graph.graph.input.remove(value)
    onnx.save(graph, output / "model.onnx")
    onnx.checker.check_model(str(output / "model.onnx"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    model = load(args.checkpoint)
    image = torch.ones(1, 4, 518, 686, device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        # The trained 37 x 37 position grid is always interpolated to the same patch grid.
        for encoder in (model.pretrained, model.depth_pretrained):
            tokens = image.new_empty((1, 1, encoder.embed_dim))
            encoder.register_buffer("fixed_pos", encoder.interpolate_pos_encoding(tokens, 518, 686))
            encoder.interpolate_pos_encoding = lambda _x, _h, _w, enc=encoder: enc.fixed_pos
        torch.backends.mha.set_fastpath_enabled(False)
        topology = args.output / "graph.onnx"
        torch.onnx.export(
            model,
            (image,),
            str(topology),
            dynamo=False,
            opset_version=17,
            export_params=False,
            do_constant_folding=False,
            input_names=["rgbd"],
            output_names=["inverse_depth"],
        )
    attach_weights(onnx.load(topology), model, args.output)


if __name__ == "__main__":
    main()
