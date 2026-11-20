"""Offline export of promptable YOLOE and MobileCLIP; never runs during pickup."""

# ruff: noqa: PLC0415 -- export dependencies are not needed by the control stack.

import json
from pathlib import Path

import numpy as np

from humanoid.config.serving import SERVING_CONFIG
from humanoid.types.serving import ServingConfig

TOKENIZER_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"


def export(config: ServingConfig) -> None:  # noqa: PLR0915 -- sequential offline export and parity gate
    import clip
    import onnxruntime as ort
    import torch
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    from ultralytics import YOLOE
    from ultralytics.nn.modules import Detect
    from ultralytics.nn.text_model import build_text_model

    root = Path(config.segmentation_artifacts).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    # A failed export must not advertise a complete artifact set.
    (root / "manifest.json").unlink(missing_ok=True)
    torch.set_num_threads(config.segmentation_threads)
    checkpoint = hf_hub_download(
        config.segmentation_model, "model.pt", revision=config.segmentation_revision
    )
    model = YOLOE(checkpoint).model.cpu().float().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.model[-1].nc = 1
    for layer in model.modules():
        if isinstance(layer, Detect):
            layer.export = True
            layer.format = "onnx"
            layer.dynamic = False
            layer.shape = None

    class Promptable(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, images, text):
            # Raw MobileCLIP features are an input, not fused weights. The model
            # applies its learned text adapter exactly once in predict(tpe=...).
            return self.model.predict(images, tpe=text)

    images = torch.zeros(1, 3, config.image_size, config.image_size)
    text = torch.ones(1, 1, 512)
    wrapper = Promptable().eval()
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (images, text),
            str(root / "vision.onnx"),
            input_names=["images", "text"],
            output_names=["detections", "prototypes"],
            opset_version=17,
            dynamo=False,
        )
    encoder = build_text_model(model.text_model, device="cpu").encoder.eval()

    def unflatten(graph, tensor, _dimension, _sizes):
        # This pinned TorchScript checkpoint loses rank metadata at the QKV
        # unflatten during legacy ONNX export. Batch=1, context=77, heads=8,
        # width=512 are fixed here; explicit shape metadata preserves the op.
        shape = [77, 1, 3, 512]
        return graph.op(
            "Reshape", tensor, graph.op("Constant", value_t=torch.tensor(shape))
        ).setType(tensor.type().with_sizes(shape))

    torch.onnx.register_custom_op_symbolic("aten::unflatten", unflatten, 17)
    try:
        with torch.inference_mode():
            torch.onnx.export(
                encoder,
                clip.tokenize(["object"]),
                str(root / "text.onnx"),
                input_names=["tokens"],
                output_names=["embedding"],
                opset_version=17,
                dynamo=False,
            )
    finally:
        torch.onnx.unregister_custom_op_symbolic("aten::unflatten", 17)
    tokenizer = Tokenizer.from_file(
        hf_hub_download(
            "openai/clip-vit-base-patch32",
            "tokenizer.json",
            revision=TOKENIZER_REVISION,
        )
    )
    tokenizer.enable_truncation(max_length=77)
    tokenizer.enable_padding(length=77, pad_id=0, pad_token="!")
    tokenizer.save(str(root / "tokenizer.json"))
    options = ort.SessionOptions()  # ty: ignore[possibly-missing-attribute] -- pybind export
    options.intra_op_num_threads = config.segmentation_threads
    options.inter_op_num_threads = 1
    text_session = ort.InferenceSession(str(root / "text.onnx"), options)
    vision_session = ort.InferenceSession(str(root / "vision.onnx"), options)
    for prompt in ("bus", "person", "red toy car", "small wooden cube"):
        tokens = clip.tokenize([prompt])
        np.testing.assert_array_equal(tokenizer.encode(prompt).ids, tokens.numpy()[0])
        embedding = np.asarray(text_session.run(None, {"tokens": tokens.numpy()})[0])
        with torch.inference_mode():
            np.testing.assert_allclose(embedding, encoder(tokens).numpy(), atol=2e-5, rtol=2e-4)
            tensor = torch.from_numpy(embedding.reshape(1, 1, 512))
            expected = model.predict(images, tpe=tensor)
        actual = vision_session.run(None, {"images": images.numpy(), "text": tensor.numpy()})
        for found, reference in zip(actual, expected, strict=True):
            np.testing.assert_allclose(np.asarray(found), reference.numpy(), atol=2e-3, rtol=2e-3)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "format": "promptable-yoloe-v1",
                "model": config.segmentation_model,
                "revision": config.segmentation_revision,
                "image_size": config.image_size,
                "tokenizer_revision": TOKENIZER_REVISION,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Exported and verified promptable YOLOE: {root}")


if __name__ == "__main__":
    export(SERVING_CONFIG)
