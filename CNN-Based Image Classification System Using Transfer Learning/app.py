"""Serve the local DigitLab demo with Python's standard-library WSGI server."""

import argparse
import base64
import binascii
import io
import json
import mimetypes
from functools import lru_cache
from pathlib import Path
from wsgiref.simple_server import make_server

import torch
from PIL import Image, ImageOps

from classifier import MODEL_NAMES, ROOT, load_model, predict

OUT = ROOT / "outputs"
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_REQUEST_BYTES = 7 * 1024 * 1024
FIGURES = {"training_curves.png", "confusion_matrices.png", "sample_predictions.png", "error_examples.png"}


@lru_cache(maxsize=2)
def model_for(name):
    return load_model(name)


def sample_manifest():
    return json.loads((OUT / "samples.json").read_text())


def response(start, status, content_type, content):
    start(status, [("Content-Type", content_type), ("Content-Length", str(len(content))),
                   ("X-Content-Type-Options", "nosniff")])
    return [content]


def json_response(start, payload, status="200 OK"):
    return response(start, status, "application/json", json.dumps(payload).encode())


def serve_file(start, path):
    kind = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return response(start, "200 OK", kind, path.read_bytes())


def request_body(environ):
    length = int(environ.get("CONTENT_LENGTH") or 0)
    if not 0 < length <= MAX_REQUEST_BYTES:
        raise ValueError("Choose an image under 5 MB.")
    return json.loads(environ["wsgi.input"].read(length))


def classify(body):
    if not isinstance(body, dict):
        raise ValueError("Send an image or choose a test sample.")
    name = body.get("model")
    if not isinstance(name, str) or name not in MODEL_NAMES:
        raise ValueError("Choose a supported model.")
    actual = None
    if "sample" in body:
        entry = next((sample for sample in sample_manifest() if sample["file"] == body["sample"]), None)
        if entry is None:
            raise ValueError("Choose a valid test sample.")
        image_bytes = (OUT / "samples" / entry["file"]).read_bytes()
        actual = entry["actual"]
    else:
        encoded = body.get("image", "")
        if not isinstance(encoded, str):
            raise ValueError("Choose a valid image.")
        image_bytes = base64.b64decode(encoded, validate=True)
        if not 0 < len(image_bytes) <= MAX_IMAGE_BYTES:
            raise ValueError("Choose an image under 5 MB.")
    with Image.open(io.BytesIO(image_bytes)) as opened:
        if opened.format not in {"PNG", "JPEG", "WEBP"} or max(opened.size) > 4096:
            raise ValueError("Use PNG, JPEG, or WebP with both sides at most 4,096 pixels.")
        image = ImageOps.exif_transpose(opened).convert("L")
    if body.get("invert") is True:
        image = ImageOps.invert(image)
    return {"model": MODEL_NAMES[name], "actual": actual, "scores": predict(model_for(name), image, name)}


def application(environ, start):
    path, method = environ["PATH_INFO"], environ["REQUEST_METHOD"]
    try:
        if method == "GET":
            if path in {"/", "/demo.js", "/style.css"}:
                return serve_file(start, ROOT / "demo" / (path.lstrip("/") or "index.html"))
            if path in {"/api/samples", "/api/metrics"}:
                return serve_file(start, OUT / (path.rsplit("/", 1)[-1] + ".json"))
            if path.startswith("/figures/") and path.removeprefix("/figures/") in FIGURES:
                return serve_file(start, OUT / path.removeprefix("/figures/"))
            if path.startswith("/samples/"):
                filename = path.removeprefix("/samples/")
                if any(sample["file"] == filename for sample in sample_manifest()):
                    return serve_file(start, OUT / "samples" / filename)
        if method == "POST" and path == "/api/predict":
            return json_response(start, classify(request_body(environ)))
    except FileNotFoundError:
        return json_response(start, {"error": "Training outputs are missing. Run python train.py first."}, "503 Service Unavailable")
    except (UnicodeDecodeError, binascii.Error, OSError, Image.DecompressionBombError):
        return json_response(start, {"error": "The request or image could not be decoded."}, "400 Bad Request")
    except ValueError as error:
        return json_response(start, {"error": str(error)}, "400 Bad Request")
    return json_response(start, {"error": "Page not found."}, "404 Not Found")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--sample", type=int, choices=range(10), metavar="DIGIT")
    source.add_argument("--image", type=Path)
    parser.add_argument("--model", choices=MODEL_NAMES, default="resnet")
    parser.add_argument("--invert", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.sample is not None or args.image is not None:
        body = {"model": args.model, "invert": args.invert}
        try:
            if args.sample is not None:
                entry = next(item for item in sample_manifest() if item["actual"] == args.sample)
                body["sample"] = entry["file"]
            else:
                if args.image.stat().st_size > MAX_IMAGE_BYTES:
                    raise ValueError("Choose an image under 5 MB.")
                body["image"] = base64.b64encode(args.image.read_bytes()).decode()
            result = classify(body)
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            parser.exit(1, f"{error}\n")
        print(result["model"])
        if result["actual"] is not None:
            print(f"Actual: {result['actual']}")
        print("Top 3 predictions")
        for item in result["scores"][:3]:
            print(f"  {item['digit']}   {item['score']:.2%}")
        return
    with make_server("127.0.0.1", args.port, application) as server:
        print(f"DigitLab is running at http://127.0.0.1:{server.server_port}/", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
