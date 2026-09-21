"""Command line: serve, doctor, generate."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from pictor import __version__
from pictor.config import Settings, load_settings


def _logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


def cmd_serve(args: argparse.Namespace, settings: Settings) -> int:
    import uvicorn

    from pictor.app import create_app

    if args.host:
        settings.host = args.host
    if args.port:
        settings.port = args.port
    if args.device:
        settings.device = args.device
    if args.no_preload:
        settings.preload = False
    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info", timeout_keep_alive=120)
    return 0


def cmd_doctor(args: argparse.Namespace, settings: Settings) -> int:
    import platform

    print(f"Pictor {__version__} on {platform.system()} {platform.machine()} python {platform.python_version()}")
    ok = True
    try:
        import torch

        dev = "cuda" if torch.cuda.is_available() else (
            "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
        print(f"torch {torch.__version__}: device {dev} (PICTOR_DEVICE={settings.device})")
        from diffusers import QwenImage21Pipeline  # noqa: F401

        print("diffusers QwenImage21Pipeline: ok — text-to-image, instruction editing (≤10 refs), RGBA")
        if dev == "cpu":
            print("  warning: no GPU detected, generation will be extremely slow")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"torch/diffusers import failed: {e}\n  -> uv sync")
    try:
        from huggingface_hub import scan_cache_dir

        repos = {r.repo_id: r for r in scan_cache_dir().repos}
        r = repos.get(settings.model_id)
        status = f"{r.size_on_disk / 1e9:.1f} GB in the HF cache" if r else "not downloaded yet (automatic on first start, ~33 GB)"
        print(f"weights {settings.model_id}: {status}")
    except Exception:  # noqa: BLE001
        pass
    print(f"quantization: {settings.quant}, cpu offload: {settings.cpu_offload}")
    print("images are returned to the client only; nothing is written to disk by the server")
    return 0 if ok else 1


def cmd_generate(args: argparse.Namespace, settings: Settings) -> int:
    """Quick local test without the HTTP server."""
    from pictor.backends import create_backend
    from pictor.backends.base import GenerationRequest
    from pictor.util.images import open_image, size_like

    backend = create_backend(settings)
    refs = [open_image(Path(p).expanduser().read_bytes()) for p in (args.image or [])]
    if refs and not (args.width and args.height):
        width, height = size_like(refs[-1], base=settings.default_width, max_side=settings.max_side)
    else:
        width, height = args.width or settings.default_width, args.height or settings.default_height
    req = GenerationRequest(
        prompt=args.prompt, width=width, height=height, steps=args.steps or settings.default_steps,
        seed=args.seed if args.seed is not None else 42, references=refs, guidance=settings.default_guidance,
        transparent=args.transparent,
    )

    def progress(stage: str, step: int, total: int) -> None:
        print(f"\r{stage} {step}/{total}", end="", flush=True)

    t0 = time.time()
    result = backend.generate(req, progress, lambda: False)
    print()
    out = Path(args.out).expanduser()
    result.image.save(out)
    print(f"{out} ({result.width}x{result.height}, seed {result.seed}, {time.time() - t0:.0f}s)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pictor", description="OpenAI-compatible gateway for local image models")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("serve", help="run the HTTP server")
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--no-preload", action="store_true", help="load the model on the first request instead of at start")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("doctor", help="check the installation")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("generate", help="generate or edit one image from the command line (debug)")
    p.add_argument("prompt")
    p.add_argument("--image", action="append", help="reference image (repeatable, up to 10)")
    p.add_argument("--width", type=int)
    p.add_argument("--height", type=int)
    p.add_argument("--steps", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--transparent", action="store_true")
    p.add_argument("--out", default="pictor-out.png", help="where to write the PNG (default: ./pictor-out.png)")
    p.set_defaults(func=cmd_generate)

    args = parser.parse_args(argv)
    _logging(args.verbose)
    settings = load_settings()
    return args.func(args, settings)


if __name__ == "__main__":
    sys.exit(main())
