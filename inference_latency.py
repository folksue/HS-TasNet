"""
Measure HS-TasNet inference latency without audio I/O.
"""

import argparse
import time
import numpy as np
import torch

from hs_tasnet import HSTasNet


def parse_args():
    parser = argparse.ArgumentParser(description="HS-TasNet inference latency benchmark")
    parser.add_argument("--model-path", type=str, default=None, help="Path to checkpoint (optional)")
    parser.add_argument("--frames", type=int, default=None, help="Frames per chunk (default: model.overlap_len)")
    parser.add_argument("--iters", type=int, default=100, help="Number of timed iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--sources", type=str, default="0", help="Comma separated source indices to keep")
    parser.add_argument("--large", action="store_true", help="Use large model variant (default: small)")
    parser.add_argument("--mono", action="store_true", help="Force mono model instead of stereo")
    parser.add_argument("--device", type=str, default=None, help="Device override (cpu or cuda)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.model_path:
        print(f"Loading model from {args.model_path}...")
        model = HSTasNet.init_and_load_from(args.model_path)
    else:
        model = HSTasNet(small=not args.large, stereo=not args.mono)
        print("Initialized default model (random weights).")

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    sources = [int(s.strip()) for s in args.sources.split(",") if s.strip() != ""]
    if len(sources) == 0:
        print("Provide at least one source index with --sources")
        return

    frames = args.frames if args.frames is not None else int(model.overlap_len)
    channels = int(model.audio_channels)
    x = np.random.randn(channels, frames).astype("float32")

    fn = model.init_stateful_transform_fn(
        return_reduced_sources=sources,
        device=device,
        auto_convert_to_stereo=True,
    )

    with torch.inference_mode():
        for _ in range(args.warmup):
            fn(x)

        times_ms = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            fn(x)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    times_ms.sort()
    avg_ms = sum(times_ms) / len(times_ms)
    p50 = times_ms[len(times_ms) // 2]
    p95 = times_ms[int(len(times_ms) * 0.95) - 1]

    print(f"Device: {device}")
    print(f"Frames: {frames}")
    print(f"Channels: {channels}")
    print(f"Iters: {args.iters} (warmup {args.warmup})")
    print(f"Avg: {avg_ms:.3f} ms  P50: {p50:.3f} ms  P95: {p95:.3f} ms")


if __name__ == "__main__":
    main()
