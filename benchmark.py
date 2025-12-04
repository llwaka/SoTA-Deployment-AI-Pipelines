import time
import asyncio
import aiohttp
import requests
import numpy as np
from pathlib import Path
from PIL import Image
import io
import argparse
import sys

# -------------------------------------------------------------
# Utilities
# -------------------------------------------------------------

def load_images(folder, max_images=5000):
    paths = list(Path(folder).glob("*.*"))
    paths = paths[:max_images]
    images = []
    for p in paths:
        try:
            images.append(Image.open(p).convert("RGB"))
        except:
            pass
    return images

def image_to_bytes(image):
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    buf.seek(0)
    return buf.getvalue()

# -------------------------------------------------------------
# Single Stream Latency Test
# -------------------------------------------------------------

def single_stream_latency(url, image, iterations=200):
    img_bytes = image_to_bytes(image)
    latencies = []

    print("\n=== Running Single Stream Latency Benchmark ===")

    # Warmup
    for _ in range(5):
        requests.post(url, files={"image": ("warmup.jpg", img_bytes, "image/jpeg")})

    # Actual test
    for _ in range(iterations):
        start = time.perf_counter()
        r = requests.post(url, files={"image": ("img.jpg", img_bytes, "image/jpeg")})
        r.raise_for_status()
        latencies.append((time.perf_counter() - start) * 1000)

    latencies = np.array(latencies)

    results = {
        "iterations": iterations,
        "avg_ms": float(latencies.mean()),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p90_ms": float(np.percentile(latencies, 90)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "min_ms":  float(latencies.min()),
        "max_ms":  float(latencies.max()),
    }

    print("=== Single Stream Latency Results ===")
    for k, v in results.items():
        print(f"{k}: {v}")

    return results

# -------------------------------------------------------------
# Concurrency Scaling Test
# -------------------------------------------------------------

async def run_worker(url, img_bytes, iterations):
    lat = []
    async with aiohttp.ClientSession() as session:
        for _ in range(iterations):
            start = time.perf_counter()
            async with session.post(url, data={"image": img_bytes}) as resp:
                await resp.read()
            lat.append((time.perf_counter() - start) * 1000)
    return lat

async def concurrency_test(url, image, concurrency, iterations):
    img_bytes = image_to_bytes(image)

    workers = [run_worker(url, img_bytes, iterations) for _ in range(concurrency)]
    results = await asyncio.gather(*workers)
    latencies = np.concatenate(results)

    total_requests = concurrency * iterations
    tot_sec = np.sum(latencies) / 1000

    out = {
        "concurrency": concurrency,
        "iterations_per_worker": iterations,
        "total_requests": total_requests,
        "throughput_img_per_sec": total_requests / tot_sec,
        "avg_ms": float(latencies.mean()),
        "p90_ms": float(np.percentile(latencies, 90)),
        "p99_ms": float(np.percentile(latencies, 99)),
    }

    print(f"\n=== Concurrency = {concurrency} ===")
    for k, v in out.items():
        print(f"{k}: {v}")

    return out

# -------------------------------------------------------------
# Main Benchmark Runner
# -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, help="Folder with images (COCO val2017)")
    parser.add_argument("--url", default="http://localhost:3000/detect")
    parser.add_argument("--single_iter", type=int, default=200)
    parser.add_argument("--concurrency_levels", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--concurrency_iter", type=int, default=50)
    args = parser.parse_args()

    # Load dataset
    images = load_images(args.folder)
    if len(images) == 0:
        print("No images loaded.")
        sys.exit(1)

    # Single fixed image for fair comparison
    image = images[0]
    print(f"Loaded 1 image for testing from dataset: {args.folder}")

    # ---------------- Single Stream -------------------
    single_results = single_stream_latency(
        url=args.url,
        image=image,
        iterations=args.single_iter
    )

    # ---------------- Concurrency ---------------------
    print("\n=== Running Concurrency Scaling Benchmark ===")
    all_concurrency_results = []

    for c in args.concurrency_levels:
        res = asyncio.run(
            concurrency_test(
                url=args.url,
                image=image,
                concurrency=c,
                iterations=args.concurrency_iter
            )
        )
        all_concurrency_results.append(res)

if __name__ == "__main__":
    main()
