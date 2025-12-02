import time
import json
import argparse
import requests
from pathlib import Path
from PIL import Image
import io
import numpy as np


def load_images(folder, max_images=500):
    images = []
    for p in Path(folder).glob("*.*"):
        if len(images) >= max_images:
            break
        try:
            img = Image.open(p).convert("RGB")
            images.append(img)
        except:
            pass
    return images

def image_to_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.getvalue()

def warmup(url, sample): # To warmup the server
    for _ in range(5):
        requests.post(url, files={"image": ("warmup.jpg", image_to_bytes(sample), "image/jpeg")})

def benchmark_per_request(url, images):
    warmup(url, images[0])

    latencies = []
    for image in images:
        img_bytes = image_to_bytes(image)
        start = time.time()
        response = requests.post(url, files={"image": ("img.jpg", img_bytes, "image/jpeg")})
        response.raise_for_status()
        latencies.append(time.time() - start)

    return {"latency_90th_percentile": np.percentile(latencies, 90),
            "average_latency": np.mean(latencies),
            "throughput_img_per_sec": len(images) / sum(latencies)}


def benchmark_batch(url, images, batch_size):
    warmup(url, images[0])
    latencies = []
    total = 0

    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]

        start = time.time()
        for image in batch:
            response = requests.post(url, files={"image": ("img.jpg", image_to_bytes(image), "image/jpeg")})
            response.raise_for_status()

        elapsed = time.time() - start
        latencies.append(elapsed)
        total += len(batch)

    mean_latency = np.mean(latencies)


    return {
        "batch": batch_size,
        "latency_per_request": mean_latency,
        "latency_per_image": mean_latency / batch_size,
        "throughput_img_per_sec": total / sum(latencies),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, help="folder with images")
    parser.add_argument("--sizes", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--url", default="http://localhost:3000/detect")
    args = parser.parse_args()

    images = load_images(args.folder)
    print(f"Loaded {len(images)} images")

    results = []
    results_batch = []

    for bs in args.sizes:
        r = benchmark_batch(args.url, images, bs)
        results_batch.append(r)

    print("\n=== One Sample Request Benchmark Results ===")
    res = benchmark_per_request(args.url, images)

    print(
        f"Lat 90th perc.: {res['latency_90th_percentile']*1000:6.1f} ms | "
        f"Avg Lat: {res['average_latency']*1000:6.1f} ms | "
        f"Throughput: {res['throughput_img_per_sec']:8.1f} img/s"
    )
    print("\n")
    

    print("\n=== Batch Benchmark Results ===")
    for r in results_batch:
        print(
            f"Batch {r['batch']:>2d} | "
            f"Lat/req: {r['latency_per_request']*1000:6.1f} ms | "
            f"Lat/img: {r['latency_per_image']*1000:6.1f} ms | "
            f"Throughput: {r['throughput_img_per_sec']:8.1f} img/s"
        )


if __name__ == "__main__":
    main()
