#!/usr/bin/env python3
"""
Run moreau-cuda Riccati benchmark on H100 via Modal.

Usage:
    modal run benchmark_h100.py
"""

import modal
from pathlib import Path

app = modal.App("moreau-cuda-riccati-h100")

LOCAL_DIR = Path(__file__).parent

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("cmake", "build-essential", "libgtest-dev", "pkg-config", "wget")
    .run_commands(
        "wget -q https://developer.download.nvidia.com/compute/cudss/redist/libcudss/linux-x86_64/libcudss-linux-x86_64-0.7.1.4_cuda12-archive.tar.xz",
        "tar xf libcudss-linux-x86_64-*.tar.xz",
        "cp -r libcudss-linux-x86_64-*/lib/* /usr/local/lib/",
        "cp -r libcudss-linux-x86_64-*/include/* /usr/local/include/",
        "ldconfig",
        "rm -rf libcudss-*",
    )
    .add_local_dir(
        str(LOCAL_DIR),
        remote_path="/opt/moreau-cuda",
        copy=True,
        ignore=[
            ".git",
            "build",
            "build_riccati",
            "__pycache__",
            "*.pyc",
            "venv",
            "moreau_cuda/*.so",
            "moreau_cuda/*.egg-info",
        ],
    )
    .run_commands(
        "cd /opt/moreau-cuda && "
        "mkdir -p build && cd build && "
        "MOREAU_CUDA_ARCH=90 cmake .. -DCMAKE_BUILD_TYPE=Release "
        "-DMOREAU_BUILD_PYTHON=OFF -DMOREAU_BUILD_EXAMPLES=OFF && "
        "make -j$(nproc) test_riccati"
    )
)


@app.function(image=image, gpu="H100", timeout=600)
def benchmark():
    import subprocess

    # Print GPU info
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=gpu_name,memory.total,clocks.max.graphics",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
    )
    print(f"GPU: {result.stdout.strip()}")

    # Run Riccati benchmark
    result = subprocess.run(
        ["/opt/moreau-cuda/build/test_riccati", "--gtest_filter=RiccatiTest.Benchmark"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)


@app.local_entrypoint()
def main():
    print(benchmark.remote())
