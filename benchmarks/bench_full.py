#!/usr/bin/env python3
"""Multi-backend / multi-precision performance benchmark for the ggml InstantMesh
port vs the PyTorch-CUDA reference.

Measures, for every {backend x precision} on a fixed multiview input:
  time_ms   wall-clock full pipeline (ms)
  fps       throughput = 1000 / time_ms
  mem_mb    peak process RSS (from /usr/bin/time -v)
  power_w   avg power drawn during the run
              - GPU backends: polled nvidia-smi power.draw (hardware)
              - CPU backend : estimated = CPU TDP x active-CPU fraction (/proc/stat)
  wat_fps   energy efficiency = power_w / fps  (lower is better)

Backends / precision matrix:
  pytorch-cuda : PyTorch reference (fp32; f16 via autocast; q8 n/a)
  ggml-cpu     : build/instantmesh          (device cpu )
  ggml-cuda    : build-cuda/instantmesh     (device gpu )
  ggml-vulkan  : build-vk/instantmesh       (device gpu )

Usage:
  python3 benchmarks/bench_full.py [image] [--reps N]
Writes results/full_bench.csv
"""
import argparse
import csv
import os
import re
import subprocess
import sys
import threading
import time

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "benchmarks", "results")
MV = os.path.join(ROOT, "benchmarks", "mv")
CPU_TDP_W = 105.0  # AMD Ryzen 9 5950X nominal TDP (used for CPU power estimate)

BINS = {"cpu": "build/instantmesh", "cuda": "build-cuda/instantmesh",
        "vulkan": "build-vk/instantmesh"}
DEVS = {"cpu": "cpu", "cuda": "gpu", "vulkan": "gpu"}
PRECS = ["f32", "f16", "q8"]
BACKENDS = ["pytorch-cuda", "ggml-cpu", "ggml-cuda", "ggml-vulkan"]


class PowerSampler:
    """Poll nvidia-smi power.draw (GPU) or /proc/stat CPU util (CPU) in a thread."""
    def __init__(self, mode):
        self.mode = mode          # 'gpu' or 'cpu'
        self.stop = False
        self.samples = []
        self._t = None
        if mode == "cpu":
            self._prev = self._cpu_ticks()

    @staticmethod
    def _cpu_ticks():
        with open("/proc/stat") as f:
            parts = f.readline().split()
        idle = int(parts[4]) + int(parts[5])
        total = sum(int(x) for x in parts[1:])
        return total, idle

    def _cpu_util(self):
        t0, i0 = self._prev
        t1, i1 = self._cpu_ticks()
        self._prev = (t1, i1)
        dt = max(t1 - t0, 1)
        return 1.0 - (i1 - i0) / dt

    def _poll(self):
        if self.mode == "gpu":
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                capture_output=True, text=True).stdout.strip()
            try:
                self.samples.append(float(out.splitlines()[0]))
            except (ValueError, IndexError):
                pass
        else:
            self.samples.append(self._cpu_util() * CPU_TDP_W)

    def start(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        while not self.stop:
            self._poll()
            time.sleep(0.05)

    def stop_and_avg(self):
        self.stop = True
        if self._t:
            self._t.join(timeout=1)
        return sum(self.samples) / len(self.samples) if self.samples else 0.0


def run_ggml(binp, dev, prec, img, out_obj):
    cmd = [binp,
           "--dino", f"models/gguf/dino_{prec}.gguf",
           "--transformer", f"models/gguf/lrm_transformer_{prec}.gguf",
           "--synthesizer", f"models/gguf/synthesizer_{prec}.gguf",
           "--image", os.path.join(MV, img, "image.bin"),
           "--camera", os.path.join(MV, img, "camera.bin"),
           "--grid-res", "88", "--device", dev, "--out", out_obj]
    return cmd


def measure(cmd, cwd, power_mode):
    sampler = PowerSampler(power_mode)
    sampler.start()
    t0 = time.perf_counter()
    r = subprocess.run(["/usr/bin/time", "-v"] + cmd, cwd=cwd,
                       capture_output=True, text=True)
    dt_ms = (time.perf_counter() - t0) * 1000
    power = sampler.stop_and_avg()
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", r.stderr)
    rss_kb = int(m.group(1)) if m else 0
    return dt_ms, rss_kb / 1024.0, power, r.returncode


def run_pytorch(img, prec):
    cmd = [sys.executable, "benchmarks/pytorch_reference.py", img]
    if prec == "f16":
        # inject fp16 via env toggle consumed by the script
        cmd = ["env", "INSTANTMESH_FP16=1"] + cmd
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", default="cute_horse")
    ap.add_argument("--reps", type=int, default=1)
    args = ap.parse_args()
    img = args.image
    os.makedirs(RES, exist_ok=True)
    out = os.path.join(RES, "full_bench.csv")
    rows = []
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "backend", "precision", "time_ms", "fps",
                    "mem_mb", "power_w", "wat_per_fps"])
        for rep in range(args.reps):
            for prec in PRECS:
                for b in BACKENDS:
                    if b == "pytorch-cuda":
                        if prec == "q8":
                            continue  # pytorch has no q8
                        cmd = run_pytorch(img, prec)
                        dt, mem, pw, rc = measure(cmd, ROOT, "gpu")
                    else:
                        gg = b.split("-")[1]
                        out_obj = os.path.join(RES, f"_bench_{img}_{gg}_{prec}.obj")
                        cmd = run_ggml(BINS[gg], DEVS[gg], prec, img, out_obj)
                        dt, mem, pw, rc = measure(cmd, ROOT, "gpu" if gg != "cpu" else "cpu")
                    fps = 1000.0 / dt if dt > 0 else 0.0
                    wpf = pw / fps if fps > 0 else 0.0
                    row = [img, b, prec, round(dt, 1), round(fps, 2),
                           round(mem, 1), round(pw, 1), round(wpf, 3)]
                    w.writerow(row)
                    rows.append(row)
                    print(f"{img} {b:13s} {prec:4s} time={dt:7.1f}ms fps={fps:6.2f} "
                          f"mem={mem:7.1f}MB pw={pw:5.1f}W w/fps={wpf:6.3f}")
    print("wrote", out)


if __name__ == "__main__":
    main()
