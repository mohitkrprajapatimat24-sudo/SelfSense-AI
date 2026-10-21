"""Edge benchmark: latency per image, peak memory, model sizes, language-model time.

    python -m shelfsense.eval.benchmark
    docker run --rm --cpus 4 --memory 8g --network none shelfsense benchmark     # inside the edge container

Reports the CPU/RAM limits it actually ran under and whether the internet was
reachable, so the numbers can be quoted as "offline, 4 vCPU / 8 GB".
"""

import argparse
import json
import os
import platform
import socket
import time
from pathlib import Path

import numpy as np
import psutil

from ..paths import ASSETS, MODELS, ROOT, SYNTHETIC

TARGET_S = 3.0


def limits() -> dict:
    """CPU and memory available to this process (cgroup limits inside a container)."""
    cpus, mem = os.cpu_count(), psutil.virtual_memory().total
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            cpus = round(int(quota) / int(period), 2)
    except (OSError, ValueError):
        pass
    try:
        m = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if m != "max":
            mem = int(m)
    except (OSError, ValueError):
        pass
    return {"cpus": cpus, "memory_gb": round(mem / 2**30, 1), "os": f"{platform.system()} {platform.machine()}"}


def online() -> bool:
    try:
        socket.create_connection(("1.1.1.1", 443), timeout=2).close()
        return True
    except OSError:
        return False


def tree_rss() -> int:
    """Resident memory of this process and its children (e.g. llama-server)."""
    p = psutil.Process()
    return p.memory_info().rss + sum(c.memory_info().rss for c in p.children(recursive=True) if c.is_running())


def images(n: int) -> list[tuple[Path, str]]:
    found = []
    samples = ASSETS / "samples" / "samples.json"
    if samples.exists():
        found += [(ASSETS / "samples" / s["file"], s["planogram_id"])
                  for s in json.loads(samples.read_text()) if not s.get("video")]
    answers = sorted((SYNTHETIC / "test" / "answers").glob("*.json"))
    for f in answers[: max(0, n - len(found))]:
        meta = json.loads(f.read_text(encoding="utf-8"))
        found.append((SYNTHETIC / "test" / "images" / meta["image_file"], meta["planogram_id"]))
    return [found[i % len(found)] for i in range(n)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=20, help="images to time")
    ap.add_argument("--slm-runs", type=int, default=3)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SHELFSENSE_THREADS", "4")))
    args = ap.parse_args()

    from ..perception.preprocess import load_image
    from ..service import OUTPUTS, AuditService

    out_dir = Path(os.environ.get("SHELFSENSE_OUTPUTS", OUTPUTS))
    env = {**limits(), "threads": args.threads, "internet": "reachable" if online() else "none"}
    print(f"Environment: {env}")

    t0 = time.perf_counter()
    svc = AuditService(folder=out_dir / "_benchmark", threads=args.threads)
    svc.perception
    load_s = time.perf_counter() - t0

    work = images(args.n)
    lat, stages, peak = [], [], tree_rss()
    records = []
    for i, (path, pid) in enumerate(work):
        img = load_image(path)
        t = time.perf_counter()
        rec = svc.audit_image(img, pid, source=path.name)
        lat.append(time.perf_counter() - t)
        stages.append(rec.timings_ms)
        records.append(rec)
        peak = max(peak, tree_rss())
    lat = lat[1:] if len(lat) > 1 else lat                  # first run warms caches
    stage_ms = {k: round(float(np.mean([s[k] for s in stages[1:] or stages])), 1)
                for k in ("detect", "identify", "ocr", "audit")}

    slm = {"available": False}
    t0 = time.perf_counter()
    if svc.writer.available():
        start_s = time.perf_counter() - t0
        times, sources = [], []
        for rec in [r for r in records if r.result.violations][:args.slm_runs]:
            rec = svc.write_tasks(rec)
            times.append(rec.tasks_ms / 1000)
            sources.append(rec.tasks_source)
            peak = max(peak, tree_rss())
        slm = {"available": True, "server_start_s": round(start_s, 1),
               "seconds_mean": round(float(np.mean(times)), 1) if times else None, "lists": sources}
        svc.writer.server.stop()

    sizes = {name: round(p.stat().st_size / 1e6, 1) for name, p in {
        "detector (YOLO11n ONNX)": MODELS / "detector" / "shelfsense_yolo11n.onnx",
        "product embedder (DINOv2-small int8 ONNX)": MODELS / "dinov2-small-int8.onnx",
        "product gallery (FAISS)": MODELS / "gallery" / "gallery.faiss",
        "language model (Qwen2.5-3B Q4_K_M)": MODELS / "qwen2.5-3b-instruct-q4_k_m.gguf"}.items() if p.exists()}

    rep = {"environment": env, "models_load_s": round(load_s, 1), "images": len(lat),
           "latency_s_mean": round(float(np.mean(lat)), 2), "latency_s_p90": round(float(np.percentile(lat, 90)), 2),
           "latency_s_max": round(float(np.max(lat)), 2), "stage_ms_mean": stage_ms,
           "peak_memory_gb": round(peak / 2**30, 2), "model_sizes_mb": sizes, "language_model": slm,
           "meets_3s_target": float(np.mean(lat)) <= TARGET_S}

    print(f"\nShelfSense edge benchmark ({env['cpus']} CPUs, {env['memory_gb']} GB, internet {env['internet']})")
    print(f"  models loaded in             {rep['models_load_s']} s")
    print(f"  audit time per image         {rep['latency_s_mean']} s mean, {rep['latency_s_p90']} s p90 "
          f"(target <= {TARGET_S:.0f} s, language model excluded) over {len(lat)} images")
    print(f"  stages (ms)                  {stage_ms}")
    if slm["available"]:
        print(f"  task list by Qwen2.5-3B      {slm['seconds_mean']} s per audit (server start {slm['server_start_s']} s)")
    else:
        print("  task list by Qwen2.5-3B      not available, template used")
    print(f"  peak memory (all processes)  {rep['peak_memory_gb']} GB")
    for k, v in sizes.items():
        print(f"  {k:<44}{v:>8.1f} MB")

    out = out_dir / "benchmark.json" if os.environ.get("SHELFSENSE_OUTPUTS") else ROOT / "reports" / "benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
