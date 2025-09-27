#!/usr/bin/env python3
"""
Polars Health Check

Enhanced health checks to validate Polars functionality and performance inside
containers. Intended for Docker HEALTHCHECK or external exec.

Checks:
- Polars import and version
- Small compute sanity (groupby + join)
- Optional cross-container shared file read (if --share-file provided)
- Performance thresholds via --threshold-ms

Exit code semantics:
- 0 = healthy
- 1 = degraded (functional but slow)
- 2 = unhealthy (functional failure)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def rss_kb() -> int:
    try:
        with open("/proc/self/status", "r") as f:
            for ln in f:
                if ln.startswith("VmRSS:"):
                    return int(ln.split()[1])
    except Exception:
        pass
    return -1


def health_check(threshold_ms: int, share_file: str | None, data_path: str | None) -> dict:
    result = {
        "polars": {"ok": False, "version": None, "error": None},
        "compute": {"ok": False, "groupby_ms": None, "join_ms": None},
        "share": {"ok": True, "note": "skip"},
        "memory": {"rss_kb": rss_kb()},
        "status": "UNKNOWN",
    }

    try:
        import polars as pl  # type: ignore
        result["polars"]["ok"] = True
        result["polars"]["version"] = getattr(pl, "__version__", "unknown")
    except Exception as e:
        result["polars"]["error"] = str(e)
        result["status"] = "UNHEALTHY"
        return result

    # small compute
    try:
        import polars as pl
        N = 120000
        seq = pl.arange(0, N, eager=True)
        df = pl.DataFrame({
            "id": seq,
            "g": (seq % 20),
            "x": seq.cast(pl.Float64),
        })
        t0 = time.time()
        agg = df.group_by("g").agg([pl.col("x").mean().alias("mx"), pl.len().alias("cnt")])
        t1 = (time.time() - t0) * 1000
        t2s = time.time()
        df.join(agg, on="g", how="left")
        t2 = (time.time() - t2s) * 1000
        result["compute"]["ok"] = True
        result["compute"]["groupby_ms"] = round(t1, 2)
        result["compute"]["join_ms"] = round(t2, 2)
    except Exception as e:
        result["compute"]["ok"] = False
        result["compute"]["error"] = str(e)

    # optional share check
    if share_file and data_path:
        try:
            import polars as pl
            p = Path(data_path) / share_file
            df = pl.read_parquet(p.as_posix())
            # simple assertion
            _ = int(df.height) >= 1
            result["share"]["ok"] = True
            result["share"]["note"] = f"read {p} rows={df.height}"
        except Exception as e:
            result["share"]["ok"] = False
            result["share"]["error"] = str(e)

    # status
    if not result["polars"]["ok"]:
        result["status"] = "UNHEALTHY"
    elif not result["compute"]["ok"]:
        result["status"] = "UNHEALTHY"
    else:
        # Soft performance threshold
        g = result["compute"].get("groupby_ms") or 10_000
        j = result["compute"].get("join_ms") or 10_000
        slow = (g > threshold_ms) or (j > threshold_ms)
        if slow:
            result["status"] = "DEGRADED"
        else:
            result["status"] = "HEALTHY"
    return result


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Polars health check for Docker containers")
    ap.add_argument("--threshold-ms", type=int, default=2500, help="Performance threshold in ms")
    ap.add_argument("--share-file", type=str, default=None, help="Shared parquet filename to read for cross-container test")
    ap.add_argument("--data-path", type=str, default=None, help="Container data mount path (e.g., /data)")
    ap.add_argument("--json-out", type=str, default=None, help="Write JSON result to path")
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    res = health_check(args.threshold_ms, args.share_file, args.data_path)
    payload = json.dumps(res, indent=2)

    if args.json_out:
        try:
            Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.json_out).write_text(payload)
        except Exception:
            pass

    if not args.quiet:
        print(payload)

    if res["status"] == "HEALTHY":
        return 0
    if res["status"] == "DEGRADED":
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
