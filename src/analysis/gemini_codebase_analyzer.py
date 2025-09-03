"""
Gemini-Enhanced Codebase Analyzer for IFRS9

Offline-first repository analysis with optional Gemini (Vertex AI) enrichment.
Safe to run without network; when configured, uses Vertex AI GenerativeModel.
"""

from __future__ import annotations

import os
import re
import json
import math
import glob
import time
import hashlib
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

# Optional imports (guarded)
try:
    import vertexai
    from vertexai.generative_models import GenerativeModel, GenerationConfig
    _HAS_VERTEX = True
except Exception:
    _HAS_VERTEX = False


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class RepoStats:
    total_files: int
    total_py_files: int
    total_loc: int
    avg_file_size_kb: float
    largest_files: List[Tuple[str, int]]
    todo_count: int
    test_coverage_files: int
    has_ci: bool
    has_docker: bool
    has_k8s: bool
    has_monitoring: bool
    has_docs: bool
    risk_hotspots: List[str]


@dataclass
class AnalyzerConfig:
    root: str = "."
    max_file_bytes: int = 200_000
    include_patterns: Tuple[str, ...] = ("**/*.py", "**/*.yml", "**/*.yaml", "**/*.toml")
    exclude_dirs: Tuple[str, ...] = (".git", "__pycache__", "ifrs9_validation_env", ".venv", ".serena", ".claude")
    gemini_model: str = "gemini-1.5-pro-002"
    gcp_project: Optional[str] = None
    gcp_location: str = "us-central1"
    credentials_path: Optional[str] = None
    enable_gemini: bool = False


class CodebaseAnalyzer:
    def __init__(self, cfg: AnalyzerConfig):
        self.cfg = cfg
        self.root_path = Path(cfg.root).resolve()
        self.files: List[Path] = []
        self.vertex_model = None
        if cfg.enable_gemini and _HAS_VERTEX:
            try:
                creds_path = cfg.credentials_path or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
                if creds_path and Path(creds_path).exists():
                    # vertexai.init can take credentials kwarg in recent versions
                    vertexai.init(project=cfg.gcp_project, location=cfg.gcp_location)
                else:
                    vertexai.init(project=cfg.gcp_project, location=cfg.gcp_location)
                self.vertex_model = GenerativeModel(cfg.gemini_model)
                logger.info("Gemini model initialized for codebase analysis")
            except Exception as e:
                logger.warning(f"Gemini initialization failed, continuing offline: {e}")
                self.vertex_model = None

    def scan(self) -> RepoStats:
        def is_excluded(p: Path) -> bool:
            parts = set(p.parts)
            return any(d in parts for d in self.cfg.exclude_dirs)

        candidates: List[Path] = []
        for pat in self.cfg.include_patterns:
            candidates.extend([Path(p) for p in glob.glob(str(self.root_path / pat), recursive=True)])
        files = [p for p in candidates if p.is_file() and not is_excluded(p)]

        py_files = [p for p in files if p.suffix == ".py"]
        sizes = [(p, p.stat().st_size) for p in files]
        total_loc = 0
        todo_count = 0
        test_coverage_files = 0
        risk_hotspots: List[str] = []

        for p in py_files:
            try:
                with p.open("r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                total_loc += content.count("\n") + 1
                todo_count += len(re.findall(r"TODO|FIXME|XXX", content))
                if re.search(r"eval\(|exec\(|pickle\.loads\(", content):
                    risk_hotspots.append(str(p))
                if re.search(r"fastapi|Flask|Django", content, re.I):
                    if not re.search(r"auth|jwt|oauth", content, re.I):
                        risk_hotspots.append(f"{p}: framework without apparent auth")
                if "pytest" in content or "unittest" in content:
                    test_coverage_files += 1
            except Exception:
                continue

        largest = sorted(sizes, key=lambda t: t[1], reverse=True)[:10]
        avg_kb = (sum(sz for _, sz in sizes) / max(len(sizes), 1)) / 1024.0

        repo_stats = RepoStats(
            total_files=len(files),
            total_py_files=len(py_files),
            total_loc=total_loc,
            avg_file_size_kb=round(avg_kb, 2),
            largest_files=[(str(p.relative_to(self.root_path)), sz) for p, sz in largest],
            todo_count=todo_count,
            test_coverage_files=test_coverage_files,
            has_ci=bool(list(self.root_path.glob('.github/workflows/*.yml'))),
            has_docker=(self.root_path / 'docker').exists() or any("docker-compose" in f.name for f in files),
            has_k8s=(self.root_path / 'k8s').exists(),
            has_monitoring=(self.root_path / 'monitoring').exists(),
            has_docs=(self.root_path / 'docs').exists(),
            risk_hotspots=risk_hotspots,
        )
        self.files = files
        return repo_stats

    def summarize_modules(self, max_per_file_bytes: Optional[int] = None) -> List[Dict[str, Any]]:
        summaries: List[Dict[str, Any]] = []
        budget = max_per_file_bytes or self.cfg.max_file_bytes
        for p in [x for x in self.files if x.suffix == ".py"]:
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
                excerpt = text[:budget]
                sha = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:12]
                summaries.append({
                    "path": str(p.relative_to(self.root_path)),
                    "bytes": len(text.encode("utf-8", errors="ignore")),
                    "sha": sha,
                    "top_level_defs": re.findall(r"^def\s+([a-zA-Z0-9_]+)\(|^class\s+([a-zA-Z0-9_]+)\(", text, re.M),
                    "excerpt": excerpt,
                })
            except Exception:
                continue
        return summaries

    def gemini_enrich(self, stats: RepoStats, modules_sample: List[Dict[str, Any]]) -> Optional[str]:
        if not self.vertex_model:
            return None
        try:
            context = {
                "stats": asdict(stats),
                "sample": modules_sample[: min(12, len(modules_sample))],
            }
            prompt = (
                "You are reviewing an IFRS9 Risk Management codebase (Python, PySpark, Polars, Docker). "
                "Provide a concise, production-hardening review with prioritized actions for: "
                "security, reliability, observability, performance, and compliance. "
                "Focus on actionable steps within the existing structure. Use short bullets."
            )
            gc = GenerationConfig(temperature=0.2, max_output_tokens=1024)
            resp = self.vertex_model.generate_content([
                "SYSTEM: Senior SRE + Security engineer reviewer.",
                f"CONTEXT:\n{json.dumps(context)[:50_000]}",
                f"TASK:\n{prompt}",
            ], generation_config=gc)
            return resp.text
        except Exception as e:
            logger.warning(f"Gemini enrichment failed: {e}")
            return None

    def run(self) -> Dict[str, Any]:
        t0 = time.time()
        stats = self.scan()
        modules = self.summarize_modules()
        gemini_summary = self.gemini_enrich(stats, modules) if self.cfg.enable_gemini else None
        return {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "stats": asdict(stats),
            "modules_sample": modules[: min(20, len(modules))],
            "gemini_review": gemini_summary,
        }


def main():
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="IFRS9 Gemini Codebase Analyzer")
    ap.add_argument("--root", default=".")
    ap.add_argument("--enable-gemini", action="store_true")
    ap.add_argument("--project")
    ap.add_argument("--location", default="us-central1")
    ap.add_argument("--credentials")
    ap.add_argument("--out", default="reports/codebase_analysis_report.json")
    args = ap.parse_args()

    cfg = AnalyzerConfig(
        root=args.root,
        enable_gemini=bool(args.enable_gemini),
        gcp_project=args.project,
        gcp_location=args.location,
        credentials_path=args.credentials,
    )
    analyzer = CodebaseAnalyzer(cfg)
    result = analyzer.run()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote analysis report to {out_path}")


if __name__ == "__main__":
    main()

