"""Compare les contrôles automatiques Qwen2.5 et Qwen3 cas par cas."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def _latest_baseline(directory: Path) -> Path:
    matches = []
    for file in directory.glob("*.json"):
        try:
            payload = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if "Qwen2.5-14B" in str(payload.get("model", "")):
            matches.append(file)
    if not matches:
        raise FileNotFoundError("Aucun rapport Qwen2.5-14B trouvé")
    return max(matches, key=lambda file: file.stat().st_mtime)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline_file = _latest_baseline(args.baseline_dir)
    baseline = json.loads(baseline_file.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    old = {row["id"]: row for row in baseline.get("results", [])}
    new = {row["id"]: row for row in candidate.get("results", [])}
    common = sorted(set(old) & set(new))
    rows = [
        {
            "id": key,
            "qwen2_5_pass": bool(old[key].get("automatic_pass")),
            "qwen3_pass": bool(new[key].get("automatic_pass")),
            "changed": bool(old[key].get("automatic_pass"))
            != bool(new[key].get("automatic_pass")),
            "qwen2_5_failures": old[key].get("failures", []),
            "qwen3_failures": new[key].get("failures", []),
        }
        for key in common
    ]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_report": str(baseline_file),
        "candidate_report": str(args.candidate),
        "baseline_model": baseline.get("model"),
        "candidate_model": candidate.get("model"),
        "summary": {
            "compared": len(rows),
            "qwen2_5_passed": sum(row["qwen2_5_pass"] for row in rows),
            "qwen3_passed": sum(row["qwen3_pass"] for row in rows),
            "improvements": sum(
                not row["qwen2_5_pass"] and row["qwen3_pass"] for row in rows
            ),
            "regressions": sum(
                row["qwen2_5_pass"] and not row["qwen3_pass"] for row in rows
            ),
        },
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
