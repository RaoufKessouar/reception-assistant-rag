"""Compare le VLM candidat au précédent second passage, sans modifier le corpus."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    old = baseline.get("summary", {}).get("metrics", {})
    new = candidate.get("summary", {}).get("metrics", {})
    common = sorted(set(old) & set(new))
    metrics = {
        name: {
            "qwen2_5_vl": old[name].get("proposal"),
            "qwen3_vl": new[name].get("proposal"),
            "delta": round(
                float(new[name].get("proposal", 0.0))
                - float(old[name].get("proposal", 0.0)),
                4,
            ),
        }
        for name in common
    }
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_report": str(args.baseline),
        "candidate_report": str(args.candidate),
        "summary": {
            "metrics_compared": len(metrics),
            "improvements": sum(row["delta"] > 0 for row in metrics.values()),
            "unchanged": sum(row["delta"] == 0 for row in metrics.values()),
            "regressions": sum(row["delta"] < 0 for row in metrics.values()),
        },
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
