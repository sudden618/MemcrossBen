#!/usr/bin/env python3
"""Run a YAML-defined main-experiment pipeline with stage-level resume markers."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--state-dir", default="outputs/pipeline_state")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    state = (root / args.state_dir).resolve()
    state.mkdir(parents=True, exist_ok=True)
    for stage in config["stages"]:
        marker = state / f'{stage["name"]}.done.json'
        if marker.exists() and not args.force:
            print(f'[resume] skip {stage["name"]}')
            continue
        command = [os.path.expandvars(str(value)) for value in stage["command"]]
        print(f'[run] {stage["name"]}: {" ".join(command)}', flush=True)
        started = datetime.now(timezone.utc).isoformat()
        subprocess.run(command, cwd=root, check=True)
        marker.write_text(json.dumps({
            "stage": stage["name"],
            "command": command,
            "started_at": started,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "config": str(config_path),
        }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
