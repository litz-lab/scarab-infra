#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import List

INPUT_DIR = Path(__file__).resolve().parent / "input"
DEFAULT_INPUT_FILENAME = "collected_stats.csv"


class Config:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        data = json.loads(self.path.read_text())
        self.csv_path: Path = INPUT_DIR / data.get("input", DEFAULT_INPUT_FILENAME)
        self.outputs: List[dict] = data.get("output", [])
