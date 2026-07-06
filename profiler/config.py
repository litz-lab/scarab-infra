#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

INPUT_DIR = Path(__file__).resolve().parent / "input"
DEFAULT_INPUT_FILENAME = "collected_stats.csv"


class Config:
    def __init__(self, path: Path, csv_filename: Optional[str] = None) -> None:
        self.path = Path(path)
        data = json.loads(self.path.read_text())
        csv_name = csv_filename or data.get("input", DEFAULT_INPUT_FILENAME)
        self.csv_path: Path = INPUT_DIR / csv_name
        self.outputs: List[dict] = data.get("output", [])
