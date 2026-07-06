#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

from config import Config
from loader import Loader
from tabulator import Tabulator
from plotter import Plotter

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "json" / "config.json"


def main() -> None:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG_PATH
    config = Config(config_path)

    loader = Loader(config.csv_path)
    tabulator = Tabulator(loader)
    plotter = Plotter(tabulator)
    plotter.plot(config.outputs)


if __name__ == "__main__":
    main()
