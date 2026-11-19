"""Command-line entry point.

Examples
--------
Run one experiment (downloads data on first use, then works offline)::

    portfolio-lab run configs/global_multi_asset.yaml

Run every config and force a fresh download::

    portfolio-lab run configs/*.yaml --refresh-data
"""

from __future__ import annotations

import argparse
import glob
import logging
import sys
from pathlib import Path

from .config import load_config
from .experiments import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="portfolio-lab", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="run the full research pipeline for one or more configs")
    p_run.add_argument("configs", nargs="+", help="YAML config path(s); globs allowed")
    p_run.add_argument("--data-dir", default="data", help="local market-data cache")
    p_run.add_argument("--results-dir", default="results", help="output directory")
    p_run.add_argument("--refresh-data", action="store_true", help="re-download market data")
    p_run.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("yfinance", "matplotlib", "PIL", "peewee"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    paths: list[str] = []
    for pattern in args.configs:  # expand globs ourselves (Windows shells don't)
        paths.extend(sorted(glob.glob(pattern)) or [pattern])

    for path in paths:
        cfg = load_config(Path(path))
        summary = run(cfg, args.data_dir, args.results_dir, args.refresh_data)
        print(f"{cfg.name}: {summary['period'][0]} -> {summary['period'][1]}, "
              f"{summary['n_rebalances']} rebalances, {summary['runtime_seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
