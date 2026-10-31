"""Command-line entry point for data retrieval and parser execution."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from ep_data.config import load_config
from ep_data.exceptions import HEPDataRetrievalError
from ep_data.pipeline import process_dataset

LOGGER = logging.getLogger("ep_data")
DEFAULT_CONFIG = Path.cwd() / "data" / "hep_config.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download HEPData YAML submissions and run configured parsers."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        metavar="KEY",
        help="process one dataset (repeat to select multiple datasets)",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--download-only", action="store_true", help="download without parsing"
    )
    modes.add_argument(
        "--parse-only", action="store_true", help="parse existing raw data offline"
    )
    parser.add_argument(
        "--force", action="store_true", help="redownload and rerun configured parsers"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_datasets",
        help="list configured datasets",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"dataset registry (default: {DEFAULT_CONFIG})",
    )
    return parser


def _unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    try:
        datasets = load_config(args.config)
    except HEPDataRetrievalError as exc:
        LOGGER.error("%s", exc)
        return 1

    if args.list_datasets:
        for key, dataset in datasets.items():
            parser_name = dataset.parser if dataset.parser is not None else "none"
            print(f"{key}\t{dataset.doi}\tparser={parser_name}")
        return 0

    selected = _unique(args.datasets) if args.datasets else list(datasets)
    data_dir = args.config.resolve().parent
    failures: list[tuple[str, str]] = []

    for key in selected:
        LOGGER.info("Processing %s", key)
        try:
            process_dataset(
                key,
                args.config,
                data_dir,
                force=args.force,
                download_only=args.download_only,
                parse_only=args.parse_only,
            )
        except Exception as exc:  # isolate failures between selected datasets
            failures.append((key, str(exc)))
            LOGGER.error("Failed %s: %s", key, exc)

    if failures:
        LOGGER.error("%d dataset(s) failed", len(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
