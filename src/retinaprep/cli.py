"""Single entry point: python -m retinaprep <command>."""

from __future__ import annotations

import argparse
import sys

COMMANDS = ("ingest", "quality", "dedupe", "split", "train", "experiment", "report")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="retinaprep", description=__doc__)
    p.add_argument("command", choices=COMMANDS)
    p.add_argument("--config", default=None, help="Path to YAML config")
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Dotted-key config override, e.g. --set train.epochs=2",
    )
    p.add_argument(
        "--split", default=None, help="Split name for train (image_random|patient_group)"
    )
    p.add_argument("--arm", default=None, help="Experiment arm: A, B, C or D")
    return p


def _parse_overrides(items: list[str]) -> dict:
    out = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"Bad override {item!r}, expected KEY=VALUE")
        k, v = item.split("=", 1)
        out[k] = yaml_scalar(v)
    return out


def yaml_scalar(v: str):
    """Coerce a CLI string into int/float/bool/None where obvious."""
    import yaml

    return yaml.safe_load(v)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from retinaprep.config import load_config

    cfg = load_config(args.config, _parse_overrides(args.overrides))

    if args.command == "ingest":
        from retinaprep.ingest import run_ingest

        run_ingest(cfg)
    elif args.command == "quality":
        from retinaprep.quality import run_quality

        run_quality(cfg)
    elif args.command == "dedupe":
        from retinaprep.dedupe import run_dedupe

        run_dedupe(cfg)
    elif args.command == "split":
        from retinaprep.splits import run_split

        run_split(cfg)
    elif args.command == "train":
        from retinaprep.train import run_train

        run_train(cfg, split_name=args.split)
    elif args.command == "experiment":
        from retinaprep.experiment import run_experiment

        run_experiment(cfg, arm=args.arm)
    elif args.command == "report":
        from retinaprep.report import run_report

        run_report(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
