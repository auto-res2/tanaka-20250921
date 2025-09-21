import argparse
import sys
from pathlib import Path

import yaml

from .train import launch_training


def load_yaml(path: Path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Run LUP experiments")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick validation")
    parser.add_argument("--full-experiment", action="store_true", help="Run full experiment")
    args = parser.parse_args()

    if not (args.smoke_test ^ args.full_experiment):
        print("ERROR: Must pass exactly one of --smoke-test or --full-experiment", file=sys.stderr)
        sys.exit(1)

    cfg_path = Path("config") / ("smoke_test.yaml" if args.smoke_test else "full_experiment.yaml")
    cfg = load_yaml(cfg_path)
    launch_training(cfg, smoke=args.smoke_test)


if __name__ == "__main__":
    main()
