"""Command-line entry point."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="NYISO electricity-market forecasting laboratory")
    parser.add_argument("command", choices=["download", "benchmark", "figures", "forecast", "all"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--date", help="Forecast delivery date YYYY-MM-DD; needs 7 prior curves")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.command in ["download", "all"]:
        from .data import fetch, prepare

        prepare(fetch("2022-01-01", "2025-12-01", args.root), args.root)
    if args.command in ["benchmark", "all"]:
        from .benchmark import run

        run(args.root, args.epochs, args.seed)
    if args.command in ["figures", "all"]:
        from .plots import generate

        generate(args.root)
    if args.command == "forecast":
        from .inference import forecast

        forecast(args.root, args.date)


if __name__ == "__main__":
    main()
