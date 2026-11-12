from __future__ import annotations

import argparse
from pprint import pprint

from _bootstrap import add_src_to_path

add_src_to_path()

from parking_search_prediction.config import load_config
from parking_search_prediction.indicators import run_indicators


def main() -> None:
    parser = argparse.ArgumentParser(description="Indicateurs de cruising par heure et par motif (étape 41).")
    parser.add_argument("--config", default="config/full_waypoints.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--methods", default=None, help="ex. relut,spatial,hybrid")
    parser.add_argument("--layout", choices=["panels", "overlay"], default=None)
    parser.add_argument("--purpose-column", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.output_dir:
        config["paths"]["output_dir"] = args.output_dir
    cfg = config.setdefault("indicators", {})
    if args.methods:
        cfg["methods"] = [m.strip() for m in args.methods.split(",") if m.strip()]
    if args.layout:
        cfg["layout"] = args.layout
    if args.purpose_column:
        cfg["purpose_column"] = args.purpose_column
    pprint(run_indicators(config)["summary"])


if __name__ == "__main__":
    main()
