from __future__ import annotations

import argparse
from pprint import pprint

from _bootstrap import add_src_to_path

add_src_to_path()

from parking_search_prediction.config import load_config
from parking_search_prediction.pl23_eda import run_pl23_eda


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EDA PL23 : distributions des waypoints voiture et stationnement déclaré (domicile / travail)."
    )
    parser.add_argument("--config", default="config/full_waypoints.yaml")
    parser.add_argument("--output-dir", default=None, help="dossier de sortie de l'étape 01 (défaut : celui du YAML)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.output_dir:
        config["paths"]["output_dir"] = args.output_dir
    pprint(run_pl23_eda(config))


if __name__ == "__main__":
    main()
