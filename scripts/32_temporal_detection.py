from __future__ import annotations

import argparse
from pprint import pprint

from _bootstrap import add_src_to_path

add_src_to_path()

from parking_search_prediction.config import load_config
from parking_search_prediction.relut_detection import run_temporal_detection


def main() -> None:
    parser = argparse.ArgumentParser(description="Détection temporelle ReLUT sur les segments voiture préparés (étape 32).")
    parser.add_argument("--config", default="config/full_waypoints.yaml")
    parser.add_argument("--output-dir", default=None, help="dossier de sortie de l'étape 01 (défaut : celui du YAML)")
    parser.add_argument("--max-legs", type=int, default=None, help="nombre de legs tirées au hasard ; absent = toutes")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--parking-variable", choices=["space_home", "space_work"], default=None)
    parser.add_argument("--no-survey", action="store_true", help="sans questionnaire (Déclic)")
    parser.add_argument("--n-generic-sample", type=int, default=None, help="segments des courbes de probabilité / carte generic sample")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.output_dir:
        config["paths"]["output_dir"] = args.output_dir
    detection = config.setdefault("temporal_detection", {})
    detection["max_legs"] = args.max_legs
    if args.seed is not None:
        detection["random_seed"] = args.seed
    if args.parking_variable:
        detection["parking_variable"] = args.parking_variable
    if args.no_survey:
        detection["use_survey"] = False
    if args.n_generic_sample is not None:
        detection["n_generic_sample"] = args.n_generic_sample
    pprint(run_temporal_detection(config)["summary"])


if __name__ == "__main__":
    main()
