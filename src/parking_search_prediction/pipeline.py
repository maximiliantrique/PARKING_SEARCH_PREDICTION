from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ensure_output_dirs, output_path
from .data_prep import (
    attach_waypoints_to_legs,
    clean_waypoints,
    filter_car_waypoints,
    read_raw_waypoints,
    read_transport_legs,
    segment_waypoints,
    to_relut_points,
    write_prepared_outputs,
)
from .spatial_study_area import apply_study_area_filter
from .loop_segments import apply_loop_filter
from .utils import write_json
from .validation import validate_relut_segments

# Les modules des étapes suivantes (21 à 44 : détection spatiale, modèle
# temporel, résultats…) sont importés à l'intérieur de predict(), review() et
# run_all(). Ce module s'importe ainsi avec les seuls fichiers de l'étape 01.

ASSIGNMENT_COLUMNS = [
    "waypoint_id",
    "source_user_id",
    "source_file",
    "source_row_number",
    "timestamp",
    "leg_id",
    "storyline_id",
    "trip_id",
    "leg_started_at",
    "leg_finished_at",
    "mode",
    "detected_mode",
    "effective_mode",
    "mode_niv1",
    "mode_niv2",
    "mode_mrmt",
    "leg_match_status",
]


_LAST_PROGRESS = [time.perf_counter()]


def _fmt_int(n: int) -> str:
    return f"{n:,}".replace(",", "'")


def _progress(
    step: str, df: pd.DataFrame | None = None, *, prefix: str = "[prepare]", **extra: Any
) -> None:
    """Affiche l'avancement : heure, durée depuis l'affichage précédent (donc durée
    de l'étape qui vient de finir), waypoints restants, legs, segments."""
    parts = []
    if df is not None:
        parts.append(f"{_fmt_int(len(df))} waypoints")
        for col, label in [("leg_id", "legs"), ("trip_candidate_id", "segments")]:
            if col in df.columns and len(df):
                parts.append(f"{_fmt_int(df[col].nunique())} {label}")
    for key, value in extra.items():
        shown = _fmt_int(value) if isinstance(value, int) and not isinstance(value, bool) else value
        parts.append(f"{key}={shown}")
    now = time.perf_counter()
    elapsed = now - _LAST_PROGRESS[0]
    _LAST_PROGRESS[0] = now
    stamp = f"{datetime.now():%H:%M:%S} (+{elapsed:6.1f} s)"
    print(f"{prefix} {stamp} {step:<30} " + " | ".join(parts), flush=True)


def _remove_path(path: Path) -> None:
    """Supprime une sortie existante, fichier (mode CSV) ou dossier (mode partitionné)."""
    import shutil

    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _match_and_filter(
    raw: pd.DataFrame,
    tracks: pd.DataFrame,
    config: dict[str, Any],
    *,
    prefix: str = "[prepare]",
) -> dict[str, Any]:
    """Étapes 2 à 6 : nettoyage, appariement aux legs, filtre voiture, périmètre.

    Partagé par le mode CSV et le mode partitionné pour garantir un traitement
    identique.
    """
    clean, cleaning_report = clean_waypoints(raw, config)
    _progress("2. nettoyage GPS", clean, prefix=prefix, retirés=len(raw) - len(clean))
    matched, matching_report = attach_waypoints_to_legs(clean, tracks)
    n_clean = int(len(clean))
    del clean
    _progress(
        "4. appariement waypoint-leg",
        prefix=prefix,
        matched_waypoints=matching_report.get("matched_waypoints"),
        matched_legs=matching_report.get("matched_legs"),
    )
    car_waypoints, car_filter_report = filter_car_waypoints(matched, config)
    _progress("5. filtre voiture", car_waypoints, prefix=prefix)
    car_legs_with_waypoints = set(car_waypoints["leg_id"].dropna()) if "leg_id" in car_waypoints else set()

    # TODO-METH-02 : filtre géographique explicite (destination de la leg
    # voiture — son dernier waypoint apparié — contenue dans le layer de
    # périmètre). Le flag est toujours calculé ; `car_waypoints` n'est
    # réellement restreint que si spatial_detection.apply_study_area_filter
    # est vrai dans la config.
    car_waypoints, _, study_area_report = apply_study_area_filter(tracks, car_waypoints, config)
    _progress(
        "6. filtre périmètre",
        car_waypoints,
        prefix=prefix,
        appliqué=study_area_report["study_area_filter_applied"],
        legs_hors_périmètre=study_area_report.get("study_area_car_legs_out_of_area"),
    )
    return {
        "n_clean": n_clean,
        "cleaning_report": cleaning_report,
        "matched": matched,
        "matching_report": matching_report,
        "car_waypoints": car_waypoints,
        "car_filter_report": car_filter_report,
        "study_area_report": study_area_report,
        "car_legs_with_waypoints": car_legs_with_waypoints,
        "leg_stats": _corpus_stats(car_waypoints, config),
    }


def _corpus_stats(car_waypoints: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Statistiques par leg du corpus commun (leg_corpus.py), sur les waypoints voiture du périmètre."""
    from .leg_corpus import corpus_settings, leg_stats

    return leg_stats(car_waypoints, corpus_settings(config), config["raw_waypoints"].get("accuracy_col", "accuracy"))


def finalize_corpus(config: dict[str, Any], tracks: pd.DataFrame, stats: pd.DataFrame,
                    car_legs_with_waypoints: set[str], *, prefix: str = "[prepare]") -> dict[str, Any]:
    """Construit et écrit legs/corpus_by_leg.parquet : la base commune des étapes 22 et 32."""
    from .leg_corpus import build_corpus, corpus_summary, write_corpus

    trips = tracks.drop_duplicates("leg_id").set_index("leg_id")["trip_duration_seconds"] if "trip_duration_seconds" in tracks else None
    corpus = build_corpus(config, tracks[["leg_id", "effective_mode"]], stats, car_legs_with_waypoints, trips)
    write_corpus(config, corpus)
    summary = corpus_summary(corpus)
    write_json(output_path(config, "reports", "leg_corpus_summary.json"), summary)
    _progress("11. corpus commun", prefix=prefix, legs_voiture_perimetre=summary["car_legs_in_area"],
              corpus=summary["in_corpus"])
    return summary


def _segment_and_validate(
    car_waypoints: pd.DataFrame,
    config: dict[str, Any],
    *,
    prefix: str = "[prepare]",
    allow_empty: bool = False,
) -> dict[str, Any]:
    """Étapes 7 à 9 : segmentation, filtre boucles, format ReLUT, validation."""
    segmented, segment_candidates = segment_waypoints(car_waypoints, config, allow_empty=allow_empty)
    _progress(
        "7. segmentation",
        segmented,
        prefix=prefix,
        candidats=len(segment_candidates),
        rejetés=int((~segment_candidates["kept"].astype(bool)).sum()),
    )

    # Filtre optionnel des traces en boucle (premier/dernier waypoint proches
    # de moins de segmentation.loop_radius_m) : ces traces performent mal dans
    # le modèle temporel. Le flag est toujours calculé sur segment_candidates ;
    # les points ne sont retirés de `segmented` que si
    # segmentation.apply_loop_filter est vrai.
    segmented, segment_candidates, loop_filter_report = apply_loop_filter(
        segmented, segment_candidates, config
    )
    _progress(
        "8. filtre boucles",
        segmented,
        prefix=prefix,
        appliqué=loop_filter_report["loop_filter_applied"],
        boucles_détectées=loop_filter_report["loop_segments_detected"],
    )

    relut_points = to_relut_points(segmented)
    del segmented
    validation, validation_summary = validate_relut_segments(relut_points, config)
    _progress(
        "9. format ReLUT + validation",
        relut_points,
        prefix=prefix,
        prêts_relut=validation_summary["segments_relut_ready"],
    )
    return {
        "relut_points": relut_points,
        "segment_candidates": segment_candidates,
        "loop_filter_report": loop_filter_report,
        "validation": validation,
        "validation_summary": validation_summary,
    }


def load_enriched_legs(
    config: dict[str, Any], *, prefix: str = "[prepare]"
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], pd.DataFrame]:
    """Lit les legs, les enrichit (leg_enrichment.py) et écrit ``legs_enriched.parquet``.

    Retourne ``(legs, rapport de lecture, rapport d'enrichissement, correspondance
    marche → voiture)``. Partagé par le mode CSV et le mode partitionné.
    """
    from .leg_enrichment import enrich_legs, enrichment_settings, post_car_walk_lookup

    tracks, legs_report = read_transport_legs(config)
    _progress("3. lecture legs", prefix=prefix, legs_lues=len(tracks))
    tracks, enrichment_report = enrich_legs(tracks, config)
    legs_path = output_path(config, "data", "legs_enriched.parquet")
    _remove_path(legs_path)
    lookup = pd.DataFrame(columns=["leg_id", "car_leg_id"])
    if enrichment_report.get("leg_enrichment_enabled"):
        tracks.to_parquet(legs_path, index=False)
        if enrichment_settings(config)["write_walk_waypoints"]:
            lookup = post_car_walk_lookup(tracks)
        _progress(
            "3b. enrichissement legs",
            prefix=prefix,
            legs=len(tracks),
            sans_trip=enrichment_report["legs_without_trip"],
            marches_après_voiture=enrichment_report["post_car_walk_matched"],
        )
    write_json(output_path(config, "reports", "leg_enrichment_summary.json"), enrichment_report)
    return tracks, legs_report, enrichment_report, lookup


def _mode_counts_frame(car_filter_report: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"effective_mode": mode, "waypoint_count": count}
            for mode, count in car_filter_report["matched_waypoints_by_mode"].items()
        ],
        columns=["effective_mode", "waypoint_count"],
    )


def _prepare_summary(
    *,
    raw_rows: int,
    clean_rows: int,
    matching_report: dict[str, Any],
    car_filter_report: dict[str, Any],
    study_area_report: dict[str, Any],
    loop_filter_report: dict[str, Any],
    validation_summary: dict[str, Any],
    prepared_points: int,
    prepared_segments: int,
) -> dict[str, Any]:
    return {
        "raw_rows": int(raw_rows),
        "clean_rows": int(clean_rows),
        "matched_waypoints": matching_report["matched_waypoints"],
        "matched_legs": matching_report["matched_legs"],
        "car_waypoints": car_filter_report["car_waypoints"],
        "car_legs": car_filter_report["car_legs"],
        "study_area_filter_configured": study_area_report["study_area_filter_configured"],
        "study_area_total_car_legs": study_area_report["study_area_total_car_legs"],
        "study_area_car_legs_in_area": study_area_report["study_area_car_legs_in_area"],
        "study_area_car_legs_out_of_area": study_area_report["study_area_car_legs_out_of_area"],
        "study_area_car_legs_after_filter": study_area_report["study_area_car_legs_after_filter"],
        "study_area_filter_applied": study_area_report["study_area_filter_applied"],
        "loop_filter_applied": loop_filter_report["loop_filter_applied"],
        "loop_filter_radius_m": loop_filter_report["loop_filter_radius_m"],
        "loop_segments_detected": loop_filter_report["loop_segments_detected"],
        "loop_segments_after_filter": loop_filter_report["loop_segments_after_filter"],
        "prepared_points": int(prepared_points),
        "prepared_segments": int(prepared_segments),
        "relut_ready_segments": validation_summary["segments_relut_ready"],
    }


def prepare(config: dict[str, Any]) -> dict[str, Any]:
    """Étape 01. Choisit la source selon ``raw_waypoints.source`` :

    - ``"csv"`` (défaut) : lecture directe des CSV, tout en mémoire ;
    - ``"parquet"`` : dataset partitionné par utilisateur, traité par tranches
      de ``run.groups_per_run`` partitions (voir ``partitioned.py``).
    """
    source = config["raw_waypoints"].get("source", "csv")
    if source == "parquet":
        from .partitioned import prepare_partitioned

        return prepare_partitioned(config)
    if source != "csv":
        raise ValueError("raw_waypoints.source doit valoir 'csv' ou 'parquet'.")
    return prepare_csv(config)


def prepare_csv(config: dict[str, Any]) -> dict[str, Any]:
    ensure_output_dirs(config)
    _progress("démarrage (source CSV)")
    raw = read_raw_waypoints(config)
    raw_rows = int(len(raw))
    _progress("1. lecture waypoints bruts", raw)
    tracks, legs_report, enrichment_report, walk_lookup = load_enriched_legs(config)

    stage = _match_and_filter(raw, tracks, config)
    del raw
    matched = stage["matched"]
    corpus_stats, corpus_car_legs = stage["leg_stats"], stage["car_legs_with_waypoints"]
    walk_path = output_path(config, "data", "waypoints_post_car_walk.parquet")
    _remove_path(walk_path)
    walk_waypoints = 0
    if not walk_lookup.empty:
        from .leg_enrichment import post_car_walk_waypoints

        walk = post_car_walk_waypoints(matched, walk_lookup, config)
        walk.to_parquet(walk_path, index=False)
        walk_waypoints = int(len(walk))
        _progress("4b. waypoints marche après voiture", walk)
    car_waypoints = stage["car_waypoints"]
    matching_report = stage["matching_report"]
    car_filter_report = stage["car_filter_report"]
    study_area_report = stage["study_area_report"]
    cleaning_report = stage["cleaning_report"]
    clean_rows = stage["n_clean"]

    write_json(
        output_path(config, "reports", "spatial_study_area_filter_summary.json"),
        study_area_report,
    )
    leg_matching_report = {
        **legs_report,
        **matching_report,
        **car_filter_report,
        **study_area_report,
    }

    assignment_path = output_path(config, "data", "waypoint_leg_assignments.parquet")
    caronly_path = output_path(config, "data", "waypoints_caronly.parquet")
    if config["leg_matching"].get("write_assignment_table", True):
        available_assignment_columns = [
            column for column in ASSIGNMENT_COLUMNS if column in matched.columns
        ]
        _remove_path(assignment_path)
        matched[available_assignment_columns].to_parquet(assignment_path, index=False)
    del matched, stage
    _remove_path(caronly_path)
    car_waypoints.to_parquet(caronly_path, index=False)
    write_json(
        output_path(config, "reports", "waypoint_leg_matching_summary.json"),
        leg_matching_report,
    )
    _mode_counts_frame(car_filter_report).to_csv(
        output_path(config, "reports", "waypoint_leg_matching_by_mode.csv"), index=False
    )

    if car_waypoints.empty:
        raise ValueError(
            "No car waypoints remain after storyline matching (and, if "
            "spatial_detection.apply_study_area_filter is true, after the "
            "geographic study-area filter). See "
            f"{output_path(config, 'reports', 'waypoint_leg_matching_summary.json')} "
            "and verify that waypoint and storyline exports cover the same users and period."
        )

    seg = _segment_and_validate(car_waypoints, config)
    relut_points = seg["relut_points"]
    validation_summary = seg["validation_summary"]
    write_json(
        output_path(config, "reports", "loop_filter_summary.json"), seg["loop_filter_report"]
    )
    write_prepared_outputs(relut_points, seg["segment_candidates"], cleaning_report, config)
    seg["validation"].to_csv(
        output_path(config, "reports", "relut_validation_by_segment.csv"), index=False
    )
    write_json(output_path(config, "reports", "relut_validation_summary.json"), validation_summary)
    _progress("10. sorties écrites", dossier=str(output_path(config, "data")))
    corpus_summary = finalize_corpus(config, tracks, corpus_stats, corpus_car_legs)

    return {
        **_prepare_summary(
            raw_rows=raw_rows,
            clean_rows=clean_rows,
            matching_report=matching_report,
            car_filter_report=car_filter_report,
            study_area_report=study_area_report,
            loop_filter_report=seg["loop_filter_report"],
            validation_summary=validation_summary,
            prepared_points=len(relut_points),
            prepared_segments=relut_points["trip_candidate_id"].nunique(),
        ),
        **_enrichment_summary(enrichment_report, walk_waypoints),
        "corpus_legs": corpus_summary["in_corpus"],
    }


def _enrichment_summary(enrichment_report: dict[str, Any], walk_waypoints: int) -> dict[str, Any]:
    return {
        "leg_enrichment_enabled": enrichment_report.get("leg_enrichment_enabled", False),
        "legs_without_trip": enrichment_report.get("legs_without_trip"),
        "post_car_walk_legs": enrichment_report.get("post_car_walk_matched"),
        "post_car_walk_waypoints": int(walk_waypoints),
    }


def predict(config: dict[str, Any]) -> dict[str, Any]:
    """Étape 32 : prédictions ReLUT, synthèses, histogrammes et cartes (voir relut_detection.py)."""
    from .relut_detection import run_temporal_detection

    return run_temporal_detection(config)["summary"]


def review(config: dict[str, Any]) -> dict[str, Any]:
    """Relit la synthèse globale de l'étape 32 (les sorties sont produites par predict())."""
    import json

    path = output_path(config, "reports", "prediction_global_summary.json")
    if not path.exists():
        raise FileNotFoundError(f"{path} introuvable : lancer predict(config) d'abord.")
    return json.loads(path.read_text(encoding="utf-8"))


def run_all(
    config: dict[str, Any],
    *,
    include_pl23: bool = False,
    include_routing_preparation: bool = False,
    include_routing: bool = False,
    include_network_relut: bool = False,
    include_territorial_indicators: bool = False,
) -> dict[str, Any]:
    """Run the complete documented workflow on the configured data scope.

    The sequence mirrors the notebooks:
    01 prepare inputs, 21 spatial EDA, 22 spatial detection on Déclic, optional
    23 spatial detection on PL23, optional 25 routing candidate preparation,
    31 temporal EDA, 32 temporal detection/review, 41 per-approach results and
    42 final spatial/temporal comparison on Déclic.

    PL23 is opt-in because the source GPKG is several GB and the step can be
    materially slower than the default Déclic sample run.
    """

    from .network_relut import run_routed_network_relut
    from .pl23_spatial import run_pl23_spatial_detection
    from .results import run_results_by_approach
    from .routing import (
        apply_dist_excess_to_spatial_candidates,
        prepare_routing_candidates,
        route_candidates_with_xyt,
    )
    from .spatial_detection import run_spatial_detection, run_spatial_temporal_comparison
    from .spatial_eda import run_spatial_eda
    from .temporal_eda import run_temporal_eda
    from .territorial_indicators import run_territorial_indicators

    result = {
        "01_prepare_inputs": prepare(config),
        "21_spatial_eda": run_spatial_eda(config),
        "22_spatial_detection": run_spatial_detection(config),
    }
    if include_pl23:
        result["23_spatial_detection_pl23"] = run_pl23_spatial_detection(config)
    if include_routing:
        result["25_prepare_routing_candidates"] = prepare_routing_candidates(config)
        result["26_route_dist_excess_xyt"] = route_candidates_with_xyt(config)
        result["27_apply_routing_dist_excess"] = apply_dist_excess_to_spatial_candidates(config)
    elif include_routing_preparation:
        result["25_prepare_routing_candidates"] = prepare_routing_candidates(config)
    result.update(
        {
            "31_temporal_eda": run_temporal_eda(config),
            "32_temporal_detection": {
                "predict": predict(config),
                "review": review(config),
            },
            "41_results_by_approach": run_results_by_approach(config),
            "42_spatial_temporal_comparison": run_spatial_temporal_comparison(config),
        }
    )
    if include_network_relut:
        result["43_routed_network_relut"] = run_routed_network_relut(config)
    if include_territorial_indicators:
        result["44_territorial_indicators"] = run_territorial_indicators(config)
    write_json(output_path(config, "reports", "pipeline_run_summary.json"), result)
    return result