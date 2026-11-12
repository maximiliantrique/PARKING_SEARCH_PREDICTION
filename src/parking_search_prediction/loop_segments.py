"""Filtre optionnel des traces GPS en boucle (segment_waypoints, 01_prepare_inputs).

Certaines traces voiture reviennent près de leur point de départ (rond-point,
tour de pâté de maisons, demi-tour, chargement/déchargement...). Ces "boucles"
performent mal dans le modèle temporel de recherche de stationnement.

Même logique que le filtre de périmètre géographique (spatial_study_area.py) :
un flag `is_loop_segment` est toujours calculé sur `segment_candidates`
(une ligne par trip_candidate_id) ; les points correspondants ne sont
réellement retirés de `segmented` que si `segmentation.apply_loop_filter`
est vrai dans la config. Les deux dénominateurs (avant/après) restent
disponibles dans le résumé retourné.

Règle : une boucle est un segment dont le premier et le dernier waypoint
(ordonnés par timestamp) sont à une distance <= `segmentation.loop_radius_m`
l'un de l'autre.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .utils import haversine_m

DEFAULT_LOOP_RADIUS_M = 150.0


def _loop_settings(config: dict[str, Any]) -> dict[str, Any]:
    seg_cfg = config.get("segmentation", {}) or {}
    return {
        "loop_radius_m": float(seg_cfg.get("loop_radius_m", DEFAULT_LOOP_RADIUS_M)),
        "apply_filter": bool(seg_cfg.get("apply_loop_filter", False)),
    }


def compute_loop_flags(segmented: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Une ligne par trip_candidate_id : distance début-fin (m) + flag boucle.

    Ne porte que sur les segments présents dans `segmented`, c'est-à-dire ceux
    déjà retenus par `segment_waypoints` (kept=True) — les candidats déjà
    rejetés par les filtres de segmentation n'ont pas de waypoints à comparer.
    """
    required = {"trip_candidate_id", "lat", "lon", "timestamp"}
    missing = sorted(required - set(segmented.columns))
    if missing:
        raise KeyError(
            f"Colonnes manquantes dans segmented pour détecter les boucles: {missing}"
        )

    settings = _loop_settings(config)
    if segmented.empty:
        return pd.DataFrame(
            columns=["trip_candidate_id", "loop_start_end_distance_m", "is_loop_segment"]
        )

    ordered = segmented.sort_values(["trip_candidate_id", "timestamp"])
    grouped = ordered.groupby("trip_candidate_id", sort=False)
    first = grouped[["lat", "lon"]].first().rename(columns={"lat": "start_lat", "lon": "start_lon"})
    last = grouped[["lat", "lon"]].last().rename(columns={"lat": "end_lat", "lon": "end_lon"})
    endpoints = first.join(last).reset_index()

    endpoints["loop_start_end_distance_m"] = haversine_m(
        endpoints["start_lat"], endpoints["start_lon"], endpoints["end_lat"], endpoints["end_lon"]
    )
    endpoints["is_loop_segment"] = endpoints["loop_start_end_distance_m"] <= settings["loop_radius_m"]
    return endpoints[["trip_candidate_id", "loop_start_end_distance_m", "is_loop_segment"]]


def flag_loop_segments(
    segmented: pd.DataFrame, segment_candidates: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Ajoute `loop_start_end_distance_m` / `is_loop_segment` à `segment_candidates`.

    Le résultat est destiné à être réinjecté dans le `segment_candidates.csv`
    déjà écrit par `write_prepared_outputs`, pour un audit trace par trace.
    """
    settings = _loop_settings(config)
    loop_flags = compute_loop_flags(segmented, config)

    segment_candidates = segment_candidates.merge(loop_flags, on="trip_candidate_id", how="left")
    segment_candidates["is_loop_segment"] = segment_candidates["is_loop_segment"].astype("boolean")

    kept_segments = int(len(loop_flags))
    loop_segments_detected = int((segment_candidates["is_loop_segment"] == True).sum())  # noqa: E712

    summary = {
        "loop_filter_radius_m": settings["loop_radius_m"],
        "loop_filter_applied": settings["apply_filter"],
        "loop_total_kept_segments": kept_segments,
        "loop_segments_detected": loop_segments_detected,
        "loop_segments_after_filter": None,  # rempli par apply_loop_filter
    }
    return segment_candidates, summary


def apply_loop_filter(
    segmented: pd.DataFrame, segment_candidates: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Calcule le flag boucle par segment puis, si `segmentation.apply_loop_filter`
    est vrai, retire de `segmented` les trip_candidate_id détectés comme boucle.

    Retourne `(segmented_filtré, segment_candidates_flaggé, résumé)` avec
    toujours les deux dénominateurs (avant/après filtre boucle).
    """
    flagged_candidates, summary = flag_loop_segments(segmented, segment_candidates, config)
    settings = _loop_settings(config)

    if settings["apply_filter"]:
        loop_ids = set(
            flagged_candidates.loc[
                flagged_candidates["is_loop_segment"] == True, "trip_candidate_id"  # noqa: E712
            ]
        )
        filtered_segmented = segmented[~segmented["trip_candidate_id"].isin(loop_ids)].copy()
    else:
        filtered_segmented = segmented

    summary["loop_segments_after_filter"] = (
        int(filtered_segmented["trip_candidate_id"].nunique()) if len(filtered_segmented) else 0
    )
    summary["loop_points_before_filter"] = int(len(segmented))
    summary["loop_points_after_filter"] = int(len(filtered_segmented))
    return filtered_segmented, flagged_candidates, summary
