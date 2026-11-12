"""Enrichissement des legs (étape 01) : trip, motif, moment, questionnaire, marche après la voiture.

Reprend les enrichissements de l'étape 23 (``pl23_spatial.py``) au niveau de
**toutes** les legs, pour que les étapes temporelle et spatiale partagent la
même table :

1. ``map_track_trip_journey.csv`` : ``trip_id`` et ``journey_id`` de chaque leg ;
   les legs sans trip sont retirées si ``drop_legs_without_trip`` ;
2. ``trips.csv`` : motif, durée, distance, modes, identifiants utilisateur et
   ``leading_activity_id`` (activité qui suit le trip, donc sa destination) ;
3. ``purpose_agg`` (table ``PURPOSE_AGG`` de l'étape 23, ``Autres`` à défaut),
   ``time_of_day_end_trip`` (bins historiques, heure de fin de la leg) et
   ``weekday`` ;
4. questionnaire PL23 (``space_home``, ``space_work``, zones, ``KLASSE_ARE_dom``,
   âge, genre, ``Weight``), joint par ``user_id_fors`` = ``IDNO`` avec la même
   dérivation que l'étape 23 (``pl23_survey.py``) ;
5. **marche après la voiture** : pour chaque leg voiture, la leg suivante du
   même trip est retenue si c'est une marche qui commence au plus
   ``max_gap_minutes`` après la fin de la voiture et fait au plus
   ``max_walk_distance_m``. Le motif de rejet est conservé.

Différence assumée avec l'étape 23 : tranche horaire et jour sont calculés à
l'heure locale (``local_timezone``, Europe/Zurich par défaut) ; l'étape 23
utilisait l'heure du fichier (UTC). Mettre ``local_timezone: UTC`` pour
reproduire l'étape 23.

Sortie : ``data/legs_enriched.parquet`` (une ligne par leg, clé ``leg_id``) et
``reports/leg_enrichment_summary.json``. Les waypoints des marches retenues
sont extraits par ``post_car_walk_waypoints`` pendant l'appariement.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import resolve_path

# Table de l'étape 23 (pl23_spatial.PURPOSE_AGG)
PURPOSE_AGG = {
    "assistance": "Accompagnement",
    "shopping": "Achats et services",
    "errand": "Achats et services",
    "medical_visit": "Achats et services",
    "study": "Formation",
    "work": "Travail",
    "sport": "Loisirs",
    "leisure": "Loisirs",
    "eat": "Loisirs",
    "family_friends": "Loisirs",
    "home": "Retour à la maison",
}

# Bins de l'étape 23 (pl23_spatial.TIME_OF_DAY_BINS)
TIME_OF_DAY_BINS = {
    "morning peak (6-9)": (6, 9),
    "morning (9-12)": (9, 12),
    "mid-day (12-14)": (12, 14),
    "afternoon (14-17)": (14, 17),
    "evening peak (17-19)": (17, 19),
    "evening (19-22)": (19, 22),
    "night (22-6)": (22, 6),
}

TRIP_COLUMNS = {
    "trip_id": "trip_id",
    "user_id_fors": "trip_user_id_fors",
    "user_id_motiontag": "trip_user_id_motiontag",
    "started_at": "trip_started_at",
    "finished_at": "trip_finished_at",
    "purpose": "trip_purpose",
    "modes": "trip_modes",
    "trip_duration": "trip_duration_seconds",
    "distance": "trip_distance_m",
    "leading_activity_id": "leading_activity_id",
}
SURVEY_COLUMNS = ["space_home", "space_work", "zone_domicile", "zone_travail", "KLASSE_ARE_dom", "Genre_actuel", "age", "Weight"]

WALK_REASONS = (
    "matched",
    "no_next_leg_in_trip",
    "next_leg_not_walk",
    "negative_gap",
    "gap_too_long",
    "walk_length_unknown",
    "walk_too_long",
)


def enrichment_settings(config: dict[str, Any]) -> dict[str, Any]:
    cfg = config.get("leg_enrichment", {}) or {}
    walk = cfg.get("post_car_walk", {}) or {}
    return {
        "enabled": bool(cfg.get("enabled", False)),
        "map_track_trip_file": cfg.get("map_track_trip_file"),
        "trips_file": cfg.get("trips_file"),
        "drop_legs_without_trip": bool(cfg.get("drop_legs_without_trip", True)),
        "local_timezone": cfg.get("local_timezone", "Europe/Zurich"),
        "add_survey": bool(cfg.get("add_survey", True)),
        "walk_enabled": bool(walk.get("enabled", True)),
        "walk_modes": list(walk.get("walk_modes", ["Mode::Walk"])),
        "max_gap_seconds": float(walk.get("max_gap_minutes", 15)) * 60,
        "max_walk_distance_m": float(walk.get("max_walk_distance_m", 2000)),
        "write_walk_waypoints": bool(walk.get("write_waypoints", True)),
        "car_modes": list(config["leg_matching"].get("car_modes", ["Mode::Car", "Mode::Ecar"])),
    }


def classify_time_of_day(hours: pd.Series) -> pd.Series:
    """Tranche horaire selon les bins historiques (vectorisé)."""
    labels = pd.Series(pd.NA, index=hours.index, dtype="object")
    for label, (start, end) in TIME_OF_DAY_BINS.items():
        mask = hours.ge(start) & hours.lt(end) if start < end else hours.ge(start) | hours.lt(end)
        labels[mask.fillna(False) & labels.isna()] = label
    return labels


def _read_csv(config: dict[str, Any], path_value: str, columns: list[str]) -> pd.DataFrame:
    path = resolve_path(config, path_value)
    if not path.exists():
        raise FileNotFoundError(f"leg_enrichment : fichier introuvable {path}")
    header = list(pd.read_csv(path, nrows=0).columns)
    missing = [column for column in columns if column not in header]
    if missing:
        raise KeyError(f"{path.name} : colonnes absentes {missing} ; colonnes disponibles : {header}")
    return pd.read_csv(path, usecols=columns, dtype="string", low_memory=False)


def _match_post_car_walk(legs: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    """Leg suivante du même trip pour chaque leg ; marche retenue après une leg voiture."""
    legs = legs.sort_values(["trip_id", "leg_started_at", "leg_finished_at"], kind="mergesort")
    grouped = legs.groupby("trip_id", sort=False)
    legs["leg_order_in_trip"] = grouped.cumcount()
    legs["legs_in_trip"] = grouped["leg_id"].transform("size")
    is_car = legs["effective_mode"].isin(settings["car_modes"])
    last_car = legs["leg_order_in_trip"].where(is_car).groupby(legs["trip_id"]).transform("max")
    legs["is_last_car_leg_in_trip"] = is_car & legs["leg_order_in_trip"].eq(last_car)

    legs["next_leg_id"] = grouped["leg_id"].shift(-1)
    legs["next_leg_mode"] = grouped["effective_mode"].shift(-1)
    legs["next_leg_gap_seconds"] = (grouped["leg_started_at"].shift(-1) - legs["leg_finished_at"]).dt.total_seconds()
    legs["next_leg_length_m"] = grouped["leg_length_m"].shift(-1)
    legs["next_leg_duration_seconds"] = grouped["leg_duration_seconds"].shift(-1)

    reason = pd.Series(pd.NA, index=legs.index, dtype="object")
    if settings["walk_enabled"]:
        gap, length = legs["next_leg_gap_seconds"], legs["next_leg_length_m"]
        rules = [
            ("no_next_leg_in_trip", legs["next_leg_id"].isna()),
            ("next_leg_not_walk", ~legs["next_leg_mode"].isin(settings["walk_modes"])),
            ("negative_gap", gap.lt(0)),
            ("gap_too_long", gap.gt(settings["max_gap_seconds"])),
            ("walk_length_unknown", length.isna()),
            ("walk_too_long", length.gt(settings["max_walk_distance_m"])),
        ]
        for label, mask in rules:
            reason[is_car & reason.isna() & mask.fillna(False)] = label
        reason[is_car & reason.isna()] = "matched"
    legs["post_car_walk_reason"] = reason
    legs["post_car_walk_matched"] = reason.eq("matched").fillna(False).astype(bool)
    legs["post_car_walk_leg_id"] = legs["next_leg_id"].where(legs["post_car_walk_matched"])
    legs["post_car_walk_gap_seconds"] = legs["next_leg_gap_seconds"].where(legs["post_car_walk_matched"])
    legs["post_car_walk_length_m"] = legs["next_leg_length_m"].where(legs["post_car_walk_matched"])
    legs["post_car_walk_duration_seconds"] = legs["next_leg_duration_seconds"].where(legs["post_car_walk_matched"])
    return legs


def enrich_legs(tracks: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Ajoute trip, motif, moment, questionnaire et marche après la voiture aux legs.

    Retourne la table enrichie (une ligne par leg) et un résumé. Les colonnes
    ``trip_id`` / ``journey_id`` remplacent celles de la source de legs.
    """
    settings = enrichment_settings(config)
    report: dict[str, Any] = {"leg_enrichment_enabled": settings["enabled"]}
    if not settings["enabled"]:
        return tracks, report

    legs = tracks.copy()
    legs_in = int(len(legs))

    # 1. Trip et journey de chaque leg
    mapping = _read_csv(config, settings["map_track_trip_file"], ["leg_id", "trip_id", "journey_id"])
    mapping = mapping.dropna(subset=["leg_id"]).drop_duplicates("leg_id")
    legs = legs.drop(columns=["trip_id", "journey_id"], errors="ignore").merge(mapping, on="leg_id", how="left")
    without_trip = legs["trip_id"].isna()
    report.update({"legs_input": legs_in, "legs_with_trip": int((~without_trip).sum()),
                   "legs_without_trip": int(without_trip.sum()),
                   "legs_without_trip_dropped": settings["drop_legs_without_trip"]})
    if settings["drop_legs_without_trip"]:
        legs = legs[~without_trip].copy()

    # 2. Informations du trip
    trips = _read_csv(config, settings["trips_file"], list(TRIP_COLUMNS)).rename(columns=TRIP_COLUMNS)
    trips = trips.drop_duplicates("trip_id")
    for column in ("trip_started_at", "trip_finished_at"):
        trips[column] = pd.to_datetime(trips[column], errors="coerce", utc=True)
    for column in ("trip_duration_seconds", "trip_distance_m"):
        trips[column] = pd.to_numeric(trips[column], errors="coerce")
    legs = legs.merge(trips, on="trip_id", how="left")
    report["legs_with_trip_info"] = int(legs["trip_started_at"].notna().sum())

    # 3. Motif agrégé, moment de la journée, jour
    legs["purpose_agg"] = legs["trip_purpose"].map(PURPOSE_AGG).fillna("Autres")
    local_end = legs["leg_finished_at"].dt.tz_convert(settings["local_timezone"])
    legs["time_of_day_end_trip"] = classify_time_of_day(local_end.dt.hour)
    legs["weekday"] = local_end.dt.day_name()
    legs["time_reference_timezone"] = settings["local_timezone"]

    # 4. Questionnaire (identifiant panel : legs_user_id_original, sinon celui du trip)
    report["survey_joined"] = False
    if settings["add_survey"]:
        from .pl23_survey import load_pl23_survey_parking

        panel_id = legs["legs_user_id_original"] if "legs_user_id_original" in legs else pd.Series(pd.NA, index=legs.index)
        legs["user_id_fors"] = panel_id.astype("string").fillna(legs["trip_user_id_fors"].astype("string"))
        survey = load_pl23_survey_parking(config, legs["user_id_fors"].dropna().unique())
        keep = ["user_id_fors", *[column for column in SURVEY_COLUMNS if column in survey.columns]]
        legs = legs.drop(columns=[c for c in SURVEY_COLUMNS if c in legs.columns]).merge(survey[keep], on="user_id_fors", how="left")
        legs["in_survey"] = legs["user_id_fors"].isin(survey["user_id_fors"])
        report.update({"survey_joined": True, "legs_in_survey": int(legs["in_survey"].sum()),
                       "users_in_survey": int(legs.loc[legs["in_survey"], "user_id_fors"].nunique())})

    # 5. Marche après la voiture
    legs = _match_post_car_walk(legs, settings)
    is_car = legs["effective_mode"].isin(settings["car_modes"])
    report.update(
        {
            "car_legs": int(is_car.sum()),
            "car_legs_last_in_trip": int(legs["is_last_car_leg_in_trip"].sum()),
            "post_car_walk_enabled": settings["walk_enabled"],
            "post_car_walk_max_gap_minutes": settings["max_gap_seconds"] / 60,
            "post_car_walk_max_distance_m": settings["max_walk_distance_m"],
            "post_car_walk_modes": settings["walk_modes"],
            "post_car_walk_matched": int(legs["post_car_walk_matched"].sum()),
            "post_car_walk_reasons": {
                key: int(value) for key, value in legs.loc[is_car, "post_car_walk_reason"].value_counts().items()
            },
            "next_leg_modes_after_car": {
                str(key): int(value) for key, value in legs.loc[is_car, "next_leg_mode"].value_counts(dropna=False).head(12).items()
            },
            "local_timezone": settings["local_timezone"],
            "legs_output": int(len(legs)),
        }
    )
    # Ordre d'origine (utilisateur, début) : celui qu'attend l'appariement des waypoints.
    legs = legs.sort_values(["source_user_id", "leg_started_at", "leg_finished_at"], kind="mergesort")
    return legs.reset_index(drop=True), report


def post_car_walk_lookup(legs: pd.DataFrame) -> pd.DataFrame:
    """Correspondance leg de marche → leg voiture qui la précède (marches retenues)."""
    if "post_car_walk_leg_id" not in legs:
        return pd.DataFrame(columns=["leg_id", "car_leg_id"])
    matched = legs.loc[legs["post_car_walk_matched"], ["leg_id", "post_car_walk_leg_id", "post_car_walk_gap_seconds"]]
    return matched.rename(columns={"leg_id": "car_leg_id", "post_car_walk_leg_id": "leg_id"}).reset_index(drop=True)


WALK_WAYPOINT_COLUMNS = [
    "waypoint_id", "source_user_id", "source_file", "source_row_number", "timestamp", "lat", "lon",
    "accuracy", "speed_kmh", "dt_seconds", "distance_from_previous_m", "leg_id", "effective_mode",
    "leg_started_at", "leg_finished_at", "trip_id", "leg_match_status",
]


def post_car_walk_waypoints(matched: pd.DataFrame, lookup: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Waypoints appariés aux marches retenues, avec la leg voiture qui précède (``car_leg_id``)."""
    accuracy = config["raw_waypoints"].get("accuracy_col", "accuracy")
    if lookup.empty or matched.empty:
        return pd.DataFrame(columns=[*WALK_WAYPOINT_COLUMNS, "car_leg_id", "post_car_walk_gap_seconds"])
    walk = matched[matched["leg_match_status"].eq("matched") & matched["leg_id"].isin(set(lookup["leg_id"]))]
    columns = [accuracy if c == "accuracy" else c for c in WALK_WAYPOINT_COLUMNS]
    walk = walk[[c for c in columns if c in walk.columns]].merge(lookup, on="leg_id", how="left")
    return walk.sort_values(["leg_id", "timestamp"], kind="mergesort").reset_index(drop=True)
