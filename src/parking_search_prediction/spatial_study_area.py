"""TODO-METH-02 — filtre géographique explicite (Genève / agglo) pour Déclic.

Mirroir léger de la logique PL23 (`pl23_spatial.py`, filtre `layer_GE`), adapté
au schéma réel des legs Déclic (`data_prep.py`) : la table des legs/tracks ne
porte aucune coordonnée de destination. La destination d'une leg voiture est
donc définie ici comme son **dernier waypoint apparié** (le point le plus
récent, par `timestamp`, parmi les waypoints voiture rattachés à cette leg par
`attach_waypoints_to_legs` + `filter_car_waypoints`).

Règle d'inclusion : la destination (ce dernier waypoint) doit être *contenue*
(within) dans le layer de périmètre fourni. Rien n'est supprimé par défaut :
le flag est toujours calculé sur `tracks`, et `car_waypoints` n'est réellement
restreint que si `spatial_detection.apply_study_area_filter` est vrai — ce qui
garde les deux dénominateurs (avant/après) auditables, comme demandé par le
critère d'acceptation du TODO.
"""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import pandas as pd

from .config import resolve_path

DEFAULT_LEG_ID_FIELD = "leg_id"
# Ces trois colonnes existent déjà sur `car_waypoints` (cf. clean_waypoints /
# attach_waypoints_to_legs dans data_prep.py) — pas besoin de les configurer
# sauf schéma différent.
DEFAULT_DESTINATION_LON_FIELD = "lon"
DEFAULT_DESTINATION_LAT_FIELD = "lat"
DEFAULT_DESTINATION_TIME_FIELD = "timestamp"


def _study_area_settings(config: dict[str, Any]) -> dict[str, Any]:
    spatial_cfg = config.get("spatial_detection", {}) or {}
    return {
        "boundary_file": spatial_cfg.get("destination_boundary_file"),
        "boundary_layer": spatial_cfg.get("destination_boundary_layer"),
        "leg_id_field": spatial_cfg.get("leg_id_field", DEFAULT_LEG_ID_FIELD),
        "lon_field": spatial_cfg.get("destination_lon_field", DEFAULT_DESTINATION_LON_FIELD),
        "lat_field": spatial_cfg.get("destination_lat_field", DEFAULT_DESTINATION_LAT_FIELD),
        "time_field": spatial_cfg.get("destination_time_field", DEFAULT_DESTINATION_TIME_FIELD),
        "source_crs": spatial_cfg.get("destination_source_crs", "EPSG:4326"),
        "apply_filter": bool(spatial_cfg.get("apply_study_area_filter", False)),
    }


def load_study_area_boundary(config: dict[str, Any]) -> gpd.GeoDataFrame | None:
    """Charge le périmètre depuis `spatial_detection.destination_boundary_file`.

    Retourne None si aucun fichier n'est configuré (filtre inactif). Les
    géométries du layer sont dissoutes en une seule, au cas où le périmètre
    comporte plusieurs polygones (ex: plusieurs communes).
    """
    settings = _study_area_settings(config)
    boundary_file = settings["boundary_file"]
    if not boundary_file:
        return None

    boundary_path = resolve_path(config, boundary_file)
    if not boundary_path.exists():
        raise FileNotFoundError(
            "spatial_detection.destination_boundary_file introuvable: "
            f"{boundary_path}"
        )

    boundary = gpd.read_file(boundary_path, layer=settings["boundary_layer"])
    if boundary.empty:
        raise ValueError(f"Le layer de périmètre est vide: {boundary_path}")

    dissolved = boundary.union_all() if hasattr(boundary, "union_all") else boundary.unary_union
    return gpd.GeoDataFrame(geometry=[dissolved], crs=boundary.crs)


def compute_leg_destinations(
    car_waypoints: pd.DataFrame,
    leg_id_field: str,
    lon_field: str,
    lat_field: str,
    time_field: str,
) -> pd.DataFrame:
    """Une ligne par leg voiture : son dernier waypoint apparié = sa destination."""
    required = {leg_id_field, lon_field, lat_field, time_field}
    missing = sorted(required - set(car_waypoints.columns))
    if missing:
        raise KeyError(
            "Colonnes manquantes dans car_waypoints pour calculer la "
            f"destination de leg: {missing}. Ajuster spatial_detection."
            "leg_id_field / destination_lon_field / destination_lat_field / "
            "destination_time_field si le schéma diffère de data_prep.py."
        )
    if car_waypoints.empty:
        return car_waypoints[[leg_id_field, lon_field, lat_field, time_field]].rename(
            columns={
                lon_field: "destination_lon",
                lat_field: "destination_lat",
                time_field: "destination_timestamp",
            }
        )

    ordered = car_waypoints.sort_values([leg_id_field, time_field])
    destinations = ordered.groupby(leg_id_field, as_index=False).tail(1)
    return destinations[[leg_id_field, lon_field, lat_field, time_field]].rename(
        columns={
            lon_field: "destination_lon",
            lat_field: "destination_lat",
            time_field: "destination_timestamp",
        }
    )


def flag_destination_in_study_area(
    tracks: pd.DataFrame, car_waypoints: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Ajoute `spatial_destination_in_study_area` / `spatial_study_area_filter_reason`
    à `tracks` (une leg = une ligne) et retourne les compteurs avant/après.
    """
    settings = _study_area_settings(config)
    tracks = tracks.copy()
    leg_id_field = settings["leg_id_field"]

    total_tracks = int(len(tracks))
    total_car_legs = (
        int(car_waypoints[leg_id_field].nunique())
        if leg_id_field in car_waypoints.columns and len(car_waypoints)
        else 0
    )

    boundary = load_study_area_boundary(config)
    if boundary is None:
        tracks["spatial_destination_in_study_area"] = pd.NA
        tracks["spatial_study_area_filter_reason"] = "no_boundary_configured"
        return tracks, {
            "study_area_filter_configured": False,
            "study_area_total_tracks": total_tracks,
            "study_area_total_car_legs": total_car_legs,
            "study_area_car_legs_with_destination": None,
            "study_area_car_legs_in_area": None,
            "study_area_car_legs_out_of_area": None,
            "study_area_car_legs_missing_destination": None,
            "study_area_filter_applied": False,
        }

    destinations = compute_leg_destinations(
        car_waypoints,
        leg_id_field,
        settings["lon_field"],
        settings["lat_field"],
        settings["time_field"],
    )

    if destinations.empty:
        destinations = destinations.assign(
            spatial_destination_in_study_area=pd.array([], dtype="boolean")
        )
    else:
        points = gpd.GeoDataFrame(
            destinations,
            geometry=gpd.points_from_xy(
                destinations["destination_lon"], destinations["destination_lat"]
            ),
            crs=settings["source_crs"],
        ).to_crs(boundary.crs)
        within = points.geometry.within(boundary.geometry.iloc[0])
        destinations = destinations.assign(
            spatial_destination_in_study_area=pd.array(within.to_numpy(), dtype="boolean")
        )

    destinations["spatial_study_area_filter_reason"] = destinations[
        "spatial_destination_in_study_area"
    ].map({True: "destination_in_study_area", False: "destination_outside_study_area"})

    tracks = tracks.merge(
        destinations[
            [leg_id_field, "spatial_destination_in_study_area", "spatial_study_area_filter_reason"]
        ],
        on=leg_id_field,
        how="left",
    )
    tracks["spatial_destination_in_study_area"] = tracks[
        "spatial_destination_in_study_area"
    ].astype("boolean")
    tracks["spatial_study_area_filter_reason"] = tracks["spatial_study_area_filter_reason"].fillna(
        "no_matched_car_destination"
    )

    legs_in_area = int((tracks["spatial_destination_in_study_area"] == True).sum())  # noqa: E712
    legs_out_of_area = int((tracks["spatial_destination_in_study_area"] == False).sum())  # noqa: E712
    legs_with_destination = int(len(destinations))
    legs_missing_destination = int(total_car_legs - legs_with_destination)

    summary = {
        "study_area_filter_configured": True,
        "study_area_boundary_file": str(settings["boundary_file"]),
        "study_area_boundary_layer": settings["boundary_layer"],
        "study_area_total_tracks": total_tracks,
        "study_area_total_car_legs": total_car_legs,
        "study_area_car_legs_with_destination": legs_with_destination,
        "study_area_car_legs_in_area": legs_in_area,
        "study_area_car_legs_out_of_area": legs_out_of_area,
        "study_area_car_legs_missing_destination": legs_missing_destination,
        "study_area_filter_applied": settings["apply_filter"],
    }
    return tracks, summary


def apply_study_area_filter(
    tracks: pd.DataFrame, car_waypoints: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Calcule le flag par leg puis, si `spatial_detection.apply_study_area_filter`
    est vrai, restreint `car_waypoints` aux legs dont la destination est dans
    le périmètre. Retourne `(car_waypoints_filtrés, tracks_flaggées, résumé)`
    avec toujours les deux dénominateurs (avant/après), que le filtre soit
    appliqué ou non — critère d'acceptation TODO-METH-02.
    """
    flagged_tracks, summary = flag_destination_in_study_area(tracks, car_waypoints, config)
    settings = _study_area_settings(config)
    leg_id_field = settings["leg_id_field"]

    if settings["apply_filter"] and summary["study_area_filter_configured"]:
        in_area_leg_ids = set(
            flagged_tracks.loc[
                flagged_tracks["spatial_destination_in_study_area"] == True,  # noqa: E712
                leg_id_field,
            ]
        )
        filtered_car_waypoints = car_waypoints[
            car_waypoints[leg_id_field].isin(in_area_leg_ids)
        ].copy()
    else:
        filtered_car_waypoints = car_waypoints

    summary["study_area_car_waypoints_before_filter"] = int(len(car_waypoints))
    summary["study_area_car_waypoints_after_filter"] = int(len(filtered_car_waypoints))
    summary["study_area_car_legs_after_filter"] = (
        int(filtered_car_waypoints[leg_id_field].nunique()) if len(filtered_car_waypoints) else 0
    )
    return filtered_car_waypoints, flagged_tracks, summary