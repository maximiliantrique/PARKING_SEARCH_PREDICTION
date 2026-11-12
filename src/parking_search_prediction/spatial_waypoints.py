"""Étape 22 — détection spatiale de la recherche de stationnement, calculée sur les waypoints.

Même population et même destination que la méthode temporelle : legs voiture de
l'étape 01 (``waypoints_caronly``), destination = **dernier waypoint voiture**.
Tout est calculé à partir des waypoints (aucune géométrie de leg), en EPSG:2056.

Fenêtre de destination
----------------------
Cercle de rayon ``buffer_radius_m`` centré sur la destination. La fenêtre part du
**point d'entrée** : intersection, interpolée entre deux waypoints, de la trace
avec le cercle lors de sa **dernière** entrée ; elle va jusqu'au dernier waypoint.
Si la trace démarre déjà dans le cercle, la fenêtre part du premier waypoint.

Métriques (sur la fenêtre)
--------------------------
1. ``spatial_dist_excess_m`` : distance observée − plus court chemin routé
   (si ``legs/routing_by_leg.parquet`` existe ; les demandes de routage sont
   écrites par ``write_routing_requests``) ;
2. ``spatial_window_distance_m`` / ``spatial_window_duration_s`` : distance
   parcourue et temps passé dans le cercle, **mesurés** entre waypoints ;
3. boucle : un waypoint repasse à moins de ``loop_radius_m`` d'un point visité au
   moins ``loop_min_delay_s`` plus tôt, après s'en être éloigné d'au moins
   ``loop_min_excursion_m`` (pour ne pas confondre un arrêt avec un repassage) ;
4. recul (éloignement de la destination) : chaque déplacement entre deux
   waypoints consécutifs est projeté sur l'axe point d'entrée → destination ;
   ``spatial_backtrack_distance_m`` est la somme des projections **négatives**
   (distance parcourue en s'éloignant de la destination). Une approche directe
   donne 0 ; dépasser la destination de 300 m puis revenir donne 300 m. Les
   détours perpendiculaires à l'axe ne sont pas comptés (voir la boucle) ;
5. passage près du point de référence (activité de destination si elle est
   localisée, sinon la destination) : un waypoint à moins de
   ``near_reference_radius_m``, hors des ``near_reference_exclusion_end_m``
   derniers mètres de trace.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ensure_output_dirs, output_path, resolve_path
from .utils import write_json

CRS_METRIC = "EPSG:2056"
ROUTE_TYPE = "buffer_entry_to_leg_end"                 # OD brute : point d'entrée interpolé → dernier waypoint
ROUTE_TYPE_MATCHED = "matched_entry_to_matched_end"     # OD corrigée : 1re → dernière position recalée par /match
# référence → (préfixe des colonnes de routing_by_leg, type de route)
ROUTING_REFERENCES = {"raw_od": ("routing_", ROUTE_TYPE), "matched_od": ("routing_matched_od_", ROUTE_TYPE_MATCHED)}

DEFAULTS: dict[str, Any] = {
    # Fenêtre
    "buffer_radius_m": 1000.0,
    # Règles propres à la méthode spatiale, appliquées EN PLUS du corpus commun
    # (legs/corpus_by_leg.parquet, filtres leg_filters du YAML). Désactivées par
    # défaut : la méthode spatiale traite alors exactement le corpus commun.
    "require_last_car_leg_in_trip": False,
    "allowed_post_car_modes": ["Mode::Walk"],
    "exclude_non_allowed_post_car_modes": False,
    "min_speed_kmh": None,
    "max_speed_kmh": None,
    "max_parking_activity_distance_m": None,
    "require_activity_location": False,
    "min_window_points": None,
    "exclude_starts_in_buffer": False,
    "h3_resolution": 8,
    # Métriques
    "min_step_m": 10.0,                     # anti-bruit : un waypoint à moins de 10 m du précédent retenu est ignoré
    "loop_radius_m": 30.0,
    "loop_min_delay_s": 60.0,
    "loop_min_excursion_m": 100.0,
    "near_reference_radius_m": 25.0,
    "near_reference_exclusion_end_m": 150.0,
    "reference_point": "activity",          # "activity" (repli sur la destination) ou "destination"
    # Détection (critères combinés par « ou » ; None = critère désactivé)
    "detect_loop": True,
    "detect_passes_near_reference": True,
    "detect_backtrack_min_m": 200.0,
    "detect_window_distance_min_m": None,
    "detect_dist_excess_min_m": 200.0,
    "detect_dist_excess_matched_min_m": None,   # détour calculé sur la trace appariée (map matching)
    # Mesure du cruising pour la collection
    "search_measure": "excess_if_available",  # "excess_if_available" ou "window"
    # Fichiers
    "staypoints_file": None,
    "routing_requests_scope": "eligible",     # "eligible" ou "all_with_window"
    "routing_request_types": ["raw_od", "matched_od"],   # demandes écrites (matched_od exige le map matching)
    "matched_od_single_matching_only": True,  # OD corrigée seulement si l'appariement est d'un seul tenant
    "routing_reference": "raw_od",            # plus court chemin utilisé pour les distances excessives
}


def spatial_settings(config: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(DEFAULTS)
    cfg.update({k: v for k, v in (config.get("spatial_waypoints") or {}).items()})
    # Rayon du cercle de destination : partagé avec le corpus commun (leg_filters)
    common = config.get("leg_filters") or {}
    if common.get("destination_window_radius_m") is not None:
        cfg["buffer_radius_m"] = float(common["destination_window_radius_m"])
    return cfg


# --------------------------------------------------------------------------- #
# Dictionnaire des variables (section, colonne, description)
# --------------------------------------------------------------------------- #

VARIABLES: list[tuple[str, str, str]] = [
    ("Contexte de la leg et du trip", "leg_id", "Identifiant de leg."),
    ("Contexte de la leg et du trip", "source_user_id", "Participant (UUID Motiontag)."),
    ("Contexte de la leg et du trip", "trip_id", "Trip de la leg."),
    ("Contexte de la leg et du trip", "trip_purpose", "Motif déclaré du trip."),
    ("Contexte de la leg et du trip", "purpose_agg", "Motif regroupé (table de l'étape 23)."),
    ("Contexte de la leg et du trip", "time_of_day_end_trip", "Tranche horaire de fin de leg (heure locale)."),
    ("Contexte de la leg et du trip", "weekday", "Jour de fin de leg (heure locale)."),
    ("Contexte de la leg et du trip", "trip_duration_seconds", "Durée du trip, s."),
    ("Contexte de la leg et du trip", "is_last_car_leg_in_trip", "Dernière leg voiture du trip."),
    ("Contexte de la leg et du trip", "post_car_modes", "Modes des legs qui suivent dans le trip."),
    ("Contexte de la leg et du trip", "post_car_non_allowed_modes_count", "Legs suivantes dans un mode non autorisé."),
    ("Contexte de la leg et du trip", "post_car_walk_length_m", "Marche retenue après la voiture, m (étape 01)."),
    ("Contexte de la leg et du trip", "leading_activity_id", "Activité de destination du trip."),
    ("Contexte de la leg et du trip", "space_home", "Stationnement déclaré au domicile (questionnaire)."),
    ("Contexte de la leg et du trip", "space_work", "Stationnement déclaré au travail / études (questionnaire)."),
    ("Géométrie de base (waypoints)", "spatial_n_waypoints", "Nombre de waypoints voiture de la leg."),
    ("Géométrie de base (waypoints)", "spatial_leg_duration_s", "Durée entre premier et dernier waypoint, s."),
    ("Géométrie de base (waypoints)", "spatial_leg_length_m", "Longueur de la trace (somme des pas entre waypoints filtrés du bruit GPS), m."),
    ("Géométrie de base (waypoints)", "spatial_leg_speed_kmh", "Vitesse moyenne (longueur / durée), km/h."),
    ("Géométrie de base (waypoints)", "spatial_origin_lon", "Premier waypoint (longitude)."),
    ("Géométrie de base (waypoints)", "spatial_origin_lat", "Premier waypoint (latitude)."),
    ("Géométrie de base (waypoints)", "spatial_destination_lon", "Destination = dernier waypoint (longitude)."),
    ("Géométrie de base (waypoints)", "spatial_destination_lat", "Destination = dernier waypoint (latitude)."),
    ("Géométrie de base (waypoints)", "spatial_origin_destination_distance_m", "Distance à vol d'oiseau origine → destination, m."),
    ("Géométrie de base (waypoints)", "spatial_activity_located", "Activité de destination localisée (staypoint)."),
    ("Géométrie de base (waypoints)", "spatial_parking_activity_distance_m", "Distance destination (stationnement) → activité, m."),
    ("Géométrie de base (waypoints)", "spatial_h3_destination", "Cellule H3 de la destination."),
    ("Géométrie de base (waypoints)", "spatial_starts_in_buffer", "La trace démarre déjà dans le cercle de destination."),
    ("Éligibilité et exclusions", "spatial_eligible", "Leg du corpus commun éligible pour la méthode spatiale (aucune règle spatiale propre active ne s'applique ; toutes désactivées par défaut)."),
    ("Éligibilité et exclusions", "spatial_exclusion_reason", "Première règle d'exclusion rencontrée (ordre de la liste des règles)."),
    ("Éligibilité et exclusions", "spatial_exclusion_reasons", "Toutes les règles d'exclusion vérifiées, séparées par « ; »."),
    ("Fenêtre de destination", "spatial_entry_lon", "Point d'entrée dans le cercle (longitude, interpolé)."),
    ("Fenêtre de destination", "spatial_entry_lat", "Point d'entrée dans le cercle (latitude, interpolé)."),
    ("Fenêtre de destination", "spatial_entry_at", "Instant d'entrée dans le cercle (interpolé, UTC)."),
    ("Fenêtre de destination", "spatial_window_points", "Waypoints dans la fenêtre (après filtre anti-bruit min_step_m)."),
    ("Fenêtre de destination", "spatial_window_max_gap_s", "Plus long intervalle entre waypoints dans la fenêtre, s."),
    ("Métrique 1 — distance excessive", "spatial_dist_excess_m", "Distance entre waypoints dans la fenêtre − plus court chemin de la référence (spatial_routing_reference), m (NaN sans routage)."),
    ("Métrique 1 — distance excessive", "spatial_routing_reference", "Plus court chemin de référence : raw_od (point d'entrée interpolé → dernier waypoint) ou matched_od (1re → dernière position recalée par le map matching)."),
    ("Métrique 1 — distance excessive", "spatial_routed_distance_m", "Plus court chemin de la référence choisie, m."),
    ("Métrique 1b — distance excessive sur trace appariée", "spatial_matched_distance_m", "Longueur de la fenêtre appariée sur le réseau (OSRM /match), m (NaN si non appariée)."),
    ("Métrique 1b — distance excessive sur trace appariée", "spatial_matching_confidence", "Confiance moyenne de l'appariement OSRM (0–1)."),
    ("Métrique 1b — distance excessive sur trace appariée", "spatial_matching_n_matchings", "Nombre de tronçons appariés (> 1 : trace coupée par un trou, longueur sous-estimée)."),
    ("Métrique 1b — distance excessive sur trace appariée", "spatial_matching_detour_steps", "Pas appariés dont la distance sur le réseau dépasse 3 × le vol d'oiseau entre deux points recalés (détour forcé : sens unique, rue absente du graphe, mauvaise chaussée)."),
    ("Métrique 1b — distance excessive sur trace appariée", "spatial_dist_excess_matched_m", "Longueur appariée − plus court chemin de la référence, m : distance excessive avec la trace appariée comme distance effective (avec matched_od, mêmes extrémités sur le réseau)."),
    ("Métrique 2 — distance et temps dans le cercle", "spatial_window_distance_m", "Distance parcourue dans le cercle, mesurée entre waypoints, m."),
    ("Métrique 2 — distance et temps dans le cercle", "spatial_window_duration_s", "Temps passé dans le cercle, mesuré entre waypoints, s."),
    ("Métrique 3 — boucle", "spatial_loop_detected", "Repassage : un waypoint revient à moins de X m d'un point visité ≥ Y s plus tôt, après une excursion."),
    ("Métrique 3 — boucle", "spatial_loop_revisit_points", "Nombre de waypoints en repassage."),
    ("Métrique 3 — boucle", "spatial_loop_first_revisit_to_end_s", "Temps entre le premier repassage et la fin de la leg, s."),
    ("Métrique 4 — recul (éloignement de la destination)", "spatial_backtrack_distance_m", "Σ des déplacements entre waypoints consécutifs dont la projection sur l'axe entrée → destination est négative (on s'éloigne de la destination), m ; 0 = approche sans recul."),
    ("Métrique 4 — recul (éloignement de la destination)", "spatial_approach_efficiency_total_sum", "Σ de toutes les projections sur l'axe entrée → destination (+ en avançant, 0 de côté, − en reculant), m. Avances et reculs se compensent : vaut la distance entrée → destination (≈ le rayon) quel que soit le chemin ; contrôle de cohérence, non discriminant."),
    ("Métrique 4 — recul (éloignement de la destination)", "spatial_backtrack_share", "Part de la distance parcourue dans le cercle en s'éloignant de la destination."),
    ("Métrique 5 — passage près du point de référence", "spatial_reference_type", "Point de référence : activity ou destination."),
    ("Métrique 5 — passage près du point de référence", "spatial_passes_near_reference", "Un waypoint passe à moins de x m du point de référence, hors des derniers mètres de trace."),
    ("Métrique 5 — passage près du point de référence", "spatial_first_pass_to_end_s", "Temps entre le premier passage et la fin de la leg, s."),
    ("Éligibilité et exclusions", "spatial_status", "Statut dans la collection : evaluated, excluded_by_corpus, excluded_by_spatial_rule, no_destination_window, not_car_leg, no_car_waypoints, outside_study_area."),
    ("Détection", "spatial_flag_loop", "Critère boucle vérifié."),
    ("Détection", "spatial_flag_passes_near_reference", "Critère passage près de la référence vérifié."),
    ("Détection", "spatial_flag_backtrack", "Critère distance d'éloignement vérifié."),
    ("Détection", "spatial_flag_window_distance", "Critère distance dans le cercle vérifié."),
    ("Détection", "spatial_flag_dist_excess", "Critère distance excessive routée vérifié."),
    ("Détection", "spatial_flag_dist_excess_matched", "Critère distance excessive sur trace appariée vérifié."),
    ("Détection", "spatial_search_detected", "Cruising détecté : au moins un critère actif vérifié (legs éligibles)."),
    ("Détection", "spatial_search_distance_m", "Distance de cruising : détour routé si disponible, sinon distance dans le cercle ; 0 sans cruising."),
    ("Détection", "spatial_search_duration_s", "Durée de cruising : temps dans le cercle, au prorata du détour si routage ; 0 sans cruising."),
    ("Détection", "spatial_search_measure", "Mesure utilisée : excess (routage) ou window (cercle)."),
]


def variable_dictionary(columns: list[str] | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(VARIABLES, columns=["section", "variable", "description"])
    return frame[frame["variable"].isin(columns)] if columns is not None else frame


# --------------------------------------------------------------------------- #
# Contexte des legs et activités
# --------------------------------------------------------------------------- #

CONTEXT_COLUMNS = [
    "leg_id", "trip_id", "trip_purpose", "purpose_agg", "time_of_day_end_trip", "weekday",
    "trip_duration_seconds", "is_last_car_leg_in_trip", "leg_order_in_trip", "effective_mode",
    "post_car_walk_length_m", "leading_activity_id", "space_home", "space_work", "user_id_fors",
]


def load_leg_context(config: dict[str, Any], settings: dict[str, Any]) -> pd.DataFrame:
    """Contexte des legs (``legs_enriched``) + modes des legs qui suivent dans le trip."""
    import pyarrow.parquet as pq

    path = output_path(config, "data", "legs_enriched.parquet")
    if not path.exists():
        raise FileNotFoundError(f"{path} introuvable : lancer l'étape 01 avec l'enrichissement des legs (pl23).")
    available = set(pq.read_schema(path).names)
    legs = pd.read_parquet(path, columns=[c for c in CONTEXT_COLUMNS if c in available])
    if {"trip_id", "leg_order_in_trip"} <= set(legs.columns):
        ordered = legs.sort_values(["trip_id", "leg_order_in_trip"])
        after: dict[str, list[str]] = {}
        for _, group in ordered.groupby("trip_id", sort=False):
            modes = group["effective_mode"].astype("string").fillna("?").tolist()
            for position, leg_id in enumerate(group["leg_id"]):
                after[leg_id] = modes[position + 1:]
        allowed = set(settings["allowed_post_car_modes"])
        legs["post_car_modes"] = legs["leg_id"].map(lambda leg: ", ".join(after.get(leg, [])))
        legs["post_car_non_allowed_modes_count"] = legs["leg_id"].map(
            lambda leg: sum(mode not in allowed for mode in after.get(leg, []))
        )
    return legs


def load_activity_locations(config: dict[str, Any], settings: dict[str, Any], activity_ids: set[str]) -> pd.DataFrame:
    """Position (EPSG:2056) des activités de destination, depuis le GeoJSON des staypoints."""
    if not settings["staypoints_file"] or not activity_ids:
        return pd.DataFrame(columns=["activity_id", "activity_x", "activity_y", "activity_purpose"])
    path = resolve_path(config, settings["staypoints_file"])
    if not path.exists():
        print(f"[spatial] staypoints introuvables : {path} — référence = destination.")
        return pd.DataFrame(columns=["activity_id", "activity_x", "activity_y", "activity_purpose"])
    import pyogrio

    info = pyogrio.read_info(path)
    fields = [f for f in ("activity_id", "purpose") if f in list(info["fields"])]
    staypoints = pyogrio.read_dataframe(path, columns=fields)
    staypoints = staypoints[staypoints["activity_id"].astype(str).isin(activity_ids)].to_crs(CRS_METRIC)
    staypoints = staypoints[staypoints.geometry.notna() & ~staypoints.geometry.is_empty]
    points = staypoints.geometry.representative_point()
    return pd.DataFrame({
        "activity_id": staypoints["activity_id"].astype(str).to_numpy(),
        "activity_x": points.x.to_numpy(), "activity_y": points.y.to_numpy(),
        "activity_purpose": staypoints.get("purpose", pd.Series(pd.NA, index=staypoints.index)).to_numpy(),
    }).drop_duplicates("activity_id")


# --------------------------------------------------------------------------- #
# Métriques d'une leg
# --------------------------------------------------------------------------- #


def _entry_fraction(p0: np.ndarray, p1: np.ndarray, centre: np.ndarray, radius: float) -> float:
    """Fraction f ∈ [0, 1] du segment p0 → p1 où |p(f) − centre| = radius (p0 dehors, p1 dedans)."""
    d = p1 - p0
    f0 = p0 - centre
    a, b, c = d @ d, 2 * f0 @ d, f0 @ f0 - radius**2
    if a == 0:
        return 0.0
    disc = max(b * b - 4 * a * c, 0.0)
    roots = [(-b - np.sqrt(disc)) / (2 * a), (-b + np.sqrt(disc)) / (2 * a)]
    inside = [r for r in roots if 0.0 <= r <= 1.0]
    return float(min(inside)) if inside else 0.0


def _loop_metrics(xy: np.ndarray, t: np.ndarray, settings: dict[str, Any]) -> tuple[bool, int, float]:
    from scipy.spatial import cKDTree

    if len(xy) < 3:
        return False, 0, np.nan
    pairs = cKDTree(xy).query_pairs(settings["loop_radius_m"], output_type="ndarray")
    if len(pairs) == 0:
        return False, 0, np.nan
    i, j = pairs.min(axis=1), pairs.max(axis=1)
    keep = (t[j] - t[i]) >= settings["loop_min_delay_s"]
    i, j = i[keep], j[keep]
    if len(i) == 0:
        return False, 0, np.nan
    latest = pd.Series(i).groupby(j).max()          # pour chaque point j, le passage antérieur le plus récent
    revisits = []
    excursion = settings["loop_min_excursion_m"]
    for jj, ii in latest.items():
        between = xy[ii + 1: jj]
        if len(between) and np.max(np.hypot(*(between - xy[ii]).T)) >= excursion:
            revisits.append(jj)
    if not revisits:
        return False, 0, np.nan
    first = min(revisits)
    return True, len(revisits), float(t[-1] - t[first])


def thin_jitter(xy: np.ndarray, min_step_m: float) -> np.ndarray:
    """Indices des waypoints retenus : chacun à au moins ``min_step_m`` du précédent retenu.

    À l'arrêt ou à très basse vitesse, la position GPS « tremble » de quelques
    mètres : sans ce filtre, ce bruit gonfle les distances, l'éloignement et crée
    de faux repassages. Le dernier waypoint (destination) est toujours gardé.
    """
    if min_step_m <= 0 or len(xy) < 3:
        return np.arange(len(xy))
    keep = [0]
    last = xy[0]
    for k in range(1, len(xy) - 1):
        if (xy[k, 0] - last[0]) ** 2 + (xy[k, 1] - last[1]) ** 2 >= min_step_m**2:
            keep.append(k)
            last = xy[k]
    keep.append(len(xy) - 1)
    return np.asarray(keep)


def leg_metrics(
    xy: np.ndarray, t: np.ndarray, reference: np.ndarray | None, settings: dict[str, Any]
) -> dict[str, Any]:
    """Métriques de fenêtre d'une leg (points triés par temps, xy en mètres, t en secondes).

    Les distances, le recul et les boucles sont calculés sur
    les waypoints filtrés du bruit (``min_step_m``) ; les durées sur les
    horodatages (le filtre ne retire pas de temps : le point gardé suivant porte
    l'intervalle).
    """
    kept = thin_jitter(xy, float(settings.get("min_step_m") or 0))
    xy, t = xy[kept], t[kept]
    radius = settings["buffer_radius_m"]
    destination = xy[-1]
    r = np.hypot(*(xy - destination).T)
    outside = np.flatnonzero(r > radius)
    out: dict[str, Any] = {}
    if len(outside) == 0:
        entry, entry_t, start = xy[0], t[0], 1
        out["spatial_starts_in_buffer"] = True
    else:
        s = outside[-1]
        f = _entry_fraction(xy[s], xy[s + 1], destination, radius)
        entry = xy[s] + f * (xy[s + 1] - xy[s])
        entry_t = t[s] + f * (t[s + 1] - t[s])
        start = s + 1
        out["spatial_starts_in_buffer"] = False
    window = np.vstack([entry, xy[start:]])
    window_t = np.concatenate([[entry_t], t[start:]])
    steps = np.diff(window, axis=0)
    lengths = np.hypot(*steps.T)
    distance = float(lengths.sum())
    out.update({
        "entry_x": float(entry[0]), "entry_y": float(entry[1]),
        "entry_t": float(entry_t),
        "spatial_window_points": int(len(window) - 1),
        "spatial_window_distance_m": distance,
        "spatial_window_duration_s": float(window_t[-1] - window_t[0]),
        "spatial_window_max_gap_s": float(np.diff(window_t).max()) if len(window_t) > 1 else np.nan,
    })

    axis = destination - entry
    axis_length = float(np.hypot(*axis))
    if distance > 0 and axis_length > 0:
        projected = steps @ (axis / axis_length)
        # Somme de toutes les projections (+ en avançant, 0 de côté, − en reculant).
        # Les avances et reculs se compensent : elle vaut toujours la distance
        # entrée → destination (≈ le rayon), quel que soit le chemin. Gardée comme contrôle.
        out["spatial_approach_efficiency_total_sum"] = float(projected.sum())
        backtrack = float(-projected[projected < 0].sum()) + 0.0
        out["spatial_backtrack_distance_m"] = backtrack
        out["spatial_backtrack_share"] = backtrack / distance
    else:
        out.update({"spatial_backtrack_distance_m": 0.0, "spatial_backtrack_share": np.nan,
                    "spatial_approach_efficiency_total_sum": 0.0})

    loop, revisits, first_revisit = _loop_metrics(window[1:], window_t[1:], settings)
    out.update({"spatial_loop_detected": loop, "spatial_loop_revisit_points": revisits,
                "spatial_loop_first_revisit_to_end_s": first_revisit})

    if reference is not None:
        cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
        before_end = (distance - cumulative) >= settings["near_reference_exclusion_end_m"]
        near = np.hypot(*(window - reference).T) < settings["near_reference_radius_m"]
        hits = np.flatnonzero(near & before_end)
        out["spatial_passes_near_reference"] = bool(len(hits))
        out["spatial_first_pass_to_end_s"] = float(window_t[-1] - window_t[hits[0]]) if len(hits) else np.nan
    return out


# --------------------------------------------------------------------------- #
# Calcul sur toutes les legs (partie par partie)
# --------------------------------------------------------------------------- #


def _parts(path: Path) -> list[Path]:
    return sorted(path.glob("*.parquet")) if path.is_dir() else [path]


def compute_leg_metrics(config: dict[str, Any], settings: dict[str, Any], activities: pd.DataFrame,
                        context: pd.DataFrame) -> pd.DataFrame:
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:4326", CRS_METRIC, always_xy=True)
    back = Transformer.from_crs(CRS_METRIC, "EPSG:4326", always_xy=True)
    activity_of_leg = context.set_index("leg_id")["leading_activity_id"] if "leading_activity_id" in context else pd.Series(dtype=object)
    activity_xy = activities.set_index("activity_id")[["activity_x", "activity_y"]]
    rows = []
    started = time.perf_counter()
    from .leg_corpus import corpus_leg_ids

    corpus_ids = corpus_leg_ids(config)
    parts = _parts(output_path(config, "data", "waypoints_caronly.parquet"))
    for number, part in enumerate(parts, start=1):
        df = pd.read_parquet(part, columns=["leg_id", "source_user_id", "timestamp", "lat", "lon"])
        df = df[df["leg_id"].isin(corpus_ids)]      # corpus commun seulement
        if df.empty:
            continue
        df = df.sort_values(["leg_id", "timestamp"], kind="mergesort")
        x, y = transformer.transform(df["lon"].to_numpy(), df["lat"].to_numpy())
        df["x"], df["y"] = x, y
        df["t"] = (pd.to_datetime(df["timestamp"], utc=True) - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
        for leg_id, group in df.groupby("leg_id", sort=False):
            xy = group[["x", "y"]].to_numpy()
            t = group["t"].to_numpy()
            row: dict[str, Any] = {"leg_id": leg_id, "source_user_id": group["source_user_id"].iloc[0],
                                   "spatial_n_waypoints": int(len(group))}
            thin = xy[thin_jitter(xy, float(settings.get("min_step_m") or 0))]
            steps = np.hypot(*np.diff(thin, axis=0).T) if len(thin) > 1 else np.array([])
            row["spatial_leg_duration_s"] = float(t[-1] - t[0])
            row["spatial_leg_length_m"] = float(steps.sum())
            row["spatial_leg_speed_kmh"] = row["spatial_leg_length_m"] / row["spatial_leg_duration_s"] * 3.6 if row["spatial_leg_duration_s"] > 0 else np.nan
            row["spatial_origin_destination_distance_m"] = float(np.hypot(*(xy[-1] - xy[0])))
            row["dest_x"], row["dest_y"] = float(xy[-1][0]), float(xy[-1][1])
            row["origin_x"], row["origin_y"] = float(xy[0][0]), float(xy[0][1])
            activity_id = activity_of_leg.get(leg_id)
            activity = activity_xy.loc[activity_id].to_numpy(dtype=float) if activity_id in activity_xy.index else None
            row["spatial_activity_located"] = activity is not None
            row["spatial_parking_activity_distance_m"] = float(np.hypot(*(xy[-1] - activity))) if activity is not None else np.nan
            if settings["reference_point"] == "activity" and activity is not None:
                reference, row["spatial_reference_type"] = activity, "activity"
            else:
                reference, row["spatial_reference_type"] = xy[-1], "destination"
            if len(xy) >= 2:
                row.update(leg_metrics(xy, t, reference, settings))
            rows.append(row)
        print(f"[spatial] partie {number}/{len(parts)} — {len(rows)} legs — {time.perf_counter() - started:.0f} s", flush=True)

    metrics = pd.DataFrame(rows)
    if metrics.empty:
        return metrics
    for prefix in ("origin", "dest", "entry"):
        if f"{prefix}_x" in metrics:
            lon, lat = back.transform(metrics[f"{prefix}_x"].to_numpy(), metrics[f"{prefix}_y"].to_numpy())
            name = {"origin": "origin", "dest": "destination", "entry": "entry"}[prefix]
            metrics[f"spatial_{name}_lon"], metrics[f"spatial_{name}_lat"] = lon, lat
    if "entry_t" in metrics:
        metrics["spatial_entry_at"] = pd.to_datetime(metrics["entry_t"], unit="s", utc=True)
    from .h3_indicators import cells_for

    metrics["spatial_h3_destination"] = cells_for(
        metrics["spatial_destination_lat"].to_numpy(), metrics["spatial_destination_lon"].to_numpy(), int(settings["h3_resolution"])
    )
    return metrics


# --------------------------------------------------------------------------- #
# Éligibilité, détection, mesures
# --------------------------------------------------------------------------- #


def exclusion_rules(df: pd.DataFrame, s: dict[str, Any]) -> list[tuple[str, pd.Series]]:
    """(motif, masque) des règles spatiales actives, en plus du corpus commun ; None / False désactive la règle."""
    rules: list[tuple[str, pd.Series]] = []

    def add(name: str, active: bool, mask: pd.Series) -> None:
        if active:
            rules.append((name, mask.fillna(False).astype(bool)))

    add("no_destination_window", True, df["spatial_window_points"].isna())
    add("not_last_car_leg_in_trip", bool(s["require_last_car_leg_in_trip"]) and "is_last_car_leg_in_trip" in df,
        ~df.get("is_last_car_leg_in_trip", pd.Series(True, index=df.index)).fillna(False).astype(bool))
    add("non_allowed_post_car_mode", bool(s["exclude_non_allowed_post_car_modes"]) and "post_car_non_allowed_modes_count" in df,
        df.get("post_car_non_allowed_modes_count", pd.Series(0, index=df.index)) > 0)
    add("speed_too_low", s["min_speed_kmh"] is not None, df["spatial_leg_speed_kmh"] < (s["min_speed_kmh"] or 0))
    add("speed_too_high", s["max_speed_kmh"] is not None, df["spatial_leg_speed_kmh"] > (s["max_speed_kmh"] or np.inf))
    add("activity_not_located", bool(s["require_activity_location"]), ~df["spatial_activity_located"].astype(bool))
    add("parking_too_far_from_activity", s["max_parking_activity_distance_m"] is not None,
        df["spatial_parking_activity_distance_m"] > (s["max_parking_activity_distance_m"] or np.inf))
    add("too_few_window_points", s["min_window_points"] is not None, df["spatial_window_points"] < (s["min_window_points"] or 0))
    add("starts_in_buffer", bool(s["exclude_starts_in_buffer"]), df["spatial_starts_in_buffer"].astype("boolean"))
    return rules


def apply_eligibility(df: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    df = df.copy()
    reasons = pd.Series("", index=df.index, dtype="object")
    first = pd.Series(pd.NA, index=df.index, dtype="object")
    for name, mask in exclusion_rules(df, settings):
        df[f"spatial_excl_{name}"] = mask
        reasons = reasons.where(~mask, reasons + np.where(reasons.eq(""), "", "; ") + name)
        first = first.where(first.notna() | ~mask, name)
    df["spatial_exclusion_reasons"] = reasons.replace("", pd.NA)
    df["spatial_exclusion_reason"] = first
    df["spatial_eligible"] = first.isna()
    return df


def load_routing(config: dict[str, Any]) -> pd.DataFrame | None:
    path = output_path(config, "legs", "routing_by_leg.parquet")
    return pd.read_parquet(path).set_index("leg_id") if path.exists() else None


def routed_distance(routing: pd.DataFrame | None, reference: str) -> pd.Series:
    """Plus court chemin de la référence choisie (``raw_od`` ou ``matched_od``), NaN si non routé."""
    if reference not in ROUTING_REFERENCES:
        raise ValueError(f"routing_reference : {list(ROUTING_REFERENCES)}")
    prefix = ROUTING_REFERENCES[reference][0]
    if routing is None or f"{prefix}status" not in routing:
        return pd.Series(dtype=float)
    return routing[f"{prefix}routed_distance_m"].where(routing[f"{prefix}status"].eq("routed"))


def apply_detection(df: pd.DataFrame, settings: dict[str, Any], routing: pd.DataFrame | None,
                    matching: pd.DataFrame | None = None) -> pd.DataFrame:
    df = df.copy()
    reference = settings.get("routing_reference", "raw_od")
    df["spatial_routing_reference"] = reference
    df["spatial_routed_distance_m"] = df["leg_id"].map(routed_distance(routing, reference))
    df["spatial_dist_excess_m"] = df["spatial_window_distance_m"] - df["spatial_routed_distance_m"]
    if matching is not None and len(matching):
        ok = matching[matching["matching_status"].eq("matched")]
        df["spatial_matched_distance_m"] = df["leg_id"].map(ok["matched_distance_m"])
        df["spatial_matching_confidence"] = df["leg_id"].map(ok["confidence"])
        df["spatial_matching_n_matchings"] = df["leg_id"].map(ok["n_matchings"])
        df["spatial_matching_detour_steps"] = df["leg_id"].map(ok["detour_steps_gt3"]) if "detour_steps_gt3" in ok else np.nan
    else:
        df["spatial_matching_detour_steps"] = np.nan
        df["spatial_matched_distance_m"] = np.nan
        df["spatial_matching_confidence"] = np.nan
        df["spatial_matching_n_matchings"] = np.nan
    df["spatial_dist_excess_matched_m"] = df["spatial_matched_distance_m"] - df["spatial_routed_distance_m"]
    # Routage et appariement sont mis en cache d'un run à l'autre : une leg exclue avec les
    # règles actuelles peut y figurer. Ses métriques de détour sont neutralisées (NaN).
    not_eligible = ~df["spatial_eligible"].astype(bool)
    for column in ("spatial_routed_distance_m", "spatial_dist_excess_m", "spatial_matched_distance_m",
                   "spatial_matching_confidence", "spatial_matching_n_matchings", "spatial_matching_detour_steps",
                   "spatial_dist_excess_matched_m"):
        df.loc[not_eligible, column] = np.nan
    s = settings
    flags = {
        "loop": (bool(s["detect_loop"]), df["spatial_loop_detected"].fillna(False).astype(bool)),
        "passes_near_reference": (bool(s["detect_passes_near_reference"]),
                                  df.get("spatial_passes_near_reference", pd.Series(False, index=df.index)).fillna(False).astype(bool)),
        "backtrack": (s["detect_backtrack_min_m"] is not None,
                      df["spatial_backtrack_distance_m"].ge(s["detect_backtrack_min_m"] or np.inf).fillna(False)),
        "window_distance": (s["detect_window_distance_min_m"] is not None,
                            df["spatial_window_distance_m"].ge(s["detect_window_distance_min_m"] or np.inf).fillna(False)),
        "dist_excess": (s["detect_dist_excess_min_m"] is not None,
                        df["spatial_dist_excess_m"].ge(s["detect_dist_excess_min_m"] or np.inf).fillna(False)),
        "dist_excess_matched": (s.get("detect_dist_excess_matched_min_m") is not None,
                                df["spatial_dist_excess_matched_m"].ge(s.get("detect_dist_excess_matched_min_m") or np.inf).fillna(False)),
    }
    detected = pd.Series(False, index=df.index)
    for name, (active, mask) in flags.items():
        df[f"spatial_flag_{name}"] = mask & active
        if active:
            detected |= mask
    eligible = df["spatial_eligible"]
    df["spatial_search_detected"] = pd.array(np.where(eligible, detected, pd.NA), dtype="boolean")

    use_excess = (s["search_measure"] == "excess_if_available") & df["spatial_dist_excess_m"].notna()
    excess = df["spatial_dist_excess_m"].clip(lower=0)
    ratio = (excess / df["spatial_window_distance_m"]).clip(upper=1).fillna(0)
    distance = np.where(use_excess, excess, df["spatial_window_distance_m"])
    duration = np.where(use_excess, df["spatial_window_duration_s"] * ratio, df["spatial_window_duration_s"])
    hit = df["spatial_search_detected"].fillna(False).astype(bool)
    df["spatial_search_distance_m"] = np.where(eligible, np.where(hit, distance, 0.0), np.nan)
    df["spatial_search_duration_s"] = np.where(eligible, np.where(hit, duration, 0.0), np.nan)
    df["spatial_search_measure"] = np.where(use_excess, "excess", "window")
    return df


# --------------------------------------------------------------------------- #
# Demandes de routage
# --------------------------------------------------------------------------- #


def _request_ids(frame: pd.DataFrame) -> list[str]:
    return [
        hashlib.sha1(f"{t}|{a:.6f}|{b:.6f}|{c:.6f}|{d:.6f}".encode()).hexdigest()[:16]
        for t, a, b, c, d in frame[["route_type", "origin_lon", "origin_lat", "destination_lon", "destination_lat"]].to_numpy()
    ]


def _matched_od_requests(config: dict[str, Any], legs: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    """OD corrigée : première et dernière position recalée par /match (tracepoints), distance observée = longueur appariée."""
    from .map_matching import load_matching

    matching = load_matching(config)
    tp_path = output_path(config, "data", "map_matching_tracepoints.parquet")
    if matching is None or not tp_path.exists():
        print("[spatial] OD corrigée : pas d'appariement disponible — lancer d'abord la cellule de map matching.")
        return pd.DataFrame()
    ok = matching[matching["matching_status"].eq("matched")]
    if settings.get("matched_od_single_matching_only", True):
        ok = ok[ok["n_matchings"].eq(1)]
    ok = ok[ok.index.isin(set(legs["leg_id"]))]
    tracepoints = pd.read_parquet(tp_path)
    tracepoints = tracepoints[tracepoints["leg_id"].isin(set(ok.index)) & tracepoints["snapped_lon"].notna()]
    ends = tracepoints.groupby("leg_id", sort=False).agg(
        origin_lon=("snapped_lon", "first"), origin_lat=("snapped_lat", "first"),
        destination_lon=("snapped_lon", "last"), destination_lat=("snapped_lat", "last"))
    duration = legs.set_index("leg_id")["spatial_window_duration_s"]
    frame = ends.join(ok["matched_distance_m"].rename("observed_distance_m")).reset_index()
    frame["observed_duration_s"] = frame["leg_id"].map(duration)
    frame["route_type"] = ROUTE_TYPE_MATCHED
    return frame


def write_routing_requests(config: dict[str, Any], legs: pd.DataFrame, settings: dict[str, Any]) -> dict[str, Any]:
    """Écrit ``data/routing_requests.parquet`` : une ou deux paires origine → destination par leg.

    - ``raw_od`` (``buffer_entry_to_leg_end``) : point d'entrée interpolé → dernier waypoint ;
      distance observée = distance entre waypoints dans le cercle ;
    - ``matched_od`` (``matched_entry_to_matched_end``) : première → dernière position recalée
      par le map matching ; distance observée = longueur appariée. Les deux distances couvrent
      alors exactement les mêmes extrémités sur le réseau.

    ``routing_request_id`` est déterministe (type de route + coordonnées arrondies au
    millionième de degré) : un nouveau run ne route que les paires nouvelles.
    """
    scope = settings["routing_requests_scope"]
    mask = legs["spatial_window_points"].notna() & legs["spatial_entry_lon"].notna()
    if scope == "eligible":
        mask &= legs["spatial_eligible"]
    elif scope != "all_with_window":
        raise ValueError("routing_requests_scope : 'eligible' ou 'all_with_window'.")
    types = list(settings.get("routing_request_types") or ["raw_od"])
    frames = []
    if "raw_od" in types:
        frames.append(pd.DataFrame({
            "leg_id": legs.loc[mask, "leg_id"].to_numpy(),
            "route_type": ROUTE_TYPE,
            "origin_lon": legs.loc[mask, "spatial_entry_lon"].to_numpy(),
            "origin_lat": legs.loc[mask, "spatial_entry_lat"].to_numpy(),
            "destination_lon": legs.loc[mask, "spatial_destination_lon"].to_numpy(),
            "destination_lat": legs.loc[mask, "spatial_destination_lat"].to_numpy(),
            "observed_distance_m": legs.loc[mask, "spatial_window_distance_m"].to_numpy(),
            "observed_duration_s": legs.loc[mask, "spatial_window_duration_s"].to_numpy(),
        }))
    if "matched_od" in types:
        matched = _matched_od_requests(config, legs[mask], settings)
        if len(matched):
            frames.append(matched)
    requests = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["leg_id", "route_type"])
    for column in ("origin_lon", "origin_lat", "destination_lon", "destination_lat"):
        requests[column] = requests[column].astype(float).round(6)
    requests["routing_request_id"] = _request_ids(requests) if len(requests) else []
    requests["buffer_radius_m"] = settings["buffer_radius_m"]
    requests["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path = output_path(config, "data", "routing_requests.parquet")
    requests.to_parquet(path, index=False)
    summary = {"path": str(path), "scope": scope, "requests": int(len(requests)),
               "by_route_type": {str(k): int(v) for k, v in requests["route_type"].value_counts().items()},
               "buffer_radius_m": settings["buffer_radius_m"]}
    write_json(output_path(config, "reports", "22_routing_requests_summary.json"), summary)
    return summary


# --------------------------------------------------------------------------- #
# Table de la collection
# --------------------------------------------------------------------------- #

COLLECTION_COLUMNS = [
    # statut et détection (contrat commun des méthodes)
    "spatial_status", "spatial_exclusion_reason", "spatial_exclusion_reasons", "spatial_search_detected",
    "spatial_search_duration_s", "spatial_search_distance_m", "spatial_search_measure",
    # critères de détection vérifiés
    "spatial_flag_loop", "spatial_flag_passes_near_reference", "spatial_flag_backtrack",
    "spatial_flag_window_distance", "spatial_flag_dist_excess", "spatial_flag_dist_excess_matched",
    # fenêtre de destination
    "spatial_entry_at", "spatial_entry_lon", "spatial_entry_lat", "spatial_starts_in_buffer",
    "spatial_window_points", "spatial_window_max_gap_s",
    # métrique 1 / 1b : distance excessive routée et sur trace appariée
    "spatial_routing_reference", "spatial_routed_distance_m", "spatial_dist_excess_m", "spatial_matched_distance_m",
    "spatial_dist_excess_matched_m", "spatial_matching_confidence", "spatial_matching_n_matchings",
    "spatial_matching_detour_steps",
    # métrique 2 : distance et temps dans le cercle
    "spatial_window_distance_m", "spatial_window_duration_s",
    # métrique 3 : boucle
    "spatial_loop_detected", "spatial_loop_revisit_points", "spatial_loop_first_revisit_to_end_s",
    # métrique 4 : recul
    "spatial_backtrack_distance_m", "spatial_backtrack_share", "spatial_approach_efficiency_total_sum",
    # métrique 5 : passage près de la référence
    "spatial_reference_type", "spatial_passes_near_reference", "spatial_first_pass_to_end_s",
    # géométrie de base
    "spatial_n_waypoints", "spatial_leg_duration_s", "spatial_leg_length_m", "spatial_leg_speed_kmh",
    "spatial_origin_destination_distance_m", "spatial_activity_located", "spatial_parking_activity_distance_m",
    "spatial_destination_lon", "spatial_destination_lat", "spatial_h3_destination",
    "spatial_run_id",
]


def build_spatial_by_leg(config: dict[str, Any], legs: pd.DataFrame, run_id: str) -> pd.DataFrame:
    """Une ligne par leg : statut du corpus commun, puis règles spatiales propres."""
    from .leg_corpus import load_corpus

    corpus = load_corpus(config)[["leg_id", "corpus_status"]]
    table = corpus.merge(legs.assign(spatial_run_id=run_id), on="leg_id", how="left")
    status = pd.Series(pd.NA, index=table.index, dtype="object")
    for value in ("not_car_leg", "no_car_waypoints", "outside_study_area"):
        status[table["corpus_status"].eq(value)] = value
    status[status.isna() & table["corpus_status"].eq("excluded")] = "excluded_by_corpus"
    status[status.isna() & table["spatial_eligible"].isna()] = "no_destination_window"
    status[status.isna() & table["spatial_eligible"].eq(False)] = "excluded_by_spatial_rule"
    status[status.isna()] = "evaluated"
    table["spatial_status"] = status
    for column in COLLECTION_COLUMNS:
        if column not in table:
            table[column] = np.nan
    out = table[["leg_id", *COLLECTION_COLUMNS]].copy()
    # types stables pour le Parquet : booléens avec valeurs manquantes, textes
    for column in out.columns:
        if out[column].dtype == object:
            values = set(out[column].dropna().unique())
            out[column] = out[column].astype("boolean" if values <= {True, False} else "string")
    return out


# --------------------------------------------------------------------------- #
# Point d'entrée
# --------------------------------------------------------------------------- #


def run_spatial_detection(config: dict[str, Any], *, write_requests: bool = True) -> dict[str, Any]:
    ensure_output_dirs(config)
    settings = spatial_settings(config)
    started = time.perf_counter()
    context = load_leg_context(config, settings)
    activity_ids = set(context["leading_activity_id"].dropna().astype(str)) if "leading_activity_id" in context else set()
    activities = load_activity_locations(config, settings, activity_ids)
    metrics = compute_leg_metrics(config, settings, activities, context)
    legs = metrics.merge(context.drop(columns=["effective_mode"], errors="ignore"), on="leg_id", how="left")
    legs = apply_eligibility(legs, settings)
    routing = load_routing(config)
    from .map_matching import load_matching

    legs = apply_detection(legs, settings, routing, load_matching(config))
    legs = legs.drop(columns=[c for c in ("entry_x", "entry_y", "entry_t", "dest_x", "dest_y", "origin_x", "origin_y") if c in legs])

    detail_path = output_path(config, "data", "spatial_waypoints_candidates.parquet")
    legs.to_parquet(detail_path, index=False)
    exclusions = (
        pd.DataFrame({name.replace("spatial_excl_", ""): legs[name].sum() for name in legs if name.startswith("spatial_excl_")}, index=["legs"])
        .T.rename_axis("motif").reset_index()
    )
    exclusions["premier_motif"] = exclusions["motif"].map(legs["spatial_exclusion_reason"].value_counts()).fillna(0).astype(int)
    exclusions.to_csv(output_path(config, "reports", "22_spatial_exclusions.csv"), index=False)
    dictionary = variable_dictionary(list(legs.columns))
    dictionary.to_csv(output_path(config, "reports", "22_spatial_dictionary.csv"), index=False)

    requests = write_routing_requests(config, legs, settings) if write_requests else None

    from .leg_collection import write_leg_table

    run_id = datetime.now().strftime("%Y%m%dT%H%M%S") + "_spatial_waypoints"
    by_leg = build_spatial_by_leg(config, legs, run_id)
    collection_path = write_leg_table(by_leg, config, "spatial_by_leg", run_id=run_id, params={
        k: settings[k] for k in settings if k not in ("excluded_destination_h3_cells",)
    })

    eligible = legs["spatial_eligible"]
    summary = {
        "legs_corpus": int(len(legs)),
        "legs_eligible": int(eligible.sum()),
        "legs_detected": int(legs["spatial_search_detected"].fillna(False).sum()),
        "detected_share_of_eligible": float(legs.loc[eligible, "spatial_search_detected"].astype(bool).mean()) if eligible.any() else None,
        "flags_among_eligible": {c.replace("spatial_flag_", ""): int(legs.loc[eligible, c].sum()) for c in legs if c.startswith("spatial_flag_")},
        "routing_available": routing is not None,
        "activity_located_share": float(legs["spatial_activity_located"].mean()),
        "starts_in_buffer": int(legs["spatial_starts_in_buffer"].fillna(False).sum()),
        "status_counts": {k: int(v) for k, v in by_leg["spatial_status"].value_counts().items()},
        "routing_requests": requests,
        "detail_table": str(detail_path),
        "collection_table": str(collection_path),
        "seconds": round(time.perf_counter() - started, 1),
        "settings": {k: v for k, v in settings.items() if k != "excluded_destination_h3_cells"},
    }
    write_json(output_path(config, "reports", "22_spatial_summary.json"), summary)
    return {"summary": summary, "legs": legs, "exclusions": exclusions, "dictionary": dictionary}


# --------------------------------------------------------------------------- #
# Panneau d'histogrammes
# --------------------------------------------------------------------------- #

PANEL_METRICS = [
    ("spatial_window_distance_m", "Distance dans le cercle", "m"),
    ("spatial_window_duration_s", "Temps dans le cercle", "min"),
    ("spatial_dist_excess_m", "Distance excessive (routage)", "m"),
    ("spatial_dist_excess_matched_m", "Distance excessive (trace appariée)", "m"),
    ("spatial_backtrack_share", "Part de la distance en recul", "part"),
    ("spatial_backtrack_distance_m", "Distance en s'éloignant de la destination", "m"),
    ("spatial_loop_revisit_points", "Waypoints en repassage (boucle)", "points"),
    ("spatial_loop_detected", "Boucle détectée", "bool"),
    ("spatial_passes_near_reference", "Passage près de la référence", "bool"),
    ("spatial_flag_dist_excess", "Distance excessive ≥ seuil (routage)", "bool"),
    ("spatial_flag_dist_excess_matched", "Distance excessive appariée ≥ seuil", "bool"),   
]

# Indicateurs oui/non dont la proportion se calcule seulement parmi les legs où la mesure existe
PANEL_DENOMINATORS = {
    "spatial_flag_dist_excess": "spatial_dist_excess_m",
    "spatial_flag_dist_excess_matched": "spatial_dist_excess_matched_m",
}


def plot_metrics_panel(legs: pd.DataFrame, *, selection: pd.DataFrame, selection_label: str,
                       group_by: str | None = None, quantile_cap: float = 0.99, bins: int = 40):
    """Histogrammes (en % des legs) des métriques spatiales ; superposition par modalité si ``group_by``."""
    import matplotlib.pyplot as plt

    from .indicators import PALETTE, REFERENCE_COLOR, _style

    fixed = {"Pas de place dédiée, besoin de cruise": PALETTE[0], "Pas de place dédiée, pas besoin de cruise": PALETTE[1],
             "Aucun": PALETTE[2], "Place dédiée": PALETTE[3]}
    groups = [(selection_label, selection)] if group_by is None else sorted(
        [(str(v), selection[selection[group_by].eq(v)]) for v in selection[group_by].dropna().unique()], key=lambda g: -len(g[1]))
    metrics = [m for m in PANEL_METRICS if m[0] in legs and legs[m[0]].notna().any()
               and (m[0] not in PANEL_DENOMINATORS or legs.get(PANEL_DENOMINATORS[m[0]], pd.Series(dtype=float)).notna().any())]
    ncols = 4
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.4 * nrows), squeeze=False)
    for ax, (column, title, unit) in zip(axes.flat, metrics):
        reference = legs[column].dropna()
        if unit == "bool":
            def base(frame):   # legs où la mesure existe (routées / appariées) pour les critères de détour
                needed = PANEL_DENOMINATORS.get(column)
                return frame[frame[needed].notna()] if needed in frame else frame
            subsets = [base(g[1]) for g in groups] + [base(legs)]
            shares = [s[column].astype(float).mean() * 100 if len(s) else np.nan for s in subsets]
            labels = [f"{g[0]} (n={len(s)})" for g, s in zip(groups, subsets[:-1])] + [f"toutes les legs éligibles (n={len(subsets[-1])})"]
            colors = [fixed.get(g[0], PALETTE[i % len(PALETTE)]) for i, g in enumerate(groups)] + [("#e5e7eb", REFERENCE_COLOR)]
            ax.barh(range(len(labels)), shares, color=[c[0] for c in colors], edgecolor=[c[1] for c in colors])
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels([label[:40] for label in labels], fontsize=7)
            ax.set_xlabel("% des legs")
            for k, v in enumerate(shares):
                ax.text(v if np.isfinite(v) else 0, k, f" {v:.0f} %", va="center", fontsize=7)
            finite = [v for v in shares if np.isfinite(v)]
            ax.set_xlim(0, max(max(finite, default=0) * 1.3, 5))
        else:
            scale = 1 / 60 if unit == "min" else 1.0
            values = reference * scale
            upper = values.quantile(quantile_cap) if len(values) else 1
            lower = min(values.min(), 0) if len(values) else 0
            edges = np.linspace(lower, upper if upper > lower else lower + 1, bins + 1)
            for index, (label, frame) in enumerate(groups):
                data = (frame[column].dropna() * scale).clip(lower, upper)
                if not len(data):
                    continue
                fill, line = fixed.get(label, PALETTE[index % len(PALETTE)])
                ax.hist(data, bins=edges, weights=np.full(len(data), 100 / len(data)), color=fill, edgecolor=line,
                        alpha=0.75, histtype="stepfilled", lw=0.8, label=f"{label[:45]} (n={len(data)})")
                ax.axvline(data.median(), color=line, ls=":", lw=1.2)
            ref = values.clip(lower, upper)
            ax.hist(ref, bins=edges, weights=np.full(len(ref), 100 / len(ref)), histtype="step", color=REFERENCE_COLOR,
                    ls="--", lw=1.2, label="toutes les legs éligibles")
            ax.set_xlabel(unit)
            ax.set_ylabel("% des legs")
        ax.set_title(title, fontsize=9)
        _style(ax)
    for ax in list(axes.flat)[len(metrics):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(len(handles), 4), frameon=False, fontsize=8)
    fig.suptitle(f"Métriques spatiales (legs éligibles) — filtre : {selection_label}\n"
                 f"valeurs au-delà du quantile {quantile_cap:.0%} regroupées dans la dernière classe ; pointillés : médianes ; "
                 f"tirets gris : toutes les legs éligibles", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig
