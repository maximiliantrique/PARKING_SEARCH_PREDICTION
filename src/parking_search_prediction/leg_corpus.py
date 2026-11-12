"""Corpus commun de legs : un seul jeu de legs pour les méthodes temporelle et spatiale.

Construit à l'étape 01 à partir des waypoints voiture des legs du périmètre
(``waypoints_caronly``), avec les filtres de la section ``leg_filters`` du YAML :

====================================  ===========================================================
``max_accuracy_m``                    waypoints plus imprécis retirés (au nettoyage, avant tout)
``min_leg_points``                    nombre minimal de waypoints voiture de la leg
``min_leg_duration_s``                durée minimale (premier → dernier waypoint)
``min_leg_length_m``                  distance parcourue minimale (somme des pas entre waypoints)
``min_origin_destination_distance_m`` distance minimale origine → destination à vol d'oiseau
``max_window_gap_s``                  trou maximal entre waypoints dans le cercle de destination
``max_trip_duration_s``               durée maximale du trip
``exclude_tunnel_h3_destinations``    destination dans une cellule H3 de tunnel
====================================  ===========================================================

``destination_window_radius_m`` est le rayon du cercle de destination, partagé
avec la méthode spatiale. Une valeur ``null`` désactive le filtre.

Sortie : ``legs/corpus_by_leg.parquet`` (collection, une ligne par leg) avec
``corpus_status`` (``in_corpus``, ``excluded``, ``not_car_leg``,
``no_car_waypoints``, ``outside_study_area``), les motifs d'exclusion et les
statistiques de chaque leg (``corpus_*``). Les étapes 32 (temporel) et 22
(spatial) ne traitent que les legs ``in_corpus``.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import output_path

CRS_METRIC = "EPSG:2056"

FILTER_DEFAULTS: dict[str, Any] = {
    "max_accuracy_m": 100.0,
    "min_leg_points": 20,
    "min_leg_duration_s": 300.0,
    "min_leg_length_m": 500.0,
    "min_origin_destination_distance_m": 100.0,
    "max_window_gap_s": 120.0,
    "max_trip_duration_s": 18000.0,
    "destination_window_radius_m": 1000.0,
    "exclude_tunnel_h3_destinations": True,
    "tunnel_h3_resolution": 8,
    "tunnel_h3_cells": [],
}

# (motif, colonne de statistique, sens, clé de filtre) — dans l'ordre de priorité
RULES = [
    ("trip_too_long", "corpus_trip_duration_s", "gt", "max_trip_duration_s"),
    ("too_few_points", "corpus_n_points", "lt", "min_leg_points"),
    ("leg_too_short_duration", "corpus_duration_s", "lt", "min_leg_duration_s"),
    ("leg_too_short_length", "corpus_length_m", "lt", "min_leg_length_m"),
    ("origin_destination_too_close", "corpus_od_distance_m", "lt", "min_origin_destination_distance_m"),
    ("window_gap_too_long", "corpus_window_max_gap_s", "gt", "max_window_gap_s"),
]

STATUSES = {
    "in_corpus": "leg voiture du périmètre qui passe tous les filtres : traitée par les deux méthodes",
    "excluded": "leg voiture du périmètre écartée par un filtre (voir corpus_exclusion_reason)",
    "not_car_leg": "leg d'un autre mode",
    "no_car_waypoints": "leg voiture sans waypoint voiture apparié",
    "outside_study_area": "leg voiture dont la destination (dernier waypoint) est hors du périmètre",
}

DESCRIPTIONS = {
    "corpus_status": "Statut dans le corpus commun : " + " ; ".join(f"{k} = {v}" for k, v in STATUSES.items()),
    "corpus_exclusion_reason": "Premier filtre du corpus non vérifié (ordre : " + ", ".join(r[0] for r in RULES) + ", destination_in_tunnel_h3).",
    "corpus_exclusion_reasons": "Tous les filtres du corpus non vérifiés, séparés par « ; ».",
    "corpus_n_points": "Waypoints voiture de la leg (après le filtre de précision).",
    "corpus_duration_s": "Durée entre le premier et le dernier waypoint, s.",
    "corpus_length_m": "Distance parcourue : somme des distances entre waypoints consécutifs, m.",
    "corpus_od_distance_m": "Distance à vol d'oiseau entre le premier et le dernier waypoint, m.",
    "corpus_max_gap_s": "Plus long intervalle entre deux waypoints consécutifs de la leg, s.",
    "corpus_window_max_gap_s": "Plus long intervalle entre waypoints dans le cercle de destination (depuis le dernier waypoint hors du cercle), s.",
    "corpus_trip_duration_s": "Durée du trip (trips.csv), s.",
    "corpus_destination_h3": "Cellule H3 de la destination (dernier waypoint).",
    "corpus_run_id": "Identifiant du run de l'étape 01 qui a produit la ligne.",
}


def corpus_settings(config: dict[str, Any]) -> dict[str, Any]:
    settings = dict(FILTER_DEFAULTS)
    settings.update(config.get("leg_filters") or {})
    if not settings.get("tunnel_h3_cells"):
        spatial = config.get("spatial_waypoints") or {}
        settings["tunnel_h3_cells"] = spatial.get("excluded_destination_h3_cells", [])
    return settings


# --------------------------------------------------------------------------- #
# Statistiques par leg (vectorisé)
# --------------------------------------------------------------------------- #


def leg_stats(points: pd.DataFrame, settings: dict[str, Any], accuracy_col: str = "accuracy",
              max_accuracy_m: float | None = None) -> pd.DataFrame:
    """Une ligne par leg : points, durée, longueur, origine–destination, trous (leg et cercle), destination.

    ``max_accuracy_m`` permet de recalculer les statistiques comme si un seuil de
    précision plus strict avait été appliqué au nettoyage (analyse de sensibilité).
    """
    from pyproj import Transformer

    columns = ["leg_id", "source_user_id", "timestamp", "lat", "lon"]
    df = points[[c for c in [*columns, accuracy_col] if c in points.columns]]
    if max_accuracy_m is not None and accuracy_col in df.columns:
        df = df[df[accuracy_col].isna() | (df[accuracy_col] <= max_accuracy_m)]
    if df.empty:
        return pd.DataFrame(columns=["leg_id", *DESCRIPTIONS])
    df = df.sort_values(["leg_id", "timestamp"], kind="mergesort").reset_index(drop=True)
    x, y = Transformer.from_crs("EPSG:4326", CRS_METRIC, always_xy=True).transform(df["lon"].to_numpy(), df["lat"].to_numpy())
    t = (pd.to_datetime(df["timestamp"], utc=True) - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()
    leg = df["leg_id"].to_numpy()
    same = np.r_[False, leg[1:] == leg[:-1]]                      # le point précédent appartient à la même leg
    step = np.where(same, np.hypot(np.diff(x, prepend=x[0]), np.diff(y, prepend=y[0])), 0.0)
    gap = np.where(same, np.diff(t, prepend=t[0]), np.nan)

    frame = pd.DataFrame({"leg_id": leg, "x": x, "y": y, "t": t, "step": step, "gap": gap})
    grouped = frame.groupby("leg_id", sort=False)
    last_x, last_y = grouped["x"].transform("last").to_numpy(), grouped["y"].transform("last").to_numpy()
    outside = np.hypot(x - last_x, y - last_y) > float(settings["destination_window_radius_m"])
    position = grouped.cumcount().to_numpy()
    last_outside = pd.Series(np.where(outside, position, -1)).groupby(leg).transform("max").to_numpy()
    # pas dans la fenêtre : du dernier point hors du cercle (inclus) jusqu'à la fin
    in_window = (position > last_outside) & same
    frame["window_gap"] = np.where(in_window, gap, np.nan)

    stats = pd.DataFrame({
        "corpus_n_points": grouped.size(),
        "corpus_duration_s": grouped["t"].last() - grouped["t"].first(),
        "corpus_length_m": grouped["step"].sum(),
        "corpus_od_distance_m": np.hypot(grouped["x"].last() - grouped["x"].first(), grouped["y"].last() - grouped["y"].first()),
        "corpus_max_gap_s": grouped["gap"].max(),
        "corpus_window_max_gap_s": frame.groupby("leg_id", sort=False)["window_gap"].max(),
    })
    last_points = df.groupby("leg_id", sort=False)[["lat", "lon", "source_user_id"]].last()
    stats = stats.join(last_points.rename(columns={"lat": "corpus_destination_lat", "lon": "corpus_destination_lon"}))
    return stats.reset_index()


def _tunnel_cells(stats: pd.DataFrame, settings: dict[str, Any]) -> pd.Series:
    from .h3_indicators import cells_for

    return pd.Series(
        cells_for(stats["corpus_destination_lat"].to_numpy(), stats["corpus_destination_lon"].to_numpy(),
                  int(settings["tunnel_h3_resolution"])),
        index=stats.index,
    )


def apply_filters(stats: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    """Ajoute ``corpus_eligible``, ``corpus_exclusion_reason(s)`` et une colonne booléenne par filtre."""
    stats = stats.copy()
    first = pd.Series(pd.NA, index=stats.index, dtype="object")
    reasons = pd.Series("", index=stats.index, dtype="object")

    def mark(name: str, mask: pd.Series) -> None:
        nonlocal first, reasons
        mask = mask.fillna(False).astype(bool)
        stats[f"corpus_excl_{name}"] = mask
        reasons = reasons.where(~mask, reasons + np.where(reasons.eq(""), "", "; ") + name)
        first = first.where(first.notna() | ~mask, name)

    for name, column, direction, key in RULES:
        threshold = settings.get(key)
        if threshold is None or column not in stats:
            continue
        values = stats[column]
        mark(name, values.gt(float(threshold)) if direction == "gt" else values.lt(float(threshold)))
    cells = {str(c).strip() for c in settings.get("tunnel_h3_cells") or [] if str(c).strip()}
    if settings.get("exclude_tunnel_h3_destinations") and cells:
        if "corpus_destination_h3" not in stats:
            stats["corpus_destination_h3"] = _tunnel_cells(stats, settings)
        mark("destination_in_tunnel_h3", stats["corpus_destination_h3"].astype("string").isin(cells))
    stats["corpus_exclusion_reason"] = first
    stats["corpus_exclusion_reasons"] = reasons.replace("", pd.NA)
    stats["corpus_eligible"] = first.isna()
    return stats


# --------------------------------------------------------------------------- #
# Table de la collection
# --------------------------------------------------------------------------- #


def build_corpus(
    config: dict[str, Any], base: pd.DataFrame, stats: pd.DataFrame, car_legs_with_waypoints: set[str],
    trip_durations: pd.Series | None = None,
) -> pd.DataFrame:
    """``base`` : toutes les legs (``leg_id``, ``effective_mode``) ; ``stats`` : legs voiture du périmètre."""
    settings = corpus_settings(config)
    stats = stats.copy()
    if trip_durations is not None:
        stats["corpus_trip_duration_s"] = stats["leg_id"].map(trip_durations)
    stats = apply_filters(stats, settings)
    if "corpus_destination_h3" not in stats and len(stats):
        stats["corpus_destination_h3"] = _tunnel_cells(stats, settings)
    car_modes = set(config["leg_matching"].get("car_modes", ["Mode::Car", "Mode::Ecar"]))
    table = base[["leg_id", "effective_mode"]].drop_duplicates("leg_id").merge(stats, on="leg_id", how="left")
    status = pd.Series(pd.NA, index=table.index, dtype="object")
    status[~table["effective_mode"].isin(car_modes)] = "not_car_leg"
    status[status.isna() & ~table["leg_id"].isin(car_legs_with_waypoints)] = "no_car_waypoints"
    status[status.isna() & table["corpus_n_points"].isna()] = "outside_study_area"
    status[status.isna() & table["corpus_eligible"].eq(False)] = "excluded"
    status[status.isna()] = "in_corpus"
    table["corpus_status"] = status
    table["corpus_run_id"] = datetime.now().strftime("%Y%m%dT%H%M%S") + "_corpus"
    columns = ["leg_id", *DESCRIPTIONS]
    for column in columns:
        if column not in table:
            table[column] = np.nan
    out = table[columns].copy()
    for column in ("corpus_exclusion_reason", "corpus_exclusion_reasons", "corpus_destination_h3", "corpus_status", "corpus_run_id"):
        out[column] = out[column].astype("string")
    return out


def write_corpus(config: dict[str, Any], corpus: pd.DataFrame) -> Path:
    from .leg_collection import write_leg_table

    settings = corpus_settings(config)
    params = {k: v for k, v in settings.items() if k != "tunnel_h3_cells"}
    return write_leg_table(corpus, config, "corpus_by_leg", run_id=str(corpus["corpus_run_id"].iloc[0]), params=params)


def load_corpus(config: dict[str, Any], *, required: bool = True) -> pd.DataFrame | None:
    path = output_path(config, "legs", "corpus_by_leg.parquet")
    if not path.exists():
        if required:
            raise FileNotFoundError(f"{path} introuvable : relancer l'étape 01 (corpus commun de legs).")
        return None
    return pd.read_parquet(path)


def corpus_leg_ids(config: dict[str, Any]) -> set[str]:
    corpus = load_corpus(config)
    return set(corpus.loc[corpus["corpus_status"].eq("in_corpus"), "leg_id"])


def corpus_summary(corpus: pd.DataFrame) -> dict[str, Any]:
    car_in_area = corpus["corpus_status"].isin(["in_corpus", "excluded"])
    return {
        "legs": int(len(corpus)),
        "status_counts": {str(k): int(v) for k, v in corpus["corpus_status"].value_counts().items()},
        "car_legs_in_area": int(car_in_area.sum()),
        "in_corpus": int(corpus["corpus_status"].eq("in_corpus").sum()),
        "first_exclusion_reason": {str(k): int(v) for k, v in corpus["corpus_exclusion_reason"].value_counts().items()},
    }


# --------------------------------------------------------------------------- #
# Sensibilité (notebook 21)
# --------------------------------------------------------------------------- #

SWEEP_PARAMETERS = {
    "max_accuracy_m": "précision GPS maximale (m)",
    "min_leg_points": "nombre minimal de waypoints",
    "min_leg_duration_s": "durée minimale (s)",
    "min_leg_length_m": "distance parcourue minimale (m)",
    "min_origin_destination_distance_m": "distance origine–destination minimale (m)",
    "max_window_gap_s": "trou maximal dans le cercle (s)",
    "max_trip_duration_s": "durée maximale du trip (s)",
}


def _accuracy_stats(config: dict[str, Any], settings: dict[str, Any], thresholds: list[float]) -> dict[float, pd.DataFrame]:
    """Statistiques recalculées pour plusieurs seuils de précision, partie par partie."""
    from .spatial_waypoints import _parts

    accuracy_col = config["raw_waypoints"].get("accuracy_col", "accuracy")
    out: dict[float, list[pd.DataFrame]] = {t: [] for t in thresholds}
    for part in _parts(output_path(config, "data", "waypoints_caronly.parquet")):
        points = pd.read_parquet(part, columns=["leg_id", "source_user_id", "timestamp", "lat", "lon", accuracy_col])
        for threshold in thresholds:
            out[threshold].append(leg_stats(points, settings, accuracy_col, max_accuracy_m=threshold))
    return {t: pd.concat(frames, ignore_index=True) for t, frames in out.items()}


def sensitivity(config: dict[str, Any], sweeps: dict[str, list[Any]]) -> pd.DataFrame:
    """Nombre de legs du corpus pour chaque valeur de chaque paramètre (les autres à leur valeur du YAML).

    Les filtres de legs sont recalculés sur les statistiques du corpus (immédiat).
    La précision GPS est recalculée à partir des waypoints voiture (seuils plus
    stricts que celui de l'étape 01 seulement : les points déjà retirés ne sont plus là).
    """
    settings = corpus_settings(config)
    corpus = load_corpus(config)
    stats = corpus[corpus["corpus_status"].isin(["in_corpus", "excluded"])].copy()
    trip = stats.set_index("leg_id")["corpus_trip_duration_s"]
    current_accuracy = settings.get("max_accuracy_m")
    rows = []
    for parameter, values in sweeps.items():
        if parameter == "max_accuracy_m":
            usable = [v for v in values if v is not None and (current_accuracy is None or v <= current_accuracy)]
            skipped = [v for v in values if v not in usable]
            if skipped:
                print(f"[corpus] max_accuracy_m : valeurs {skipped} ignorées (au-delà du seuil de l'étape 01, "
                      f"{current_accuracy} m : ces points ont déjà été retirés).")
            for threshold, recomputed in _accuracy_stats(config, settings, usable).items():
                recomputed["corpus_trip_duration_s"] = recomputed["leg_id"].map(trip)
                filtered = apply_filters(recomputed, settings)
                rows.append({"parametre": parameter, "valeur": threshold, "legs_corpus": int(filtered["corpus_eligible"].sum())})
            continue
        if parameter not in settings:
            raise KeyError(f"Paramètre inconnu {parameter!r} ; disponibles : {list(SWEEP_PARAMETERS)}")
        for value in values:
            filtered = apply_filters(stats, {**settings, parameter: value})
            rows.append({"parametre": parameter, "valeur": value, "legs_corpus": int(filtered["corpus_eligible"].sum())})
    table = pd.DataFrame(rows)
    table["legs_voiture_perimetre"] = int(len(stats))
    table["part_corpus"] = table["legs_corpus"] / max(len(stats), 1)
    table["valeur_actuelle"] = table["parametre"].map(settings)
    return table


def plot_sensitivity(table: pd.DataFrame):
    import matplotlib.pyplot as plt

    from .indicators import PALETTE, _style

    parameters = list(dict.fromkeys(table["parametre"]))
    ncols = 4
    nrows = int(np.ceil(len(parameters) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.4 * nrows), squeeze=False)
    fill, line = PALETTE[3]
    for ax, parameter in zip(axes.flat, parameters):
        sub = table[table["parametre"] == parameter]
        labels = ["désactivé" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:g}" for v in sub["valeur"]]
        x = np.arange(len(sub))
        ax.plot(x, sub["legs_corpus"], color=line, marker="o", lw=2)
        ax.fill_between(x, 0, sub["legs_corpus"], color=fill, alpha=0.5)
        current = sub["valeur_actuelle"].iloc[0]
        hits = [i for i, v in enumerate(sub["valeur"]) if v == current]
        for i in hits:
            ax.axvline(i, color="#6b7280", ls="--", lw=1)
            ax.annotate("YAML", (i, sub["legs_corpus"].iloc[i]), textcoords="offset points", xytext=(4, 6), fontsize=7, color="#6b7280")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=30)
        ax.set_ylim(bottom=0)
        ax.set_title(SWEEP_PARAMETERS.get(parameter, parameter), fontsize=9)
        ax.set_ylabel("legs dans le corpus")
        secondary = ax.secondary_yaxis("right", functions=(lambda v, n=sub["legs_voiture_perimetre"].iloc[0]: v / max(n, 1) * 100,
                                                           lambda p, n=sub["legs_voiture_perimetre"].iloc[0]: p * max(n, 1) / 100))
        secondary.set_ylabel("% des legs voiture du périmètre", fontsize=7)
        _style(ax)
    for ax in list(axes.flat)[len(parameters):]:
        ax.set_visible(False)
    fig.suptitle("Taille du corpus commun selon chaque filtre (les autres filtres à leur valeur du YAML ; tirets : valeur actuelle)", fontsize=10)
    fig.tight_layout()
    return fig


def segments_per_leg(config: dict[str, Any], corpus: pd.DataFrame) -> pd.DataFrame:
    """Pour les legs du corpus : segments candidats, segments retenus et état du dernier segment."""
    candidates = pd.read_csv(output_path(config, "data", "segment_candidates.csv"),
                             usecols=["trip_candidate_id", "leg_id", "finished_at", "kept"])
    candidates["finished_at"] = pd.to_datetime(candidates["finished_at"], utc=True, errors="coerce")
    per_leg = candidates.groupby("leg_id").agg(segments=("trip_candidate_id", "size"), segments_kept=("kept", "sum"))
    last = candidates.sort_values(["leg_id", "finished_at"]).groupby("leg_id").tail(1).set_index("leg_id")["kept"]
    legs = corpus.loc[corpus["corpus_status"].eq("in_corpus"), ["leg_id"]].set_index("leg_id")
    legs = legs.join(per_leg).join(last.rename("last_segment_kept"))
    legs[["segments", "segments_kept"]] = legs[["segments", "segments_kept"]].fillna(0).astype(int)
    legs["last_segment_kept"] = legs["last_segment_kept"].fillna(False).astype(bool)
    return legs.reset_index()


def plot_segments_per_leg(per_leg: pd.DataFrame):
    import matplotlib.pyplot as plt

    from .indicators import PALETTE, _style

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    classes = per_leg["segments_kept"].clip(upper=3).map({0: "0", 1: "1", 2: "2", 3: "3 et +"})
    counts = classes.value_counts().reindex(["0", "1", "2", "3 et +"]).fillna(0)
    axes[0].bar(counts.index, counts.values, color=PALETTE[3][0], edgecolor=PALETTE[3][1])
    for i, v in enumerate(counts.values):
        axes[0].text(i, v, f"{int(v)}\n({v / max(len(per_leg), 1):.0%})", ha="center", va="bottom", fontsize=8)
    axes[0].set_ylim(0, max(counts.max(), 1) * 1.25)
    axes[0].set_xlabel("segments retenus couvrant la leg")
    axes[0].set_ylabel("legs du corpus")
    axes[0].set_title("Legs du corpus selon le nombre de segments (temporel)", fontsize=9)
    share = per_leg["last_segment_kept"].mean() if len(per_leg) else np.nan
    axes[1].bar(["évaluables par les deux méthodes", "spatial seulement\n(dernier segment non retenu)"],
                [per_leg["last_segment_kept"].sum(), (~per_leg["last_segment_kept"]).sum()],
                color=[PALETTE[1][0], PALETTE[2][0]], edgecolor=[PALETTE[1][1], PALETTE[2][1]])
    axes[1].set_title(f"Dernier segment retenu : {share:.0%} des legs du corpus", fontsize=9)
    axes[1].set_ylabel("legs du corpus")
    for ax in axes:
        _style(ax)
    fig.suptitle("Couverture du corpus par les segments de la méthode temporelle", fontsize=10)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Entonnoir des waypoints (relit les rapports de l'étape 01)
# --------------------------------------------------------------------------- #


def waypoint_funnel(config: dict[str, Any], *, include_segments: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Nombre de waypoints (et de legs) restant après chaque étape de l'étape 01, jusqu'au corpus.

    Retourne ``(entonnoir, pertes_corpus)`` : l'entonnoir étape par étape, et les
    waypoints des legs écartées du corpus, par premier filtre non vérifié.
    Relit ``cleaning_report.json``, ``waypoint_leg_matching_summary.json`` et le
    corpus ; rien n'est recalculé (sauf le comptage des points ReLUT, une colonne lue).
    """
    def report(name: str) -> dict[str, Any]:
        path = output_path(config, "reports", name)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    cleaning = report("cleaning_report.json")
    matching = report("waypoint_leg_matching_summary.json")
    corpus = load_corpus(config)
    car_area = corpus["corpus_status"].isin(["in_corpus", "excluded"])
    in_corpus = corpus["corpus_status"].eq("in_corpus")
    status = matching.get("status_counts", {})

    rows = [
        ("waypoints bruts (CSV / Parquet)", cleaning.get("raw_rows"), None),
        ("après nettoyage (coordonnées, précision, doublons)", cleaning.get("clean_rows"), None),
        ("appariés à une leg (tous modes)", matching.get("matched_waypoints"), matching.get("matched_legs")),
        ("legs voiture", matching.get("car_waypoints"), matching.get("car_legs")),
        ("destination dans le périmètre", matching.get("study_area_car_waypoints_after_filter"), int(car_area.sum())),
        ("legs du corpus commun", int(corpus.loc[in_corpus, "corpus_n_points"].sum()), int(in_corpus.sum())),
    ]
    if include_segments:
        prepared = output_path(config, "data", "prepared_relut_points.parquet")
        if prepared.exists():
            import pyarrow.parquet as pq

            ids = set(corpus.loc[in_corpus, "leg_id"])
            parts = sorted(prepared.glob("*.parquet")) if prepared.is_dir() else [prepared]
            n_points = sum(int(pq.read_table(p, columns=["leg_id"])["leg_id"].to_pandas().isin(ids).sum()) for p in parts)
            rows.append(("dans les segments ReLUT du corpus (méthode temporelle)", n_points, None))
    funnel = pd.DataFrame(rows, columns=["etape", "waypoints", "legs"])
    first = funnel["waypoints"].iloc[0] or np.nan
    funnel["part_des_bruts"] = funnel["waypoints"] / first
    funnel["retires_a_cette_etape"] = (-funnel["waypoints"].diff()).astype("Int64")

    details = {
        "retirés : coordonnées manquantes ou hors bornes": cleaning.get("dropped_required_or_bounds"),
        "retirés : précision insuffisante (max_accuracy_m)": cleaning.get("dropped_accuracy"),
        "retirés : doublons": cleaning.get("dropped_duplicates"),
        "non appariés : hors de toute leg": status.get("outside_track"),
        "non appariés : legs qui se chevauchent": status.get("overlapping_tracks"),
        "non appariés : utilisateur sans legs": status.get("user_without_tracks"),
    }
    losses = (corpus[car_area & ~in_corpus].groupby("corpus_exclusion_reason")
              .agg(legs=("leg_id", "size"), waypoints=("corpus_n_points", "sum"))
              .sort_values("waypoints", ascending=False).reset_index()
              .rename(columns={"corpus_exclusion_reason": "premier_filtre_non_verifie"}))
    funnel.attrs["details"] = {k: v for k, v in details.items() if v is not None}
    return funnel, losses


def plot_waypoint_funnel(funnel: pd.DataFrame):
    import matplotlib.pyplot as plt

    from .indicators import PALETTE

    fig, ax = plt.subplots(figsize=(10, 0.55 * len(funnel) + 1.2))
    y = np.arange(len(funnel))[::-1]
    values = funnel["waypoints"].astype(float).fillna(0) / 1e6
    ax.barh(y, values, color=PALETTE[3][0], edgecolor=PALETTE[3][1])
    for yi, (v, share, legs) in zip(y, zip(values, funnel["part_des_bruts"], funnel["legs"])):
        text = f" {v:,.2f} M ({share:.0%})".replace(",", "'") + (f" — {int(legs):,} legs".replace(",", "'") if pd.notna(legs) else "")
        ax.text(v, yi, text, va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(funnel["etape"], fontsize=9)
    ax.set_xlabel("millions de waypoints")
    ax.set_xlim(0, values.max() * 1.45 if values.max() > 0 else 1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Waypoints restant après chaque étape (% des waypoints bruts)", fontsize=10)
    fig.tight_layout()
    return fig
