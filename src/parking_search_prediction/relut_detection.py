"""Étape 32 — détection temporelle de la recherche de stationnement (modèle ReLUT).

Regroupe ce qui était réparti entre ``relut_model.py``, ``metrics.py`` et
``viz.py`` :

1. **Modèle** : réseau ReLUT *ParkingSearchPrediction* (start2park, Frankfurt
   UAS, licence BSD-3, https://github.com/ReLUT/parking-search-prediction).
   Les poids sont lus directement dans le fichier ``.h5`` et le réseau (dense
   13 → 128 → 32 → 1, ReLU / sigmoïde) est évalué en numpy : TensorFlow n'est
   pas nécessaire. Variables, seuils « auto » et règle « une fois en recherche,
   toujours en recherche » sont repris à l'identique de ``predict.py`` amont.
2. **Synthèse par segment** : mêmes colonnes que ``metrics.summarize_predictions``,
   calculées de façon vectorisée.
3. **Sorties** : histogrammes des durées prédites (ensemble, sans et avec besoin
   de chercher selon le questionnaire), tableaux, cartes de traces.

Les segments sont traités par paquets (``temporal_detection.chunk_segments``) :
la mémoire ne dépend pas du nombre total de segments.
"""

from __future__ import annotations

import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import ensure_output_dirs, output_path, resolve_path
from .utils import haversine_m, write_json

# --------------------------------------------------------------------------- #
# Modèle ReLUT
# --------------------------------------------------------------------------- #

N_LAGS = 5
TRAINED_VARS = [
    "distToParkingSpot",
    "speed_kmh",
    *[f"speed_lag{i}" for i in range(1, N_LAGS + 1)],
    *[f"sampRate_lag{i}" for i in range(0, N_LAGS + 1)],
]
# Seuils « auto » de predict.py amont : choisis selon la médiane de l'intervalle
# d'échantillonnage du trajet (1, 5, 10, 15 s).
SAMP_RATE_VALUES = np.array([1.0, 5.0, 10.0, 15.0])
CUT_OFF_POINTS = np.array([0.527, 0.395, 0.341, 0.316])

METADATA_COLUMNS = [
    "leg_id", "storyline_id", "trip_id", "mode", "detected_mode", "effective_mode",
    "mode_niv1", "mode_niv2", "mode_mrmt", "leg_started_at", "leg_finished_at",
]


class ReLUTModel:
    """Réseau dense ReLUT évalué en numpy à partir des poids du fichier Keras ``.h5``."""

    def __init__(self, path: str | Path):
        import h5py

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Modèle ReLUT introuvable : {path}. Il est fourni dans models/ (ou "
                "https://github.com/ReLUT/parking-search-prediction, dossier model/)."
            )
        with h5py.File(path, "r") as handle:
            config = handle.attrs["model_config"]
            config = json.loads(config if isinstance(config, str) else config.decode("utf-8"))
            weights = handle["model_weights"]
            self.layers: list[tuple[np.ndarray, np.ndarray, str]] = []
            for layer in config["config"]["layers"]:
                if layer["class_name"] != "Dense":
                    continue  # Dropout : inactif en prédiction ; InputLayer : sans poids
                name = layer["config"]["name"]
                group = weights[name][name]
                kernel = np.asarray(group[next(k for k in group if k.startswith("kernel"))], dtype=np.float32)
                bias = np.asarray(group[next(k for k in group if k.startswith("bias"))], dtype=np.float32)
                self.layers.append((kernel, bias, layer["config"]["activation"]))
        if self.layers[0][0].shape[0] != len(TRAINED_VARS):
            raise ValueError(f"Le modèle attend {self.layers[0][0].shape[0]} variables, {len(TRAINED_VARS)} préparées.")

    def predict(self, features: np.ndarray, batch_size: int = 200_000) -> np.ndarray:
        out = np.empty(len(features), dtype=np.float32)
        for start in range(0, len(features), batch_size):
            x = np.asarray(features[start : start + batch_size], dtype=np.float32)
            for kernel, bias, activation in self.layers:
                x = x @ kernel + bias
                if activation == "relu":
                    x = np.maximum(x, 0.0)
                elif activation == "sigmoid":
                    x = 1.0 / (1.0 + np.exp(-x))
                elif activation != "linear":
                    raise ValueError(f"Activation non prise en charge : {activation}")
            out[start : start + batch_size] = x[:, 0]
        return out


def prepare_features(points: pd.DataFrame) -> pd.DataFrame:
    """Variables du modèle, comme ``prepare_data`` amont (une trajectoire = ``trip_candidate_id``).

    Différence volontaire : le remplissage « bfill » amont s'appliquait à toutes
    les colonnes (et pouvait recopier des métadonnées d'un segment à l'autre) ;
    ici il ne touche que les variables retardées, ce qui ne change pas les prédictions.
    """
    df = points.sort_values(["trip_candidate_id", "timestamp"], kind="mergesort").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    grouped = df.groupby("trip_candidate_id", sort=False)

    df["remainingTime"] = (grouped["timestamp"].transform("last") - df["timestamp"]).dt.total_seconds()
    df["distToParkingSpot"] = haversine_m(
        df["lat"], df["lon"], grouped["lat"].transform("last"), grouped["lon"].transform("last")
    )
    lagged = {f"speed_lag{i}": grouped["speed_kmh"].shift(i) for i in range(1, N_LAGS + 1)}
    samp0 = grouped["timestamp"].diff().dt.total_seconds().fillna(0)
    lagged["sampRate_lag0"] = samp0
    samp_grouped = samp0.groupby(df["trip_candidate_id"], sort=False)
    lagged.update({f"sampRate_lag{i}": samp_grouped.shift(i) for i in range(1, N_LAGS + 1)})
    df = pd.concat([df, pd.DataFrame(lagged).bfill()], axis=1)
    return df


def predict_segments(
    points: pd.DataFrame,
    model: ReLUTModel,
    *,
    p_search: float | str = "auto",
    max_search_duration_minutes: float | None = 15,
) -> pd.DataFrame:
    """Probabilité et label ``searching`` / ``driving`` pour chaque point."""
    if not (p_search == "auto" or (isinstance(p_search, (int, float)) and 0 <= p_search <= 1)):
        raise ValueError("p_search doit valoir 'auto' ou être compris entre 0 et 1.")
    df = prepare_features(points)
    df["y_hat_p"] = model.predict(df[TRAINED_VARS].to_numpy())

    if p_search == "auto":
        median = df.groupby("trip_candidate_id", sort=False)["sampRate_lag0"].transform("median")
        df["sampRateMedian"] = median
        closest = np.abs(median.to_numpy()[:, None] - SAMP_RATE_VALUES[None, :]).argmin(axis=1)
        df["probability_threshold"] = CUT_OFF_POINTS[closest]
    else:
        df["probability_threshold"] = float(p_search)
    searching = df["y_hat_p"].to_numpy() > df["probability_threshold"].to_numpy()

    # Une fois la recherche commencée, tous les points suivants du trajet sont en recherche.
    searching = pd.Series(searching, index=df.index).groupby(df["trip_candidate_id"], sort=False).cummax()
    if max_search_duration_minutes not in (None, False):
        searching &= df["remainingTime"] <= float(max_search_duration_minutes) * 60
    df["y_hat_labels"] = np.where(searching, "searching", "driving")
    df = df.drop(columns=[c for c in TRAINED_VARS if c.startswith(("speed_lag", "sampRate_lag")) and c != "sampRate_lag0"])
    return df.rename(columns={"sampRate_lag0": "samplingRate"})


# --------------------------------------------------------------------------- #
# Synthèse par segment (colonnes de metrics.summarize_predictions)
# --------------------------------------------------------------------------- #


def summarize_segments(predictions: pd.DataFrame) -> pd.DataFrame:
    df = predictions.sort_values(["trip_candidate_id", "timestamp"], kind="mergesort").reset_index(drop=True)
    grouped = df.groupby("trip_candidate_id", sort=True)
    step = haversine_m(grouped["lat"].shift(), grouped["lon"].shift(), df["lat"], df["lon"])
    df["route_step_m"] = pd.Series(step, index=df.index).fillna(0)
    df["route_distance_m"] = df.groupby("trip_candidate_id", sort=False)["route_step_m"].cumsum()
    df["_interval"] = grouped["timestamp"].diff().dt.total_seconds()
    df["_searching"] = df["y_hat_labels"].eq("searching")
    grouped = df.groupby("trip_candidate_id", sort=True)

    summary = pd.DataFrame(
        {
            "source_user_id": grouped["source_user_id"].first(),
            "started_at": grouped["timestamp"].first(),
            "finished_at": grouped["timestamp"].last(),
            "point_count": grouped.size().astype(int),
        }
    )
    summary["duration_seconds"] = (summary["finished_at"] - summary["started_at"]).dt.total_seconds()
    summary["duration_minutes"] = summary["duration_seconds"] / 60
    summary["route_distance_m"] = grouped["route_step_m"].sum()
    summary["median_interval_seconds"] = grouped["_interval"].median()
    summary["mean_speed_kmh"] = grouped["speed_kmh"].mean()
    summary["max_probability"] = grouped["y_hat_p"].max().astype(float)
    if "probability_threshold" in df.columns:
        summary["probability_threshold"] = grouped["probability_threshold"].first().astype(float)
    summary["mean_probability"] = grouped["y_hat_p"].mean().astype(float)
    summary["searching_points"] = grouped["_searching"].sum().astype(int)
    summary["searching_share"] = summary["searching_points"] / summary["point_count"]
    for column in METADATA_COLUMNS:
        if column in df.columns:
            summary[column] = grouped[column].first()

    first = df[df["_searching"]].groupby("trip_candidate_id", sort=True).head(1).set_index("trip_candidate_id")
    summary["search_predicted"] = summary.index.isin(first.index)
    summary["search_started_at"] = first["timestamp"].reindex(summary.index)
    summary["search_duration_seconds"] = (
        (summary["finished_at"] - summary["search_started_at"]).dt.total_seconds().fillna(0.0)
    )
    summary["search_start_remaining_distance_m"] = first["distToParkingSpot"].reindex(summary.index).astype(float)
    summary["search_start_route_distance_m"] = first["route_distance_m"].reindex(summary.index).astype(float)
    summary["search_start_probability"] = first["y_hat_p"].reindex(summary.index).astype(float)
    return summary.reset_index()


# --------------------------------------------------------------------------- #
# Sélection des segments (run complet ou limité à N legs)
# --------------------------------------------------------------------------- #


def select_segments(points: pd.DataFrame, max_legs: int | None, random_seed: int = 42) -> pd.DataFrame:
    """Tous les segments, ou ceux de ``max_legs`` legs tirées au hasard (graine fixe)."""
    if max_legs is None:
        return points
    key = "leg_id" if "leg_id" in points.columns and points["leg_id"].notna().all() else "trip_candidate_id"
    legs = np.sort(points[key].dropna().unique())
    if max_legs >= len(legs):
        return points
    chosen = np.random.default_rng(random_seed).choice(legs, size=int(max_legs), replace=False)
    return points[points[key].isin(set(chosen))]


# --------------------------------------------------------------------------- #
# Groupes « besoin de chercher » (questionnaire PL23)
# --------------------------------------------------------------------------- #


def _settings(config: dict[str, Any]) -> dict[str, Any]:
    from .pl23_survey import BESOIN, PAS_BESOIN

    model_cfg = config.get("model", {}) or {}
    cfg = config.get("temporal_detection", {}) or {}
    return {
        "model_path": resolve_path(config, model_cfg.get("path", "models/ParkingSearchPrediction.h5")),
        "p_search": model_cfg.get("p_search", "auto"),
        "max_search_duration_minutes": model_cfg.get("max_search_duration_minutes", 15),
        "max_legs": cfg.get("max_legs"),
        "random_seed": int(cfg.get("random_seed", 42)),
        "chunk_segments": int(cfg.get("chunk_segments", 2000)),
        "use_survey": bool(cfg.get("use_survey", True)),
        "parking_variable": cfg.get("parking_variable", "space_home"),
        "groups": {
            "Pas besoin de chercher": list(cfg.get("no_cruise_categories", [PAS_BESOIN])),
            "Besoin de chercher": list(cfg.get("cruise_categories", [BESOIN])),
        },
        "n_longest_map": int(cfg.get("n_longest_map", 20)),
        "n_sample_map": int(cfg.get("n_sample_map", 20)),
        "n_generic_sample": int(cfg.get("n_generic_sample", 12)),
        "write_leg_table": bool(cfg.get("write_leg_table", True)),
    }


def add_parking_groups(
    summary: pd.DataFrame, config: dict[str, Any], settings: dict[str, Any], users: pd.DataFrame | None = None
) -> pd.DataFrame:
    from .pl23_survey import parking_by_user

    if users is None:
        users = parking_by_user(config, summary["source_user_id"].dropna().unique())
    summary = summary.merge(users[["source_user_id", "user_id_fors", "space_home", "space_work"]], on="source_user_id", how="left")
    variable = summary[settings["parking_variable"]]
    summary["cruise_group"] = pd.NA
    for group, categories in settings["groups"].items():
        summary.loc[variable.isin(categories), "cruise_group"] = group
    return summary


def duration_stats(
    summary: pd.DataFrame, by: str | None, labels: list[str] | None = None, removed: pd.DataFrame | None = None
) -> pd.DataFrame:
    """n_segments, temps moyen et médian (tous segments, puis segments avec recherche).

    Avec ``removed`` (segments écartés comme boucles), ajoute
    ``n_segments_ecartes_boucles`` pour le même groupe.
    """
    if by is not None and labels is None:
        labels = list(dict.fromkeys(
            list(summary[by].dropna().unique()) + (list(removed[by].dropna().unique()) if removed is not None and by in removed else [])
        ))
    groups = [("Ensemble", summary)] if by is None else [(label, summary[summary[by].eq(label)]) for label in labels]
    rows = []
    for label, sub in groups:
        minutes = sub["search_duration_seconds"] / 60
        searched = minutes[sub["search_predicted"]]
        rows.append({
            "groupe": label,
            "n_segments": int(len(sub)),
            "temps_moyen_min": minutes.mean() if len(sub) else np.nan,
            "temps_median_min": minutes.median() if len(sub) else np.nan,
            "n_segments_avec_recherche": int(len(searched)),
            "part_avec_recherche": len(searched) / len(sub) if len(sub) else np.nan,
            "temps_moyen_min_si_recherche": searched.mean() if len(searched) else np.nan,
            "temps_median_min_si_recherche": searched.median() if len(searched) else np.nan,
        })
        if removed is not None:
            rows[-1]["n_segments_ecartes_boucles"] = int(
                len(removed) if by is None else (removed[by].eq(label).sum() if by in removed else 0)
            )
    return pd.DataFrame(rows)


def add_search_distance(summary: pd.DataFrame) -> pd.DataFrame:
    """Distance de recherche prédite : somme des distances entre waypoints consécutifs depuis la bascule (0 sans recherche)."""
    summary = summary.copy()
    summary["search_distance_m"] = np.where(
        summary["search_predicted"], summary["route_distance_m"] - summary["search_start_route_distance_m"], 0.0
    )
    return summary


def distance_stats(summary: pd.DataFrame, by: str | None, labels: list[str] | None = None) -> pd.DataFrame:
    """n_segments, distance moyenne et médiane (tous segments, puis segments avec recherche), en mètres."""
    if by is not None and labels is None:
        labels = list(summary[by].dropna().unique())
    groups = [("Ensemble", summary)] if by is None else [(label, summary[summary[by].eq(label)]) for label in labels]
    rows = []
    for label, sub in groups:
        meters = sub["search_distance_m"]
        searched = meters[sub["search_predicted"]]
        rows.append({
            "groupe": label,
            "n_segments": int(len(sub)),
            "distance_moyenne_m": meters.mean() if len(sub) else np.nan,
            "distance_mediane_m": meters.median() if len(sub) else np.nan,
            "n_segments_avec_recherche": int(len(searched)),
            "distance_moyenne_m_si_recherche": searched.mean() if len(searched) else np.nan,
            "distance_mediane_m_si_recherche": searched.median() if len(searched) else np.nan,
            "distance_q90_m_si_recherche": searched.quantile(0.9) if len(searched) else np.nan,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Figures et cartes
# --------------------------------------------------------------------------- #


def plot_distance_histograms(panels: list[tuple[str, pd.DataFrame]], max_m: float, bins: int = 30):
    """Distance de recherche prédite (entre waypoints, depuis la bascule), un panneau par groupe."""
    import matplotlib.pyplot as plt

    edges = np.linspace(0, max_m, bins + 1)
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4), sharey=True, squeeze=False)
    colors = ["#3b6ea5", "#4c9a6a", "#c0504d"]
    for ax, (title, sub), color in zip(axes[0], panels, colors):
        meters = sub["search_distance_m"]
        if len(sub):
            ax.hist(meters.clip(upper=max_m), bins=edges, color=color,
                    weights=np.full(len(meters), 100 / len(meters)), edgecolor="white", linewidth=0.3)
            ax.axvline(meters.mean(), color="black", lw=1, ls="--", label=f"moyenne {meters.mean():.0f} m")
            ax.axvline(meters.median(), color="black", lw=1, ls=":", label=f"médiane {meters.median():.0f} m")
            ax.legend(frameon=False, fontsize=8)
        ax.set_title(f"{title}\n(n = {len(sub):,} segments)".replace(",", "'"), fontsize=10)
        ax.set_xlabel("Distance de recherche prédite (m)")
    axes[0][0].set_ylabel("% des segments")
    fig.suptitle("Distance de recherche prédite par ReLUT : somme des distances entre waypoints depuis la bascule\n"
                 f"(0 = pas de recherche détectée ; au-delà de {max_m:.0f} m, regroupé dans la dernière classe)", fontsize=11)
    fig.tight_layout()
    return fig


def plot_duration_histograms(summary: pd.DataFrame, panels: list[tuple[str, pd.DataFrame]], max_minutes: float):
    import matplotlib.pyplot as plt

    bins = np.arange(0, max_minutes + 0.5, 0.5)
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4), sharey=True, squeeze=False)
    colors = ["#3b6ea5", "#4c9a6a", "#c0504d"]
    for ax, (title, sub), color in zip(axes[0], panels, colors):
        minutes = sub["search_duration_seconds"] / 60
        if len(sub):
            ax.hist(minutes.clip(upper=max_minutes), bins=bins, color=color,
                    weights=np.full(len(minutes), 100 / len(minutes)), edgecolor="white", linewidth=0.3)
            ax.axvline(minutes.mean(), color="black", lw=1, ls="--", label=f"moyenne {minutes.mean():.2f} min")
            ax.axvline(minutes.median(), color="black", lw=1, ls=":", label=f"médiane {minutes.median():.2f} min")
            ax.legend(frameon=False, fontsize=8)
        ax.set_title(f"{title}\n(n = {len(sub):,} segments)".replace(",", "'"), fontsize=10)
        ax.set_xlabel("Durée de recherche prédite (min)")
    axes[0][0].set_ylabel("% des segments")
    fig.suptitle("Durée de recherche de stationnement prédite par ReLUT (0 = pas de recherche détectée)")
    fig.tight_layout()
    return fig


def segment_threshold(group: pd.DataFrame, p_search: float | str) -> tuple[float, str]:
    """Seuil de probabilité appliqué au segment, et son libellé pour la légende."""
    if p_search != "auto":
        return float(p_search), f"seuil p = {float(p_search):g} (fixe)"
    median = float(group["sampRateMedian"].iloc[0])
    index = int(np.abs(SAMP_RATE_VALUES - median).argmin())
    return float(CUT_OFF_POINTS[index]), f"seuil p = {CUT_OFF_POINTS[index]:g} (auto, échantillonnage médian {median:g} s)"


PASTEL_RED = "#f4a6a6"
PASTEL_RED_BACKGROUND = "#fde4e4"


def plot_probability_curves(
    predictions: pd.DataFrame,
    segments: pd.DataFrame,
    p_search: float | str,
    max_search_duration_minutes: float | None,
    ncols: int = 3,
):
    """Une courbe p(recherche) par segment ; partie en cruising surlignée en rouge pastel."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    ids = segments["trip_candidate_id"].tolist()
    n = len(ids)
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 2.9 * nrows), squeeze=False)
    durations = segments.set_index("trip_candidate_id")["search_duration_seconds"]
    for ax, trip_id in zip(axes.flat, ids):
        group = predictions[predictions["trip_candidate_id"] == trip_id].sort_values("timestamp")
        minutes = ((group["timestamp"] - group["timestamp"].iloc[0]).dt.total_seconds() / 60).to_numpy()
        prob = group["y_hat_p"].to_numpy(dtype=float)
        searching = group["y_hat_labels"].eq("searching").to_numpy()
        threshold, threshold_label = segment_threshold(group, p_search)

        ax.plot(minutes, prob, color="#374151", lw=1.2)
        handles = [Line2D([], [], color="#374151", lw=1.2, label="p(recherche)")]
        if searching.any():
            start = int(np.argmax(searching))
            ax.axvspan(minutes[start], minutes[-1], color=PASTEL_RED_BACKGROUND, zorder=0)
            ax.plot(minutes[start:], prob[start:], color=PASTEL_RED, lw=2.6)
            ax.axvline(minutes[start], color=PASTEL_RED, lw=1, ls="--")
            handles.append(Patch(facecolor=PASTEL_RED_BACKGROUND, edgecolor=PASTEL_RED,
                                 label=f"cruising ({durations.get(trip_id, 0) / 60:.1f} min)"))
        ax.axhline(threshold, color="#6b7280", lw=0.9, ls=":")
        handles.append(Line2D([], [], color="#6b7280", lw=0.9, ls=":", label=threshold_label))
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(minutes[0], max(minutes[-1], 1e-3))
        ax.set_title(str(trip_id), loc="left", fontsize=8)
        ax.set_xlabel("minutes depuis le début du segment", fontsize=8)
        ax.set_ylabel("p(recherche)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(handles=handles, fontsize=7, loc="upper left", frameon=True, framealpha=0.85)
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    cap = f" ; recherche limitée aux {max_search_duration_minutes:g} dernières minutes" if max_search_duration_minutes else ""
    fig.suptitle(f"Probabilité de recherche ReLUT — échantillon de {n} segments (rouge pastel : cruising prédit{cap})", fontsize=11)
    fig.tight_layout()
    return fig


SWISSTOPO_GREY = "https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.pixelkarte-grau/default/current/3857/{z}/{x}/{y}.jpeg"
COLORS = {"driving": "#2f80ed", "searching": "#d62728"}


def write_trace_map(predictions: pd.DataFrame, segments: pd.DataFrame, path: Path, title: str) -> Path | None:
    """Carte folium : traces conduite (bleu) / recherche (rouge), début de recherche marqué."""
    import folium

    if segments.empty:
        return None
    points = predictions[predictions["trip_candidate_id"].isin(segments["trip_candidate_id"])]
    fmap = folium.Map(location=[points["lat"].median(), points["lon"].median()], zoom_start=12, tiles=None)
    from .map_style import add_basemaps

    add_basemaps(fmap)
    fmap.get_root().html.add_child(folium.Element(
        f'<div style="position:fixed;top:10px;left:50px;z-index:9999;background:white;padding:6px 10px;'
        f'border-radius:4px;font:14px sans-serif;box-shadow:0 1px 4px #0003">{title}</div>'
    ))
    for row in segments.itertuples():
        group = points[points["trip_candidate_id"] == row.trip_candidate_id].sort_values("timestamp")
        label = f"{row.trip_candidate_id} — recherche {row.search_duration_seconds / 60:.1f} min"
        if hasattr(row, "cruise_group") and pd.notna(row.cruise_group):
            label += f" — {row.cruise_group}"
        feature = folium.FeatureGroup(name=label, show=True)
        for state, color in COLORS.items():
            sub = group[group["y_hat_labels"] == state]
            if len(sub) >= 2:
                folium.PolyLine(sub[["lat", "lon"]].values.tolist(), color=color,
                                weight=4 if state == "searching" else 3, opacity=0.85, tooltip=label).add_to(feature)
        start, end = group.iloc[0], group.iloc[-1]
        folium.CircleMarker([start["lat"], start["lon"]], radius=4, color="#111827", fill=True, tooltip="début").add_to(feature)
        folium.CircleMarker([end["lat"], end["lon"]], radius=5, color="#111827", fill=True,
                            tooltip="fin / stationnement supposé").add_to(feature)
        search = group[group["y_hat_labels"] == "searching"]
        if not search.empty:
            folium.Marker([search.iloc[0]["lat"], search.iloc[0]["lon"]], tooltip=f"début recherche — {label}",
                          icon=folium.Icon(color="red", icon="flag")).add_to(feature)
        feature.add_to(fmap)
    fmap.get_root().html.add_child(folium.Element(TRACE_MAP_LEGEND))
    folium.LayerControl(collapsed=True).add_to(fmap)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(path)
    return path


# Légende des cartes de traces : couleurs des tronçons et marqueurs
_FLAG = ('<span style="display:inline-block;width:18px;height:18px;background:#d63e2a;border-radius:50% 50% 50% 0;'
         'transform:rotate(-45deg);text-align:center;vertical-align:middle;margin-right:6px">'
         '<i class="glyphicon glyphicon-flag" style="color:#fff;font-size:9px;line-height:18px;transform:rotate(45deg)"></i></span>')
TRACE_MAP_LEGEND = (
    '<div style="position:fixed;bottom:24px;left:12px;z-index:9999;background:#ffffffee;padding:8px 10px;'
    'border-radius:6px;font:12px sans-serif;box-shadow:0 1px 4px #0003;line-height:1.9">'
    f'<div><span style="display:inline-block;width:26px;height:3px;background:{COLORS["driving"]};vertical-align:middle;margin-right:6px"></span>conduite</div>'
    f'<div><span style="display:inline-block;width:26px;height:4px;background:{COLORS["searching"]};vertical-align:middle;margin-right:6px"></span>recherche de stationnement (prédite)</div>'
    f'<div>{_FLAG}début de la recherche (bascule)</div>'
    '<div><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#111827;margin:0 11px 0 5px;vertical-align:middle"></span>début du segment</div>'
    '<div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#111827;margin:0 10px 0 4px;vertical-align:middle"></span>fin du segment (stationnement supposé)</div>'
    '</div>'
)


# --------------------------------------------------------------------------- #
# Table au niveau de la leg (collection legs/, voir leg_collection.py)
# --------------------------------------------------------------------------- #


def _unique_from_parts(path: Path, column: str, filters: dict[str, Iterable] | None = None) -> set[str] | None:
    """Valeurs distinctes d'une colonne d'un Parquet (fichier ou dossier de tranches), lu partie par partie."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    if not path.exists():
        return None
    parts = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    values: set[str] = set()
    columns = [column, *(filters or {})]
    for part in parts:
        table = pq.read_table(part, columns=columns)
        for name, allowed in (filters or {}).items():
            table = table.filter(pc.is_in(table[name], value_set=pa_array(list(allowed))))
        values.update(v for v in table[column].unique().to_pylist() if v is not None)
    return values


def pa_array(values: list[Any]):
    import pyarrow as pa

    return pa.array(values)


def build_relut_by_leg(
    config: dict[str, Any],
    summary: pd.DataFrame,
    *,
    sampled_legs: set[str] | None,
    settings: dict[str, Any],
    run_id: str,
) -> pd.DataFrame:
    """Une ligne par leg (toutes les legs) : statut ReLUT et résultats du **dernier** segment.

    Pas de repli sur un segment antérieur : si le dernier segment de la leg n'est
    pas prédit, la leg est non évaluée et la raison est donnée par ``relut_status``.
    Règle commune de la collection : 0 = évalué sans recherche, NaN = non évalué.
    """
    from .leg_collection import table_path

    car_modes = set(config["leg_matching"].get("car_modes", ["Mode::Car", "Mode::Ecar"]))
    legs_path = table_path(config, "legs_enriched")
    if legs_path.exists():
        legs = pd.read_parquet(legs_path, columns=["leg_id", "effective_mode"])
    else:
        # declic (pas d'enrichissement) : legs présentes dans la table d'appariement.
        # Les legs sans aucun waypoint apparié n'y figurent pas.
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        assignments = output_path(config, "data", "waypoint_leg_assignments.parquet")
        parts = sorted(assignments.glob("*.parquet")) if assignments.is_dir() else [assignments]
        frames = []
        for part in parts:
            chunk = pq.read_table(part, columns=["leg_id", "effective_mode", "leg_match_status"])
            chunk = chunk.filter(pc.equal(chunk["leg_match_status"], "matched")).to_pandas()
            frames.append(chunk[["leg_id", "effective_mode"]].drop_duplicates("leg_id"))
        legs = pd.concat(frames, ignore_index=True)
    legs = legs.drop_duplicates("leg_id").reset_index(drop=True)

    prepared = _unique_from_parts(output_path(config, "data", "prepared_relut_points.parquet"), "trip_candidate_id") or set()

    candidates = pd.read_csv(
        output_path(config, "data", "segment_candidates.csv"),
        usecols=["trip_candidate_id", "leg_id", "candidate_sequence", "finished_at", "kept",
                 "loop_start_end_distance_m", "is_loop_segment", "distance_m"],
        dtype={"trip_candidate_id": "string", "leg_id": "string"},
    )
    candidates["finished_at"] = pd.to_datetime(candidates["finished_at"], utc=True, errors="coerce")
    counts = candidates.groupby("leg_id").size().rename("relut_segments_in_leg")
    last = (
        candidates.sort_values(["leg_id", "finished_at", "candidate_sequence"], kind="mergesort")
        .groupby("leg_id", sort=False).tail(1).set_index("leg_id")
    )
    validation_path = output_path(config, "reports", "relut_validation_by_segment.csv")
    ready = (
        pd.read_csv(validation_path, usecols=["trip_candidate_id", "relut_ready"]).set_index("trip_candidate_id")["relut_ready"]
        if validation_path.exists() else pd.Series(dtype=bool)
    )

    table = legs.join(counts, on="leg_id").join(
        last[["trip_candidate_id", "kept", "loop_start_end_distance_m", "distance_m"]].rename(columns={
            "trip_candidate_id": "relut_segment_id", "kept": "relut_segment_kept",
            "loop_start_end_distance_m": "relut_segment_loop_distance_m",
            "distance_m": "relut_segment_candidate_distance_m",
        }),
        on="leg_id",
    )
    table["relut_segments_in_leg"] = table["relut_segments_in_leg"].fillna(0).astype(int)
    table["relut_segment_kept"] = table["relut_segment_kept"].astype("boolean")
    table["relut_segment_relut_ready"] = table["relut_segment_id"].map(ready).astype("boolean")

    # Statut, par ordre de priorité : d'abord celui du corpus commun (étape 01)
    from .leg_corpus import load_corpus

    corpus = load_corpus(config).set_index("leg_id")["corpus_status"]
    corpus_status = table["leg_id"].map(corpus)
    status = pd.Series(pd.NA, index=table.index, dtype="object")
    for value in ("not_car_leg", "no_car_waypoints", "outside_study_area"):
        status[corpus_status.eq(value)] = value
    status[status.isna() & corpus_status.eq("excluded")] = "excluded_by_corpus"
    status[status.isna() & corpus_status.isna()] = "unknown"
    status[status.isna() & table["relut_segment_id"].isna()] = "no_segment"
    status[status.isna() & table["relut_segment_kept"].eq(False).fillna(False)] = "last_segment_rejected"
    status[status.isna() & ~table["relut_segment_id"].isin(prepared)] = "last_segment_loop"   # boucles retirées à l'étape 01
    if sampled_legs is not None:
        status[status.isna() & ~table["leg_id"].isin(sampled_legs)] = "not_sampled"
    predicted = summary.set_index("trip_candidate_id")
    evaluated = status.isna() & table["relut_segment_id"].isin(predicted.index)
    status[evaluated] = "evaluated"
    status[status.isna()] = "unknown"
    table["relut_status"] = status

    # Résultats du dernier segment (NaN si non évalué)
    s = predicted.reindex(table["relut_segment_id"].where(evaluated)).reset_index(drop=True)
    detected = s["search_predicted"].astype("boolean")
    table["relut_search_detected"] = detected.where(evaluated, pd.NA)
    with_search = detected.fillna(False).to_numpy(dtype=bool)
    table["relut_search_duration_s"] = s["search_duration_seconds"].where(evaluated)
    table["relut_search_distance_m"] = np.where(
        with_search, s["route_distance_m"] - s["search_start_route_distance_m"], 0.0
    )
    table["relut_straight_line_distance_m"] = np.where(with_search, s["search_start_remaining_distance_m"], 0.0)
    for column in ("relut_search_distance_m", "relut_straight_line_distance_m"):
        table[column] = table[column].where(evaluated)
    table["relut_search_started_at"] = pd.to_datetime(s["search_started_at"], utc=True).where(evaluated)
    table["relut_segment_duration_s"] = s["duration_seconds"].where(evaluated)
    table["relut_segment_distance_m"] = s["route_distance_m"].where(evaluated)
    table["relut_max_probability"] = s["max_probability"].where(evaluated)
    table["relut_threshold"] = (s["probability_threshold"] if "probability_threshold" in s else pd.Series(np.nan, index=s.index)).where(evaluated)
    table["relut_run_id"] = run_id

    order = [
        "leg_id", "relut_status", "relut_search_detected", "relut_search_duration_s", "relut_search_distance_m",
        "relut_straight_line_distance_m", "relut_search_started_at", "relut_segment_id", "relut_segments_in_leg",
        "relut_segment_kept", "relut_segment_duration_s", "relut_segment_distance_m", "relut_segment_loop_distance_m",
        "relut_segment_relut_ready", "relut_max_probability", "relut_threshold", "relut_run_id",
    ]
    table["relut_segment_id"] = table["relut_segment_id"].astype("string")
    return table[order]


# --------------------------------------------------------------------------- #
# Point d'entrée
# --------------------------------------------------------------------------- #


def run_temporal_detection(config: dict[str, Any]) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    started = time.perf_counter()
    ensure_output_dirs(config)
    settings = _settings(config)

    points = pd.read_parquet(output_path(config, "data", "prepared_relut_points.parquet"))
    # Corpus commun (étape 01) : seules les legs « in_corpus » sont traitées, comme par la méthode spatiale
    from .leg_corpus import corpus_leg_ids

    corpus_ids = corpus_leg_ids(config)
    segments_before_corpus = int(points["trip_candidate_id"].nunique())
    points = points[points["leg_id"].isin(corpus_ids)]
    print(f"[relut] corpus commun : {len(corpus_ids)} legs — {points['trip_candidate_id'].nunique()} segments "
          f"sur {segments_before_corpus} préparés", flush=True)
    all_segments = int(points["trip_candidate_id"].nunique())
    legs_available = int(points["leg_id"].nunique()) if "leg_id" in points else None
    points = select_segments(points, settings["max_legs"], settings["random_seed"])
    # Run limité seulement si le tirage a réellement réduit le nombre de legs
    is_sample = settings["max_legs"] is not None and legs_available is not None and points["leg_id"].nunique() < legs_available
    sampled_legs = set(points["leg_id"].dropna()) if is_sample else None
    removed = None
    loop_info = {"note": "filtres de legs appliqués en amont par le corpus commun (legs/corpus_by_leg.parquet)",
                 "corpus_legs": len(corpus_ids), "segments_before_corpus": segments_before_corpus}
    segment_ids = np.sort(points["trip_candidate_id"].unique())
    n_legs = int(points["leg_id"].nunique()) if "leg_id" in points else None
    scope = "toutes les legs" if settings["max_legs"] is None else f"{settings['max_legs']} legs tirées au hasard (graine {settings['random_seed']})"
    fmt = lambda n: f"{n:,}".replace(",", "'")  # noqa: E731
    print(f"[relut] {fmt(len(segment_ids))} segments sur {fmt(all_segments)} ({scope}) — {fmt(len(points))} points", flush=True)

    model = ReLUTModel(settings["model_path"])
    predictions_dir = output_path(config, "data", "relut_predictions.parquet")
    if predictions_dir.is_dir():
        shutil.rmtree(predictions_dir)
    elif predictions_dir.exists():
        predictions_dir.unlink()
    predictions_dir.mkdir(parents=True)

    summaries, searching_points, total_points = [], 0, 0
    points = points.set_index("trip_candidate_id", drop=False).sort_index()
    for index, start in enumerate(range(0, len(segment_ids), settings["chunk_segments"])):
        chunk_ids = segment_ids[start : start + settings["chunk_segments"]]
        chunk = predict_segments(
            points.loc[chunk_ids].reset_index(drop=True),
            model,
            p_search=settings["p_search"],
            max_search_duration_minutes=settings["max_search_duration_minutes"],
        )
        chunk.to_parquet(predictions_dir / f"part-{index:04d}.parquet", index=False)
        summaries.append(summarize_segments(chunk))
        searching_points += int(chunk["y_hat_labels"].eq("searching").sum())
        total_points += len(chunk)
        done = min(start + settings["chunk_segments"], len(segment_ids))
        print(f"[relut] {fmt(done)}/{fmt(len(segment_ids))} segments — {time.perf_counter() - started:.0f} s", flush=True)
    del points

    summary = pd.concat(summaries, ignore_index=True)
    parking_available = False
    if settings["use_survey"] and (config["leg_matching"].get("user_id_map") or {}).get("file"):
        from .pl23_survey import parking_by_user

        all_users = set(summary["source_user_id"].dropna()) | (set(removed["source_user_id"].dropna()) if removed is not None else set())
        users = parking_by_user(config, all_users)
        summary = add_parking_groups(summary, config, settings, users)
        if removed is not None:
            removed = add_parking_groups(removed, config, settings, users)
        parking_available = True
    if removed is not None:
        removed.to_csv(output_path(config, "reports", "32_loop_filtered_segments.csv"), index=False)
    summary.to_csv(output_path(config, "reports", "prediction_summary_by_segment.csv"), index=False)

    # Tableaux
    group_labels = list(settings["groups"])
    stats = duration_stats(summary, None, removed=removed)
    if parking_available:
        stats = pd.concat([stats, duration_stats(summary, "cruise_group", group_labels, removed=removed)], ignore_index=True)
        missing_label = "non renseigné / absent du questionnaire"
        with_category = lambda df: df.assign(categorie=df[settings["parking_variable"]].fillna(missing_label))  # noqa: E731
        by_category = duration_stats(
            with_category(summary), "categorie", removed=with_category(removed) if removed is not None else None
        )
        by_category.insert(0, "variable", settings["parking_variable"])
        by_category.to_csv(output_path(config, "reports", "32_search_duration_by_parking_category.csv"), index=False)
    else:
        by_category = None
    stats.to_csv(output_path(config, "reports", "32_search_duration_stats.csv"), index=False)
    summary = add_search_distance(summary)
    distance_table = distance_stats(summary, None)
    if parking_available:
        distance_table = pd.concat([distance_table, distance_stats(summary, "cruise_group", group_labels)], ignore_index=True)
    distance_table.to_csv(output_path(config, "reports", "32_search_distance_stats.csv"), index=False)

    # Histogrammes
    panels = [("Ensemble des segments", summary)]
    if parking_available:
        panels += [(f"{label}\n({settings['parking_variable']})", summary[summary["cruise_group"].eq(label)]) for label in group_labels]
    max_minutes = float(settings["max_search_duration_minutes"] or max(summary["search_duration_seconds"].max() / 60, 1))
    fig = plot_duration_histograms(summary, panels, max_minutes)
    histogram_path = output_path(config, "figures", "32_search_duration_histograms.png")
    fig.savefig(histogram_path, dpi=150)
    plt.close(fig)
    searched_m = summary.loc[summary["search_predicted"], "search_distance_m"]
    max_m = float(searched_m.quantile(0.99)) if len(searched_m) else 1000.0
    fig = plot_distance_histograms([(title, add_search_distance(sub) if "search_distance_m" not in sub else sub)
                                    for title, sub in panels], max(max_m, 1.0))
    distance_histogram_path = output_path(config, "figures", "32_search_distance_histograms.png")
    fig.savefig(distance_histogram_path, dpi=150)
    plt.close(fig)

    # Cartes
    rng_seed = settings["random_seed"]
    longest = summary[summary["search_predicted"]].nlargest(settings["n_longest_map"], "search_duration_seconds")
    map_selections = {"longest": (longest, f"{len(longest)} recherches prédites les plus longues")}
    if parking_available:
        for key, label in (("need_cruise", "Besoin de chercher"), ("no_cruise", "Pas besoin de chercher")):
            pool = summary[summary["cruise_group"].eq(label)]
            sample = pool.sample(min(settings["n_sample_map"], len(pool)), random_state=rng_seed) if len(pool) else pool
            map_selections[key] = (sample, f"Échantillon de {len(sample)} segments — {label} ({settings['parking_variable']})")
    generic = summary.sample(min(settings["n_generic_sample"], len(summary)), random_state=rng_seed).sort_values("trip_candidate_id")
    map_selections["generic_sample"] = (generic, f"Échantillon générique de {len(generic)} segments (courbes de probabilité)")
    wanted = pd.concat([selection for selection, _ in map_selections.values()])["trip_candidate_id"].unique().tolist()
    map_points = pd.read_parquet(predictions_dir, filters=[("trip_candidate_id", "in", wanted)]) if wanted else pd.DataFrame()
    file_names = {
        "longest": f"32_longest_search_top{settings['n_longest_map']}.html",
        "need_cruise": "32_sample_need_cruise.html",
        "no_cruise": "32_sample_no_cruise.html",
        "generic_sample": "32_generic_sample.html",
    }
    maps = {
        key: write_trace_map(map_points, selection, output_path(config, "maps", file_names[key]), title)
        for key, (selection, title) in map_selections.items()
    }

    curves_path = None
    if len(generic):
        fig = plot_probability_curves(
            map_points, generic, settings["p_search"], settings["max_search_duration_minutes"]
        )
        curves_path = output_path(config, "figures", "32_probability_curves_generic_sample.png")
        fig.savefig(curves_path, dpi=150)
        plt.close(fig)
        generic.to_csv(output_path(config, "reports", "32_generic_sample_segments.csv"), index=False)

    # Table au niveau de la leg (collection legs/)
    leg_table_info = None
    if settings["write_leg_table"]:
        from .leg_collection import write_leg_table, write_persons_table

        run_id = datetime.now().strftime("%Y%m%dT%H%M%S") + (
            f"_{settings['max_legs']}legs_seed{settings['random_seed']}" if is_sample else "_all"
        )
        relut_by_leg = build_relut_by_leg(config, summary, sampled_legs=sampled_legs, settings=settings, run_id=run_id)
        leg_table_path = write_leg_table(
            relut_by_leg, config, "relut_by_leg", run_id=run_id, sample=is_sample,
            params={
                "p_search": settings["p_search"],
                "max_search_duration_minutes": settings["max_search_duration_minutes"],
                "corpus": "legs/corpus_by_leg.parquet (leg_filters du YAML)",
                "max_legs": settings["max_legs"],
                "random_seed": settings["random_seed"],
                "model_path": str(settings["model_path"]),
                "segment_rule": "dernier segment de la leg, sans repli",
            },
        )
        write_persons_table(config)
        leg_table_info = {
            "path": str(leg_table_path),
            "run_id": run_id,
            "sample": is_sample,
            "status_counts": {k: int(v) for k, v in relut_by_leg["relut_status"].value_counts().items()},
        }
        print(f"[relut] table par leg : {leg_table_path} — {leg_table_info['status_counts']}", flush=True)

    global_summary = {
        "scope": scope,
        "segments_available": all_segments,
        "segments_total": int(len(summary)),
        "legs_total": n_legs,
        "segments_with_search": int(summary["search_predicted"].sum()),
        "points_total": total_points,
        "searching_points": searching_points,
        "mean_search_duration_minutes": float(summary["search_duration_seconds"].mean() / 60),
        "median_search_duration_minutes": float(summary["search_duration_seconds"].median() / 60),
        "mean_search_start_remaining_distance_m": float(summary["search_start_remaining_distance_m"].mean()),
        "model": {"path": str(settings["model_path"]), "p_search": settings["p_search"],
                  "max_search_duration_minutes": settings["max_search_duration_minutes"]},
        "loop_filter": loop_info,
        "relut_by_leg": leg_table_info,
        "parking_variable": settings["parking_variable"] if parking_available else None,
        "parking_groups": settings["groups"] if parking_available else None,
        "seconds": round(time.perf_counter() - started, 1),
        "outputs": {
            "predictions": str(predictions_dir),
            "summary_by_segment": str(output_path(config, "reports", "prediction_summary_by_segment.csv")),
            "stats": str(output_path(config, "reports", "32_search_duration_stats.csv")),
            "histograms": str(histogram_path),
            "distance_stats": str(output_path(config, "reports", "32_search_distance_stats.csv")),
            "distance_histograms": str(distance_histogram_path),
            "probability_curves": str(curves_path) if curves_path else None,
            "maps": {key: str(value) if value else None for key, value in maps.items()},
        },
    }
    write_json(output_path(config, "reports", "prediction_global_summary.json"), global_summary)
    return {"summary": global_summary, "stats": stats, "by_category": by_category, "distance_stats": distance_table}


# --------------------------------------------------------------------------- #
# Part des legs au-delà de seuils, par catégorie de stationnement déclaré
# --------------------------------------------------------------------------- #


def threshold_share_table(
    config: dict[str, Any], variable: str, measure: str, thresholds: list[float], *,
    sample_run_id: str | None = None,
) -> pd.DataFrame:
    """Part des legs évaluées (ReLUT) dont la durée ou la distance de recherche dépasse chaque seuil.

    ``variable`` : ``space_home`` ou ``space_work`` ; ``measure`` : ``duration`` (seuils en
    secondes) ou ``distance`` (seuils en mètres). Une ligne par catégorie (plus
    « Ensemble »), une colonne par seuil (« > seuil », strictement), plus ``n_legs``.
    Population : legs dont le dernier segment est prédit (``relut_status == "evaluated"``),
    une leg sans recherche comptant pour 0.
    """
    from .leg_collection import load_leg_table

    column = {"duration": "relut_search_duration_s", "distance": "relut_search_distance_m"}[measure]
    unit = "s" if measure == "duration" else "m"
    legs = load_leg_table(config, ["relut_by_leg"], columns={
        "legs_enriched": ["leg_id", variable], "relut_by_leg": ["relut_status", column],
    }, sample_run_id=sample_run_id, verbose=False)
    legs = legs[legs["relut_status"].eq("evaluated")]
    legs[variable] = legs[variable].astype("string").fillna("non renseigné")
    groups = [("Ensemble", legs)] + [(str(value), legs[legs[variable].eq(value)])
                                     for value in legs[variable].value_counts().index]
    rows = []
    for label, sub in groups:
        row = {variable: label, "n_legs": int(len(sub))}
        for threshold in thresholds:
            row[f"> {threshold:g} {unit}"] = float((sub[column] > threshold).mean()) if len(sub) else np.nan
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(output_path(config, "reports", f"32_part_au_dela_seuils_{measure}_{variable}.csv"), index=False)
    return table


# --------------------------------------------------------------------------- #
# Part des legs au-delà de seuils, par catégorie de stationnement déclaré
# --------------------------------------------------------------------------- #


def share_above_thresholds(
    config: dict[str, Any], variable: str, quantity: str, thresholds: list[float], *, sample_run_id: str | None = None
) -> pd.DataFrame:
    """Part des legs évaluées dont la durée (``quantity="duration"``, seuils en minutes) ou la
    distance (``"distance"``, seuils en mètres) de recherche prédite atteint chaque seuil,
    par catégorie de ``variable`` (``space_home`` ou ``space_work``).

    Une ligne par catégorie (plus « toutes les legs »), une colonne par seuil.
    Population : legs du corpus évaluées par ReLUT (sans recherche = 0). Comptage par leg, non pondéré.
    """
    from .leg_collection import load_leg_table

    column, scale, unit = {
        "duration": ("relut_search_duration_s", 1 / 60, "min"),
        "distance": ("relut_search_distance_m", 1.0, "m"),
    }[quantity]
    legs = load_leg_table(config, ["relut_by_leg"], columns={
        "legs_enriched": ["leg_id", variable], "relut_by_leg": ["relut_status", column]},
        sample_run_id=sample_run_id, verbose=False)
    legs = legs[legs["relut_status"].eq("evaluated")].copy()
    legs["valeur"] = legs[column] * scale
    legs[variable] = legs[variable].astype("string").fillna("non renseigné")
    groups = [("toutes les legs", legs)] + [(str(k), g) for k, g in legs.groupby(variable, sort=True)]
    rows = []
    for label, group in groups:
        row: dict[str, Any] = {variable: label, "n_legs": int(len(group))}
        for threshold in thresholds:
            row[f"≥ {threshold:g} {unit}"] = float(group["valeur"].ge(threshold).mean()) if len(group) else np.nan
        rows.append(row)
    table = pd.DataFrame(rows)
    path = output_path(config, "reports", f"32_part_legs_{quantity}_par_{variable}.csv")
    table.to_csv(path, index=False)
    return table
