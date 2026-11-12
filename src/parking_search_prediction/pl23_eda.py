"""Étape 21 — exploration des waypoints voiture et du stationnement déclaré (PL23).

Entrées : sorties de l'étape 01 (``waypoints_caronly.parquet``, fichier ou
dossier de tranches), table de correspondance ``user_id_motiontag`` →
``user_id_fors`` (``leg_matching.user_id_map``) et questionnaire PL23
(``pl23_survey.survey_file``).

Produit :

- ``figures/21_waypoint_distributions.png`` : précision GPS, vitesse utilisée,
  intervalle et distance au point précédent ;
- ``figures/21_parking_shares.png`` : parts place dédiée / pas de place sans
  ou avec besoin de chercher, au domicile et au travail ;
- ``reports/21_waypoint_distributions.csv`` (histogrammes),
  ``reports/21_users_parking.csv`` (une ligne par participant),
  ``reports/21_parking_shares.csv`` et ``reports/21_eda_summary.json``.

Les histogrammes sont accumulés fichier par fichier : la mémoire ne dépend que
de la taille d'une tranche, pas du volume total.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ensure_output_dirs, output_path
from .pl23_survey import PARKING_CATEGORIES, parking_by_user
from .utils import write_json

# variable -> (colonne, libellé, unité, bornes de l'histogramme)
# Les valeurs hors bornes sont comptées à part (part « > max »).
DISTRIBUTIONS = {
    "accuracy": ("accuracy", "Précision GPS des waypoints", "m", (0.0, 100.0, 100)),
    "speed": ("speed_kmh", "Vitesse utilisée", "km/h", (0.0, 200.0, 100)),
    "interval": ("dt_seconds", "Intervalle entre points", "s", (0.0, 60.0, 60)),
    "distance": ("distance_from_previous_m", "Distance au point précédent", "m", (0.0, 200.0, 100)),
}


# --------------------------------------------------------------------------- #
# Histogrammes des waypoints voiture
# --------------------------------------------------------------------------- #


def _parquet_parts(path: Path) -> list[Path]:
    if path.is_dir():
        parts = sorted(path.glob("*.parquet"))
        if not parts:
            raise FileNotFoundError(f"Aucun fichier Parquet dans {path}")
        return parts
    if not path.exists():
        raise FileNotFoundError(f"{path} introuvable : lancer d'abord l'étape 01.")
    return [path]


def waypoint_histograms(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Histogrammes des waypoints voiture, accumulés tranche par tranche.

    L'intervalle et la distance au point précédent sont ceux calculés par
    ``clean_waypoints`` ; le premier point de chaque leg est exclu, car son
    « point précédent » appartient à un autre déplacement (arrêt entre deux legs).
    """
    distributions = dict(DISTRIBUTIONS)
    # Axes alignés sur les seuils du nettoyage (au-delà, les points ont déjà été retirés ou plafonnés).
    cleaning = config.get("cleaning", {}) or {}
    max_accuracy = float(cleaning.get("max_accuracy_m") or DISTRIBUTIONS["accuracy"][3][1])
    max_speed = float(cleaning.get("max_speed_kmh") or DISTRIBUTIONS["speed"][3][1])
    distributions["accuracy"] = (
        config["raw_waypoints"]["accuracy_col"], *DISTRIBUTIONS["accuracy"][1:3],
        (0.0, max_accuracy, max(10, int(round(max_accuracy)))),
    )
    distributions["speed"] = (*DISTRIBUTIONS["speed"][:3], (0.0, max_speed, max(10, int(round(max_speed / 2)))))

    state = {
        name: {
            "edges": np.linspace(low, high, bins + 1),
            "counts": np.zeros(bins, dtype=np.int64),
            "n": 0, "missing": 0, "below": 0, "above": 0, "sum": 0.0,
        }
        for name, (_, _, _, (low, high, bins)) in distributions.items()
    }
    columns = sorted({spec[0] for spec in distributions.values()} | {"leg_id", "timestamp"})

    for part in _parquet_parts(output_path(config, "data", "waypoints_caronly.parquet")):
        frame = pd.read_parquet(part, columns=columns)
        frame = frame.sort_values(["leg_id", "timestamp"], kind="mergesort")
        first_of_leg = ~frame["leg_id"].duplicated()
        for name, (column, *_rest) in distributions.items():
            values = frame[column].astype("float64")
            if name in ("interval", "distance"):
                values = values.mask(first_of_leg)
            values = values.to_numpy()
            item = state[name]
            valid = values[~np.isnan(values)]
            item["missing"] += int(np.isnan(values).sum())
            item["n"] += int(valid.size)
            item["sum"] += float(valid.sum())
            item["below"] += int((valid < item["edges"][0]).sum())
            item["above"] += int((valid > item["edges"][-1]).sum())
            item["counts"] += np.histogram(valid, bins=item["edges"])[0]

    result = {}
    for name, (column, label, unit, _) in distributions.items():
        item = state[name]
        cumulative = np.cumsum(item["counts"]) + item["below"]

        def quantile(q: float, item=item, cumulative=cumulative) -> float | None:
            """Quantile approché : interpolation linéaire dans la classe ; None s'il tombe hors bornes."""
            target = q * item["n"]
            if not item["n"] or target <= item["below"]:
                return None
            index = int(np.searchsorted(cumulative, target))
            if index >= len(cumulative):
                return None
            before = cumulative[index - 1] if index else item["below"]
            low, high = item["edges"][index], item["edges"][index + 1]
            fraction = (target - before) / item["counts"][index] if item["counts"][index] else 0.0
            return float(low + fraction * (high - low))
        result[name] = {
            "column": column, "label": label, "unit": unit,
            "edges": item["edges"], "counts": item["counts"],
            "n": item["n"], "missing": item["missing"],
            "share_above_range": item["above"] / item["n"] if item["n"] else None,
            "mean": item["sum"] / item["n"] if item["n"] else None,
            "median_approx": quantile(0.5),   # interpolé dans la classe ; None si hors bornes
            "p90_approx": quantile(0.9),
        }
    return result


def plot_waypoint_histograms(histograms: dict[str, dict[str, Any]]):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, item in zip(axes.flat, histograms.values()):
        edges, counts = item["edges"], item["counts"]
        shares = counts / item["n"] * 100 if item["n"] else counts
        ax.bar(edges[:-1], shares, width=np.diff(edges), align="edge", color="#3b6ea5", edgecolor="white", linewidth=0.3)
        above = item["share_above_range"] or 0.0
        median = item["median_approx"]
        ax.set_title(f"{item['label']} ({item['unit']})")
        ax.set_xlabel(item["unit"])
        ax.set_ylabel("% des waypoints")
        ax.text(
            0.98, 0.95,
            f"n = {item['n']:,}".replace(",", "'")
            + (f"\nmédiane ≈ {median:.1f} {item['unit']}" if median is not None else "")
            + f"\n> {edges[-1]:g} {item['unit']} : {above:.1%}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "#cccccc"},
        )
    fig.suptitle("Waypoints voiture PL23 (sortie de l'étape 01)")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Stationnement déclaré (questionnaire) par participant
# --------------------------------------------------------------------------- #


def user_parking_table(config: dict[str, Any]) -> pd.DataFrame:
    """Une ligne par participant présent dans les waypoints voiture, avec
    ``space_home`` / ``space_work`` du questionnaire (voir ``pl23_survey.parking_by_user``).
    """
    import pyarrow.parquet as pq

    users: set[str] = set()
    for part in _parquet_parts(output_path(config, "data", "waypoints_caronly.parquet")):
        users.update(pq.read_table(part, columns=["source_user_id"]).column(0).unique().to_pylist())
    return parking_by_user(config, users)


def parking_shares(users: pd.DataFrame) -> pd.DataFrame:
    """Parts des 3 catégories parmi les participants qui en ont une, au domicile et au travail.

    Les autres cas (``Aucun`` au domicile, non renseigné, absent du questionnaire)
    sont comptés dans ``n_hors_categories`` mais exclus du dénominateur.
    """
    rows = []
    for column, place in (("space_home", "Domicile"), ("space_work", "Travail / études")):
        values = users[column]
        in_categories = values.isin(PARKING_CATEGORIES)
        total = int(in_categories.sum())
        for category in PARKING_CATEGORIES:
            count = int(values.eq(category).sum())
            rows.append({
                "lieu": place, "categorie": category, "n": count,
                "part": count / total if total else np.nan,
                "n_categorises": total,
                "n_hors_categories": int((~in_categories).sum()),
            })
    return pd.DataFrame(rows)


def plot_parking_shares(shares: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.5))
    places = list(dict.fromkeys(shares["lieu"]))
    width = 0.38
    x = np.arange(len(PARKING_CATEGORIES))
    for offset, (place, color) in zip((-width / 2, width / 2), zip(places, ("#3b6ea5", "#e0913a"))):
        sub = shares[shares["lieu"] == place].set_index("categorie").reindex(PARKING_CATEGORIES)
        bars = ax.bar(x + offset, sub["part"] * 100, width, label=f"{place} (n = {int(sub['n_categorises'].iloc[0])})", color=color)
        ax.bar_label(bars, labels=[f"{v:.0%}" if pd.notna(v) else "" for v in sub["part"]], fontsize=9, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(["Place dédiée", "Pas de place,\npas besoin de chercher", "Pas de place,\nbesoin de chercher"])
    ax.set_ylabel("% des participants")
    ax.set_title("Stationnement déclaré — participants PL23 avec des waypoints voiture")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Point d'entrée
# --------------------------------------------------------------------------- #


def run_pl23_eda(config: dict[str, Any]) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    ensure_output_dirs(config)

    histograms = waypoint_histograms(config)
    fig = plot_waypoint_histograms(histograms)
    distributions_png = output_path(config, "figures", "21_waypoint_distributions.png")
    fig.savefig(distributions_png, dpi=150)
    plt.close(fig)
    pd.concat(
        [
            pd.DataFrame({
                "variable": name, "colonne": item["column"], "unite": item["unit"],
                "borne_basse": item["edges"][:-1], "borne_haute": item["edges"][1:], "n": item["counts"],
            })
            for name, item in histograms.items()
        ],
        ignore_index=True,
    ).to_csv(output_path(config, "reports", "21_waypoint_distributions.csv"), index=False)

    users = user_parking_table(config)
    users.to_csv(output_path(config, "reports", "21_users_parking.csv"), index=False)
    shares = parking_shares(users)
    shares.to_csv(output_path(config, "reports", "21_parking_shares.csv"), index=False)
    fig = plot_parking_shares(shares)
    shares_png = output_path(config, "figures", "21_parking_shares.png")
    fig.savefig(shares_png, dpi=150)
    plt.close(fig)

    summary = {
        "waypoint_distributions": {
            name: {key: item[key] for key in ("column", "unit", "n", "missing", "mean", "median_approx", "p90_approx", "share_above_range")}
            for name, item in histograms.items()
        },
        "users_with_car_waypoints": int(len(users)),
        "users_mapped_to_panel_id": int(users["user_id_fors"].notna().sum()),
        "users_in_survey": int(users["in_survey"].sum()),
        "space_home_counts": users["space_home"].value_counts(dropna=False).rename(index=lambda v: str(v)).to_dict(),
        "space_work_counts": users["space_work"].value_counts(dropna=False).rename(index=lambda v: str(v)).to_dict(),
        "figures": {"distributions": str(distributions_png), "parking_shares": str(shares_png)},
    }
    write_json(output_path(config, "reports", "21_eda_summary.json"), summary)
    return summary
