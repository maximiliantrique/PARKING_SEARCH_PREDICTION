"""Étape 42 — comparaison des méthodes temporelle (ReLUT) et spatiale, leg par leg.

Lit ``relut_by_leg`` et ``spatial_by_leg`` (collection ``legs/``, même corpus
commun) et compare les durées et les distances de cruising prédites / estimées :
histogrammes superposés, nuage de points avec régression linéaire (moindres
carrés), corrélations de Pearson et de Spearman.

Populations (``population``) :

- ``both_evaluated`` : legs évaluées par les deux méthodes (0 = pas de cruising) ;
- ``both_detected`` : legs avec cruising détecté par les deux méthodes ;
- ``either_detected`` : legs évaluées par les deux, avec cruising détecté par au moins une.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import output_path
from .leg_collection import load_leg_table

POPULATIONS = {
    "both_evaluated": "legs évaluées par les deux méthodes",
    "both_detected": "legs avec cruising détecté par les deux méthodes",
    "either_detected": "legs évaluées par les deux, cruising détecté par au moins une",
}
LABELS = {"relut": "Temporel (ReLUT)", "spatial": "Spatial"}
COLORS = {"relut": ("#F6B6C8", "#D9708F"), "spatial": ("#B9E3C6", "#5FAE7C")}


def load_comparison(
    config: dict[str, Any], *, population: str = "both_evaluated",
    relut_duration: str = "relut_search_duration_s", relut_distance: str = "relut_search_distance_m",
    spatial_duration: str = "spatial_search_duration_s", spatial_distance: str = "spatial_search_distance_m",
) -> pd.DataFrame:
    """Une ligne par leg de la population choisie, avec durée (min) et distance (m) pour chaque méthode."""
    if population not in POPULATIONS:
        raise ValueError(f"population : {list(POPULATIONS)}")
    legs = load_leg_table(config, ["relut_by_leg", "spatial_by_leg"], columns={
        "legs_enriched": ["leg_id", "purpose_agg", "space_home", "space_work"],
        "relut_by_leg": ["relut_status", "relut_search_detected", relut_duration, relut_distance],
        "spatial_by_leg": ["spatial_status", "spatial_search_detected", spatial_duration, spatial_distance],
    }, verbose=False)
    both = legs["relut_status"].eq("evaluated") & legs["spatial_status"].eq("evaluated")
    relut_hit = legs["relut_search_detected"].astype("boolean").fillna(False).astype(bool)
    spatial_hit = legs["spatial_search_detected"].astype("boolean").fillna(False).astype(bool)
    mask = both & {"both_evaluated": True, "both_detected": relut_hit & spatial_hit,
                   "either_detected": relut_hit | spatial_hit}[population]
    out = legs[mask].copy()
    out["relut_duration_min"] = out[relut_duration] / 60
    out["spatial_duration_min"] = out[spatial_duration] / 60
    out["relut_distance_m"] = out[relut_distance]
    out["spatial_distance_m"] = out[spatial_distance]
    out.attrs.update({"population": population, "columns": {
        "relut_duration": relut_duration, "relut_distance": relut_distance,
        "spatial_duration": spatial_duration, "spatial_distance": spatial_distance}})
    return out


def correlation_table(comparison: pd.DataFrame) -> pd.DataFrame:
    """Pearson, Spearman (avec p-valeurs), régression linéaire spatial = a · temporel + b, médianes."""
    from scipy import stats

    rows = []
    for quantity, unit in (("duration", "min"), ("distance", "m")):
        x = comparison[f"relut_{quantity}_{unit}"]
        y = comparison[f"spatial_{quantity}_{unit}"]
        valid = x.notna() & y.notna()
        x, y = x[valid].to_numpy(float), y[valid].to_numpy(float)
        row: dict[str, Any] = {"grandeur": "durée (min)" if quantity == "duration" else "distance (m)", "n_legs": int(len(x))}
        constant = len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0
        if constant:
            row.update({k: np.nan for k in ("pearson_r", "pearson_p", "spearman_rho", "spearman_p", "pente", "ordonnee", "r2")})
        else:
            pearson = stats.pearsonr(x, y)
            spearman = stats.spearmanr(x, y)
            slope, intercept = np.polyfit(x, y, 1)
            row.update({"pearson_r": pearson[0], "pearson_p": pearson[1], "spearman_rho": spearman[0],
                        "spearman_p": spearman[1], "pente": slope, "ordonnee": intercept, "r2": pearson[0] ** 2})
        row.update({"mediane_temporel": np.median(x) if len(x) else np.nan, "mediane_spatial": np.median(y) if len(y) else np.nan,
                    "ecart_moyen_spatial_moins_temporel": float(np.mean(y - x)) if len(x) else np.nan})
        rows.append(row)
    return pd.DataFrame(rows)


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#e5e7eb", lw=0.8)
    ax.set_axisbelow(True)


def plot_overlaid_histograms(comparison: pd.DataFrame, *, quantile_cap: float = 0.99, bins: int = 40):
    """Durées et distances : histogrammes superposés des deux méthodes (en % des legs)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for ax, (quantity, unit, title) in zip(axes, (("duration", "min", "Durée de cruising"), ("distance", "m", "Distance de cruising"))):
        both = pd.concat([comparison[f"relut_{quantity}_{unit}"], comparison[f"spatial_{quantity}_{unit}"]]).dropna()
        if both.empty:
            continue
        upper = both.quantile(quantile_cap)
        edges = np.linspace(0, upper if upper > 0 else 1, bins + 1)
        for method in ("relut", "spatial"):
            values = comparison[f"{method}_{quantity}_{unit}"].dropna()
            fill, line = COLORS[method]
            ax.hist(values.clip(upper=edges[-1]), bins=edges, weights=np.full(len(values), 100 / max(len(values), 1)),
                    histtype="stepfilled", alpha=0.6, color=fill, edgecolor=line, lw=1,
                    label=f"{LABELS[method]} — médiane {values.median():.1f} {unit}")
            ax.axvline(values.median(), color=line, ls=":", lw=1.5)
        ax.set_xlabel(f"{title.lower()} ({unit}) — au-delà du quantile {quantile_cap:.0%} : dernière classe")
        ax.set_ylabel("% des legs")
        ax.set_title(title, fontsize=10)
        ax.legend(frameon=False, fontsize=8)
        _style(ax)
    fig.suptitle(f"{POPULATIONS[comparison.attrs.get('population', 'both_evaluated')]} (n = {len(comparison)})", fontsize=10)
    fig.tight_layout()
    return fig


def plot_scatter_regression(comparison: pd.DataFrame, quantity: str, *, quantile_cap: float = 0.995):
    """Nuage temporel (x) vs spatial (y), droite de régression et diagonale y = x."""
    import matplotlib.pyplot as plt

    unit = "min" if quantity == "duration" else "m"
    x = comparison[f"relut_{quantity}_{unit}"]
    y = comparison[f"spatial_{quantity}_{unit}"]
    valid = x.notna() & y.notna()
    x, y = x[valid].to_numpy(float), y[valid].to_numpy(float)
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    if len(x):
        limit = float(np.nanquantile(np.r_[x, y], quantile_cap)) or 1.0
        ax.scatter(x, y, s=10, alpha=0.35 if len(x) > 500 else 0.7, color="#B7D3F2", edgecolor="#5B8FCB", lw=0.3)
        ax.plot([0, limit], [0, limit], color="#9ca3af", ls="--", lw=1, label="y = x")
        if len(x) >= 3 and np.ptp(x) > 0:
            slope, intercept = np.polyfit(x, y, 1)
            r = np.corrcoef(x, y)[0, 1] if np.ptp(y) > 0 else np.nan
            grid = np.linspace(0, limit, 50)
            ax.plot(grid, slope * grid + intercept, color="#D9708F", lw=2,
                    label=f"régression : y = {slope:.2f} x + {intercept:.2f}  (R² = {r ** 2:.2f})")
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    name = "Durée de cruising" if quantity == "duration" else "Distance de cruising"
    ax.set_xlabel(f"{LABELS['relut']} ({unit})")
    ax.set_ylabel(f"{LABELS['spatial']} ({unit})")
    ax.set_title(f"{name} : {LABELS['relut']} vs {LABELS['spatial']} (n = {len(x)})", fontsize=10)
    _style(ax)
    fig.tight_layout()
    return fig


def run_comparison(config: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    comparison = load_comparison(config, **kwargs)
    table = correlation_table(comparison)
    figures = {
        "histograms": plot_overlaid_histograms(comparison),
        "scatter_duration": plot_scatter_regression(comparison, "duration"),
        "scatter_distance": plot_scatter_regression(comparison, "distance"),
    }
    population = comparison.attrs["population"]
    for name, fig in figures.items():
        fig.savefig(output_path(config, "figures", f"42_{name}_{population}.png"), dpi=150)
    table.to_csv(output_path(config, "reports", f"42_correlations_{population}.csv"), index=False)
    return {"comparison": comparison, "correlations": table, "figures": figures}
