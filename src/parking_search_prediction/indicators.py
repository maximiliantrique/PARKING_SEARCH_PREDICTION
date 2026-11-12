"""Étape 41 — indicateurs de cruising par heure et par motif, pour une ou plusieurs méthodes.

Lit la collection de tables par leg (``leg_collection.load_leg_table``) :
``legs_enriched`` (heure de fin de leg, motif) + une table par méthode
(``relut_by_leg`` aujourd'hui, ``spatial_by_leg`` / ``hybrid_by_leg`` plus tard).

Contrat d'une méthode (préfixe ``<p>``, mêmes conventions que ``relut_by_leg``) :

- ``<p>_status`` : ``evaluated`` pour les legs évaluées ;
- ``<p>_search_detected`` : cruising détecté (booléen) ;
- ``<p>_search_duration_s`` : durée du cruising, s ;
- ``<p>_search_distance_m`` : distance du cruising, m (méthode temporelle :
  distance parcourue entre waypoints depuis la bascule).

Indicateurs (population = legs évaluées par la méthode) :

1–3. part de legs avec cruising, durée et distance du cruising par heure ;
4–6. les mêmes par motif du trip ;
7–8. répartition des legs **avec cruising** par heure et par motif (somme = 100 %),
     comparée à la répartition de toutes les legs évaluées (référence grise).

``min_cruising_duration_s`` (0 par défaut) : une leg ne compte « avec cruising »
que si la durée prédite atteint ce seuil (filtre les micro-recherches).

L'heure est celle de **fin de la leg** (moment du stationnement), à l'heure
locale ``local_timezone``, arrondie à l'heure la plus proche : la classe « 9 h »
va de 8:30:00 à 9:29:59. Durées et distances sont décrites **parmi les legs avec
cruising détecté** (médiane, quartiles, moyenne). Les classes de moins de
``min_legs`` legs évaluées (ou détectées pour durées et distances) ne sont pas
tracées mais restent dans les tableaux. Indicateurs non pondérés (niveau leg).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import ensure_output_dirs, output_path
from .leg_collection import TABLES, load_leg_table, table_path
from .utils import write_json

# Couleurs pastel (remplissage) et teinte plus soutenue de la même couleur (traits, bords)
PALETTE = [
    ("#F6B6C8", "#D9708F"),  # rose
    ("#B9E3C6", "#5FAE7C"),  # vert
    ("#F9E3A3", "#D1A93A"),  # jaune
    ("#B7D3F2", "#5B8FCB"),  # bleu
]

PURPOSE_ORDER = [
    "Travail", "Formation", "Achats et services", "Loisirs",
    "Accompagnement", "Retour à la maison", "Autres",
]


@dataclass(frozen=True)
class Method:
    key: str
    label: str
    table: str
    prefix: str

    @property
    def status(self) -> str:
        return f"{self.prefix}_status"

    @property
    def detected(self) -> str:
        return f"{self.prefix}_search_detected"

    @property
    def duration(self) -> str:
        return f"{self.prefix}_search_duration_s"

    @property
    def distance(self) -> str:
        return f"{self.prefix}_search_distance_m"


METHODS = {
    "relut": Method("relut", "Temporel (ReLUT)", "relut_by_leg", "relut"),
    "spatial": Method("spatial", "Spatial", "spatial_by_leg", "spatial"),
    "hybrid": Method("hybrid", "Hybride", "hybrid_by_leg", "hybrid"),
}


# --------------------------------------------------------------------------- #
# Données
# --------------------------------------------------------------------------- #


def load_indicator_table(
    config: dict[str, Any], methods: list[str], *, sample_run_id: str | None = None
) -> tuple[pd.DataFrame, list[Method]]:
    """Vue d'assemblage avec les méthodes disponibles (les tables absentes sont signalées et ignorées)."""
    available: list[Method] = []
    for key in methods:
        if key not in METHODS:
            raise KeyError(f"Méthode inconnue {key!r} ; méthodes définies : {sorted(METHODS)}")
        method = METHODS[key]
        if table_path(config, method.table, sample_run_id).exists():
            available.append(method)
        else:
            print(f"[indicateurs] {method.label} : table {method.table} absente, méthode ignorée.")
    if not available:
        raise FileNotFoundError("Aucune table de méthode disponible (lancer l'étape 32 d'abord).")
    if not table_path(config, "legs_enriched").exists():
        raise FileNotFoundError("legs_enriched.parquet absent : l'heure et le motif viennent de l'enrichissement (pl23).")

    import pyarrow.parquet as pq

    enriched_columns = set(pq.read_schema(table_path(config, "legs_enriched")).names)
    wanted = ["leg_id", "leg_finished_at", "purpose_agg", "trip_purpose", "effective_mode"]
    columns = {"legs_enriched": [c for c in wanted if c in enriched_columns]}
    for method in available:
        needed = [method.status, method.detected, method.duration, method.distance]
        present = set(pq.read_schema(table_path(config, method.table, sample_run_id)).names)
        missing = [c for c in needed if c not in present]
        if missing:
            raise KeyError(f"{method.table} ne respecte pas le contrat des méthodes : colonnes absentes {missing}")
        columns[method.table] = needed
    legs = load_leg_table(config, [m.table for m in available], columns=columns, sample_run_id=sample_run_id, verbose=False)
    return legs, available


def hour_bins(instants: pd.Series, timezone: str) -> pd.Series:
    """Heure arrondie à l'heure la plus proche, en heure locale : 8:30:00–9:29:59 → 9."""
    local = pd.to_datetime(instants, utc=True).dt.tz_convert(timezone)
    return ((local + pd.Timedelta(minutes=30)).dt.hour).astype("Int64")


def wilson_interval(successes: np.ndarray, totals: np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    totals = np.asarray(totals, dtype=float)
    p = np.divide(successes, totals, out=np.full_like(totals, np.nan), where=totals > 0)
    with np.errstate(divide="ignore", invalid="ignore"):   # classes vides : NaN, sans avertissement
        denominator = 1 + z**2 / totals
        centre = (p + z**2 / (2 * totals)) / denominator
        half = z * np.sqrt(p * (1 - p) / totals + z**2 / (4 * totals**2)) / denominator
    return centre - half, centre + half


def detected_mask(evaluated: pd.DataFrame, method: Method, min_duration_s: float = 0.0) -> pd.Series:
    """Cruising détecté, et d'au moins ``min_duration_s`` secondes si un seuil est fixé."""
    mask = evaluated[method.detected].astype("boolean").fillna(False).astype(bool)
    if min_duration_s:
        mask &= evaluated[method.duration].ge(min_duration_s).fillna(False)
    return mask


def indicator_stats(
    legs: pd.DataFrame, methods: list[Method], by: str, order: list[Any], min_duration_s: float = 0.0
) -> pd.DataFrame:
    """Une ligne par (méthode, classe) : part détectée, durée et distance parmi les détectés."""
    rows = []
    for method in methods:
        evaluated = legs[legs[method.status].eq("evaluated") & legs[by].notna()]
        detected_mask_ = detected_mask(evaluated, method, min_duration_s)
        for value in order:
            group = evaluated[evaluated[by].eq(value)]
            hits = group[detected_mask_.loc[group.index]]
            duration_min = hits[method.duration] / 60
            distance = hits[method.distance]
            row = {
                "methode": method.label,
                "classe": value,
                "n_legs_evaluees": int(len(group)),
                "n_legs_cruising": int(len(hits)),
                "part_cruising": len(hits) / len(group) if len(group) else np.nan,
            }
            for name, values in (("duree_min", duration_min), ("distance_m", distance)):
                row.update({
                    f"{name}_moyenne": values.mean() if len(values) else np.nan,
                    f"{name}_mediane": values.median() if len(values) else np.nan,
                    f"{name}_q25": values.quantile(0.25) if len(values) else np.nan,
                    f"{name}_q75": values.quantile(0.75) if len(values) else np.nan,
                })
            rows.append(row)
    stats = pd.DataFrame(rows)
    low, high = wilson_interval(stats["n_legs_cruising"].to_numpy(), stats["n_legs_evaluees"].to_numpy())
    stats.insert(stats.columns.get_loc("part_cruising") + 1, "part_ic95_bas", low)
    stats.insert(stats.columns.get_loc("part_ic95_bas") + 1, "part_ic95_haut", high)
    return stats


def composition_stats(
    legs: pd.DataFrame, methods: list[Method], by: str, order: list[Any], min_duration_s: float = 0.0
) -> pd.DataFrame:
    """Répartition des legs **avec cruising** entre les classes (somme = 100 % par méthode).

    ``part_reference`` donne la même répartition pour toutes les legs évaluées :
    une classe sur-représentée parmi les cruisings a ``part_cruising > part_reference``.
    """
    rows = []
    for method in methods:
        evaluated = legs[legs[method.status].eq("evaluated") & legs[by].notna()]
        hits = evaluated[detected_mask(evaluated, method, min_duration_s)]
        n_hits, n_evaluated = len(hits), len(evaluated)
        for value in order:
            k = int(hits[by].eq(value).sum())
            k_ref = int(evaluated[by].eq(value).sum())
            rows.append({
                "methode": method.label, "classe": value,
                "n_legs_cruising": k, "n_total_cruising": n_hits,
                "part_cruising": k / n_hits if n_hits else np.nan,
                "n_legs_evaluees": k_ref, "n_total_evaluees": n_evaluated,
                "part_reference": k_ref / n_evaluated if n_evaluated else np.nan,
            })
    stats = pd.DataFrame(rows)
    low, high = wilson_interval(stats["n_legs_cruising"].to_numpy(), stats["n_total_cruising"].to_numpy())
    stats.insert(stats.columns.get_loc("part_cruising") + 1, "part_ic95_bas", low)
    stats.insert(stats.columns.get_loc("part_ic95_bas") + 1, "part_ic95_haut", high)
    return stats


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

INDICATORS = {
    "share": ("part_cruising", "part_ic95_bas", "part_ic95_haut", "n_legs_evaluees", "% des legs évaluées", 100.0),
    "duration": ("duree_min_mediane", "duree_min_q25", "duree_min_q75", "n_legs_cruising", "durée du cruising (min)", 1.0),
    "distance": ("distance_m_mediane", "distance_m_q25", "distance_m_q75", "n_legs_cruising", "distance du cruising (m)", 1.0),
}
TITLES = {
    "share": "Part de legs avec cruising",
    "duration": "Durée du cruising (médiane, intervalle interquartile)",
    "distance": "Distance du cruising (médiane, intervalle interquartile)",
}


def method_colors(methods: list[str]) -> dict[str, tuple[str, str]]:
    """Couleur fixe par méthode (rose = temporel, vert = spatial, jaune = hybride, bleu = autre)."""
    fixed = {method.label: PALETTE[i % len(PALETTE)] for i, method in enumerate(METHODS.values())}
    extra = iter(PALETTE[len(METHODS):] or PALETTE)
    return {m: fixed.get(m) or next(extra, PALETTE[-1]) for m in methods}


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e7eb", lw=0.8)
    ax.set_axisbelow(True)


def plot_indicator(
    stats: pd.DataFrame, indicator: str, dimension: str, *, layout: str = "panels", min_legs: int = 10,
    min_duration_s: float = 0.0,
):
    """``dimension`` = ``hour`` (courbe sur 24 h) ou ``purpose`` (barres par motif)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    value_col, low_col, high_col, n_col, ylabel, scale = INDICATORS[indicator]
    methods = list(dict.fromkeys(stats["methode"]))
    colors = method_colors(methods)
    classes = list(dict.fromkeys(stats["classe"]))
    n_axes = len(methods) if layout == "panels" else 1
    width_per_axis = (6.5 if dimension == "hour" else 7.0) if layout == "panels" else 10.0
    fig, axes = plt.subplots(1, n_axes, figsize=(width_per_axis * n_axes, 4.6), sharey=True, squeeze=False)
    axes = axes[0]

    for index, method in enumerate(methods):
        ax = axes[index if layout == "panels" else 0]
        fill, line = colors[method]
        sub = stats[stats["methode"] == method].set_index("classe").reindex(classes)
        enough = sub[n_col] >= min_legs
        value = (sub[value_col] * scale).where(enough)
        low, high = (sub[low_col] * scale).where(enough), (sub[high_col] * scale).where(enough)
        if dimension == "hour":
            x = np.array(classes, dtype=float)
            ax.fill_between(x, low, high, color=fill, alpha=0.55 if layout == "panels" else 0.35, lw=0)
            ax.plot(x, value, color=line, lw=2, marker="o", ms=3.5, label=method)
            ax.set_xticks(range(0, 24, 2))
            ax.set_xticklabels([f"{h}h" for h in range(0, 24, 2)])
            ax.set_xlim(-0.5, 23.5)
            ax.set_xlabel("heure de fin de la leg (classe ±30 min)")
        else:
            n_bars = len(methods) if layout == "overlay" else 1
            bar_width = 0.8 / n_bars
            offset = (index - (n_bars - 1) / 2) * bar_width if layout == "overlay" else 0.0
            x = np.arange(len(classes)) + offset
            error = np.vstack([(value - low).clip(lower=0).fillna(0), (high - value).clip(lower=0).fillna(0)])
            ax.bar(x, value.fillna(0), bar_width * 0.92, color=fill, edgecolor=line, lw=1,
                   yerr=error, ecolor=line, capsize=3, label=method)
            for xi, (v, n) in enumerate(zip(value, sub[n_col])):
                ax.text(x[xi], 0, f"n={int(n) if pd.notna(n) else 0}", ha="center", va="bottom",
                        fontsize=7, color="#6b7280", rotation=90 if layout == "overlay" else 0)
            ax.set_xticks(np.arange(len(classes)))
            ax.set_xticklabels([str(c).replace(" ", "\n", 1) for c in classes], fontsize=8)
        if layout == "panels":
            ax.set_title(method, fontsize=10, color=line)
        _style(ax)
    axes[0].set_ylabel(ylabel)
    axes[0].set_ylim(bottom=0)
    if layout == "overlay" and len(methods) > 1:
        handles = [Patch(facecolor=colors[m][0], edgecolor=colors[m][1], label=m) for m in methods]
        axes[0].legend(handles=handles, frameon=False, fontsize=9, loc="lower center",
                       bbox_to_anchor=(0.5, 1.0), ncol=len(handles))
    band = "IC 95 %" if indicator == "share" else "intervalle interquartile"
    dimension_label = "par heure" if dimension == "hour" else "par motif"
    fig.suptitle(
        f"{TITLES[indicator]} {dimension_label}" + (f" (cruising ≥ {min_duration_s:g} s)" if min_duration_s else "") + "\n"
        f"{'bande' if dimension == 'hour' else 'barres d’erreur'} : {band} — classes de moins de {min_legs} legs non tracées",
        fontsize=10,
    )
    fig.tight_layout()
    return fig


REFERENCE_COLOR = "#9ca3af"


def plot_composition(stats: pd.DataFrame, dimension: str, *, layout: str = "panels", show_reference: bool = True,
                     min_duration_s: float = 0.0):
    """Répartition des legs avec cruising par heure (barres / courbes) ou par motif (barres)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    methods = list(dict.fromkeys(stats["methode"]))
    colors = method_colors(methods)
    classes = list(dict.fromkeys(stats["classe"]))
    n_axes = len(methods) if layout == "panels" else 1
    width = (6.5 if dimension == "hour" else 7.0) if layout == "panels" else 10.0
    fig, axes = plt.subplots(1, n_axes, figsize=(width * n_axes, 4.6), sharey=True, squeeze=False)
    axes = axes[0]
    x_base = np.arange(len(classes), dtype=float)

    for index, method in enumerate(methods):
        ax = axes[index if layout == "panels" else 0]
        fill, line = colors[method]
        sub = stats[stats["methode"] == method].set_index("classe").reindex(classes)
        share = sub["part_cruising"] * 100
        error = np.vstack([(share - sub["part_ic95_bas"] * 100).clip(lower=0).fillna(0),
                           (sub["part_ic95_haut"] * 100 - share).clip(lower=0).fillna(0)])
        overlay_hour = layout == "overlay" and dimension == "hour"
        if overlay_hour:
            ax.plot(x_base, share, color=line, lw=2, marker="o", ms=3.5)
            ax.fill_between(x_base, sub["part_ic95_bas"] * 100, sub["part_ic95_haut"] * 100, color=fill, alpha=0.35, lw=0)
        else:
            n_bars = len(methods) if layout == "overlay" else 1
            bar_width = 0.8 / n_bars
            offset = (index - (n_bars - 1) / 2) * bar_width if layout == "overlay" else 0.0
            ax.bar(x_base + offset, share.fillna(0), bar_width * 0.92, color=fill, edgecolor=line, lw=1,
                   yerr=error, ecolor=line, capsize=2 if dimension == "hour" else 3)
        if show_reference and (layout == "panels" or index == 0):
            reference = sub["part_reference"] * 100
            if dimension == "hour":
                ax.step(x_base, reference, where="mid", color=REFERENCE_COLOR, lw=1.3, ls="--")
            else:
                ax.scatter(x_base, reference, marker="D", s=22, color=REFERENCE_COLOR, zorder=3)
        if layout == "panels":
            total = int(sub["n_total_cruising"].dropna().iloc[0]) if sub["n_total_cruising"].notna().any() else 0
            ax.set_title(f"{method} (n = {total:,} legs avec cruising)".replace(",", "'"), fontsize=10, color=line)
        if dimension == "hour":
            ax.set_xticks(range(0, 24, 2))
            ax.set_xticklabels([f"{h}h" for h in range(0, 24, 2)])
            ax.set_xlim(-0.6, 23.6)
            ax.set_xlabel("heure de fin de la leg (classe ±30 min)")
        else:
            ax.set_xticks(x_base)
            ax.set_xticklabels([str(c).replace(" ", "\n", 1) for c in classes], fontsize=8)
        _style(ax)
    axes[0].set_ylabel("% des legs avec cruising")
    axes[0].set_ylim(bottom=0)

    handles = []
    if layout == "overlay":
        handles += [Patch(facecolor=colors[m][0], edgecolor=colors[m][1], label=m) for m in methods]
    if show_reference:
        handles.append(
            Line2D([], [], color=REFERENCE_COLOR, ls="--", lw=1.3, label="toutes les legs évaluées")
            if dimension == "hour"
            else Line2D([], [], color=REFERENCE_COLOR, marker="D", ls="", label="toutes les legs évaluées")
        )
    if handles:
        axes[0].legend(handles=handles, frameon=False, fontsize=9, loc="lower center",
                       bbox_to_anchor=(0.5, 1.0), ncol=len(handles))
    threshold = f" (cruising ≥ {min_duration_s:g} s)" if min_duration_s else ""
    dimension_label = "par heure de fin de leg" if dimension == "hour" else "par motif"
    fig.suptitle(
        f"Répartition des legs avec cruising {dimension_label}{threshold}\n"
        "somme = 100 % par méthode ; barres d’erreur / bande : IC 95 % ; gris : même répartition sur toutes les legs évaluées",
        fontsize=10,
    )
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Histogramme des durées avec filtres (questionnaire, motif)
# --------------------------------------------------------------------------- #


def load_duration_table(config: dict[str, Any], method: str = "relut", *, sample_run_id: str | None = None) -> pd.DataFrame:
    """Legs évaluées par la méthode, avec durée, motifs et variables de stationnement du questionnaire."""
    import pyarrow.parquet as pq

    spec = METHODS[method]
    enriched = set(pq.read_schema(table_path(config, "legs_enriched")).names)
    wanted = ["leg_id", "purpose_agg", "trip_purpose", "space_home", "space_work", "user_id_fors"]
    columns = {
        "legs_enriched": [c for c in wanted if c in enriched],
        spec.table: [spec.status, spec.detected, spec.duration, spec.distance],
    }
    legs = load_leg_table(config, [spec.table], columns=columns, sample_run_id=sample_run_id, verbose=False)
    legs = legs[legs[spec.status].eq("evaluated")].copy()
    legs["duree_min"] = legs[spec.duration] / 60
    legs["cruising"] = legs[spec.detected].astype("boolean").fillna(False).astype(bool)
    return legs


def filter_legs(
    legs: pd.DataFrame,
    *,
    parking: dict[str, list[str]] | None = None,
    parking_mode: str = "and",
    purposes: list[str] | None = None,
    purpose_column: str = "purpose_agg",
) -> tuple[pd.DataFrame, str]:
    """Restreint aux catégories de stationnement demandées et aux motifs demandés.

    ``parking`` : ``{"space_home": [...], "space_work": [...]}`` ; plusieurs
    variables sont combinées en ``and`` (toutes vérifiées) ou ``or`` (au moins une).
    Retourne les legs filtrées et une description lisible du filtre.
    """
    mask = pd.Series(True, index=legs.index)
    labels = []
    if parking:
        conditions = []
        for variable, categories in parking.items():
            if variable not in legs:
                raise KeyError(f"{variable!r} absente de legs_enriched (questionnaire non joint ?)")
            conditions.append(legs[variable].isin(categories))
            labels.append(f"{variable} ∈ {{{', '.join(categories)}}}")
        combined = conditions[0]
        for condition in conditions[1:]:
            combined = combined & condition if parking_mode == "and" else combined | condition
        mask &= combined
        if len(conditions) > 1:
            labels = [(" ET " if parking_mode == "and" else " OU ").join(labels)]
    if purposes:
        mask &= legs[purpose_column].isin(purposes)
        labels.append(f"{purpose_column} ∈ {{{', '.join(purposes)}}}")
    return legs[mask], " ; ".join(labels) or "aucun filtre"


def plot_duration_histogram(
    legs: pd.DataFrame,
    *,
    selection: pd.DataFrame,
    selection_label: str,
    group_by: str | None = None,
    include_no_cruising: bool = False,
    min_duration_s: float = 0.0,
    max_minutes: float = 15.0,
    bin_minutes: float = 0.5,
    show_reference: bool = True,
):
    """Histogramme (en % des legs) des durées de cruising prédites.

    Par défaut parmi les legs avec cruising (``include_no_cruising=False``) ;
    ``group_by`` superpose un histogramme par modalité (ex. ``space_home``) ;
    la référence grise est l'ensemble des legs évaluées, sans filtre.
    """
    import matplotlib.pyplot as plt

    def keep(frame: pd.DataFrame) -> pd.Series:
        minutes = frame["duree_min"]
        if not include_no_cruising:
            minutes = minutes[frame["cruising"] & frame["duree_min"].ge(min_duration_s / 60)]
        return minutes.dropna()

    bins = np.arange(0, max_minutes + bin_minutes, bin_minutes)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    rows = []
    groups = [(selection_label if group_by is None else None, selection)]
    if group_by is not None:
        groups = [(str(value), selection[selection[group_by].eq(value)]) for value in selection[group_by].dropna().unique()]
        groups.sort(key=lambda item: -len(item[1]))
    # Couleurs fixes pour les catégories de stationnement (stables d'un run à l'autre)
    fixed = {"Pas de place dédiée, besoin de cruise": PALETTE[0], "Pas de place dédiée, pas besoin de cruise": PALETTE[1],
             "Aucun": PALETTE[2], "Place dédiée": PALETTE[3]}
    spare = iter([c for c in PALETTE if c not in fixed.values()] or PALETTE)
    for index, (label, frame) in enumerate(groups):
        minutes = keep(frame)
        fill, line = fixed.get(label) or (PALETTE[index % len(PALETTE)] if group_by in (None,) or group_by not in ("space_home", "space_work") else next(spare, PALETTE[index % len(PALETTE)]))
        if len(minutes):
            weights = np.full(len(minutes), 100 / len(minutes))
            ax.hist(minutes.clip(upper=max_minutes), bins=bins, weights=weights, color=fill, alpha=0.75 if group_by else 0.9,
                    edgecolor=line, linewidth=0.6, histtype="stepfilled" if group_by else "bar",
                    label=f"{label} (n = {len(minutes)}, médiane {minutes.median():.2f} min)")
            ax.axvline(minutes.median(), color=line, lw=1.4, ls=":")
        rows.append({"groupe": label, "n_legs": int(len(frame)), "n_legs_histogramme": int(len(minutes)),
                     "duree_moyenne_min": minutes.mean(), "duree_mediane_min": minutes.median(),
                     "duree_q25_min": minutes.quantile(0.25), "duree_q75_min": minutes.quantile(0.75)})
    if show_reference:
        minutes = keep(legs)
        if len(minutes):
            ax.hist(minutes.clip(upper=max_minutes), bins=bins, weights=np.full(len(minutes), 100 / len(minutes)),
                    histtype="step", color=REFERENCE_COLOR, lw=1.4, ls="--",
                    label=f"toutes les legs évaluées (n = {len(minutes)}, médiane {minutes.median():.2f} min)")
        rows.append({"groupe": "référence : toutes les legs évaluées", "n_legs": int(len(legs)),
                     "n_legs_histogramme": int(len(minutes)), "duree_moyenne_min": minutes.mean(),
                     "duree_mediane_min": minutes.median(), "duree_q25_min": minutes.quantile(0.25),
                     "duree_q75_min": minutes.quantile(0.75)})
    population = "legs évaluées (0 = pas de cruising)" if include_no_cruising else (
        "legs avec cruising" + (f" ≥ {min_duration_s:g} s" if min_duration_s else ""))
    ax.set_xlabel("durée de cruising prédite (min) — trait pointillé : médiane")
    ax.set_ylabel(f"% des {population}")
    ax.set_title(f"Durées de cruising prédites — filtre : {selection_label}", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    return fig, pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Point d'entrée
# --------------------------------------------------------------------------- #


def run_indicators(config: dict[str, Any]) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    cfg = config.get("indicators", {}) or {}
    methods = list(cfg.get("methods", ["relut"]))
    layout = cfg.get("layout", "panels")
    if layout not in ("panels", "overlay"):
        raise ValueError("indicators.layout doit valoir 'panels' ou 'overlay'.")
    min_legs = int(cfg.get("min_legs_per_class", 10))
    purpose_column = cfg.get("purpose_column", "purpose_agg")
    min_duration_s = float(cfg.get("min_cruising_duration_s") or 0)
    show_reference = bool(cfg.get("show_reference", True))
    timezone = cfg.get("local_timezone") or (config.get("leg_enrichment") or {}).get("local_timezone", "Europe/Zurich")

    ensure_output_dirs(config)
    figures_dir = output_path(config, "figures", "indicators")
    reports_dir = output_path(config, "reports", "indicators")
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    legs, available = load_indicator_table(config, methods, sample_run_id=cfg.get("sample_run_id"))
    legs["heure"] = hour_bins(legs["leg_finished_at"], timezone)
    legs["motif"] = legs[purpose_column].fillna("non renseigné")
    purpose_order = [p for p in PURPOSE_ORDER if p in set(legs["motif"])] + sorted(
        set(legs["motif"]) - set(PURPOSE_ORDER)
    )

    tables = {
        "hour": indicator_stats(legs, available, "heure", list(range(24)), min_duration_s),
        "purpose": indicator_stats(legs, available, "motif", purpose_order, min_duration_s),
    }
    compositions = {
        "hour": composition_stats(legs, available, "heure", list(range(24)), min_duration_s),
        "purpose": composition_stats(legs, available, "motif", purpose_order, min_duration_s),
    }
    outputs: dict[str, Any] = {"figures": {}, "tables": {}}
    for dimension, stats in tables.items():
        name = "par_heure" if dimension == "hour" else "par_motif"
        path = reports_dir / f"41_indicateurs_{name}.csv"
        stats.rename(columns={"classe": "heure" if dimension == "hour" else "motif"}).to_csv(path, index=False)
        outputs["tables"][dimension] = str(path)
        for indicator in INDICATORS:
            fig = plot_indicator(stats, indicator, dimension, layout=layout, min_legs=min_legs, min_duration_s=min_duration_s)
            figure_path = figures_dir / f"41_{indicator}_{name}.png"
            fig.savefig(figure_path, dpi=150)
            plt.close(fig)
            outputs["figures"][f"{indicator}_{dimension}"] = str(figure_path)

        composition = compositions[dimension]
        path = reports_dir / f"41_repartition_cruising_{name}.csv"
        composition.rename(columns={"classe": "heure" if dimension == "hour" else "motif"}).to_csv(path, index=False)
        outputs["tables"][f"composition_{dimension}"] = str(path)
        fig = plot_composition(composition, dimension, layout=layout, show_reference=show_reference,
                               min_duration_s=min_duration_s)
        figure_path = figures_dir / f"41_repartition_cruising_{name}.png"
        fig.savefig(figure_path, dpi=150)
        plt.close(fig)
        outputs["figures"][f"composition_{dimension}"] = str(figure_path)

    summary = {
        "methods": [m.label for m in available],
        "layout": layout,
        "min_legs_per_class": min_legs,
        "min_cruising_duration_s": min_duration_s,
        "purpose_column": purpose_column,
        "local_timezone": timezone,
        "hour_reference": "fin de la leg, classe arrondie à l'heure la plus proche (h-30 min à h+29:59)",
        "population": "legs évaluées par la méthode ; durées et distances parmi les legs avec cruising détecté",
        "legs_evaluated": {m.label: int(legs[m.status].eq("evaluated").sum()) for m in available},
        **outputs,
    }
    write_json(reports_dir / "41_indicateurs_summary.json", summary)
    return {"summary": summary, "tables": tables, "compositions": compositions}
