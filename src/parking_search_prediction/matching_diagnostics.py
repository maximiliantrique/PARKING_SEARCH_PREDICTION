"""Contrôle de l'appariement waypoints ↔ legs, en relisant les sorties de l'étape 01 (sans relancer).

Sources
-------
- ``data/legs_enriched.parquet`` : toutes les legs gardées par l'étape 01 ;
- ``data/waypoint_leg_assignments.parquet`` : tous les waypoints **nettoyés**,
  avec leur leg et leur statut d'appariement ;
- optionnel : le dataset Parquet des waypoints **bruts** (avant nettoyage), lu
  colonne par colonne, pour savoir si un participant avait des points un jour donné.

Les dates sont des jours **UTC** (comme les horodatages des fichiers).

Motifs de non-appariement d'une leg (dans cet ordre)
----------------------------------------------------
- ``user_without_waypoints`` : le participant n'a aucun waypoint ;
- ``no_raw_waypoints_that_day`` : aucun waypoint brut ce jour-là (période non couverte, application coupée) ;
- ``all_waypoints_cleaned_that_day`` : des points bruts ce jour-là, mais aucun après nettoyage (précision…) ;
- ``only_overlapping_points`` : des points dans l'intervalle de la leg, tous marqués ``overlapping_tracks`` ;
- ``no_waypoint_in_interval`` : des points ce jour-là, mais aucun entre le début et la fin de la leg (trou d'enregistrement, leg très courte).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import output_path

REASONS = [
    "user_without_waypoints",
    "no_raw_waypoints_that_day",
    "all_waypoints_cleaned_that_day",
    "only_overlapping_points",
    "no_waypoint_in_interval",
]


def _parts(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(path.glob("*.parquet"))
    if path.exists():
        return [path]
    raise FileNotFoundError(f"{path} introuvable : lancer d'abord l'étape 01.")


def scan_assignments(config: dict[str, Any], verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Waypoints nettoyés par (participant, jour) et par leg, lus partie par partie.

    Retourne ``(par_jour, par_leg)`` :
    - par_jour : ``source_user_id, date, clean_points, matched_points`` ;
    - par_leg : ``leg_id, matched_points, overlapping_points``.
    """
    days, legs = [], []
    started = time.perf_counter()
    parts = _parts(output_path(config, "data", "waypoint_leg_assignments.parquet"))
    for number, part in enumerate(parts, start=1):
        df = pd.read_parquet(part, columns=["source_user_id", "timestamp", "leg_id", "leg_match_status"])
        df["date"] = pd.to_datetime(df["timestamp"], utc=True).dt.floor("D")
        df["matched"] = df["leg_match_status"].eq("matched")
        days.append(df.groupby(["source_user_id", "date"]).agg(clean_points=("matched", "size"), matched_points=("matched", "sum")))
        with_leg = df.dropna(subset=["leg_id"])
        legs.append(pd.DataFrame({
            "matched_points": with_leg["matched"].astype(int),
            "overlapping_points": with_leg["leg_match_status"].eq("overlapping_tracks").astype(int),
            "leg_id": with_leg["leg_id"],
        }).groupby("leg_id").sum())
        if verbose:
            print(f"[diagnostic] appariements {number}/{len(parts)} — {time.perf_counter() - started:.0f} s", flush=True)
    per_day = pd.concat(days).groupby(level=[0, 1]).sum().reset_index()
    per_leg = pd.concat(legs).groupby(level=0).sum().rename_axis("leg_id").reset_index()
    return per_day, per_leg


def scan_raw_days(config: dict[str, Any], verbose: bool = True) -> pd.DataFrame | None:
    """Points bruts par (participant, jour), depuis le dataset Parquet (colonnes utilisateur et horodatage seulement)."""
    import pyarrow.parquet as pq

    from .data_prep import normalize_user_ids
    from .waypoint_parquet import NULL_PARTITION, conversion_settings, load_manifest

    try:
        manifest = load_manifest(config)
    except (FileNotFoundError, KeyError):
        return None
    settings = conversion_settings(config)
    user_col, time_col = settings["user_col"], config["raw_waypoints"]["time_col"]
    rows = []
    started = time.perf_counter()
    labels = [label for label in manifest["partitions"] if label != NULL_PARTITION]
    for number, label in enumerate(sorted(labels), start=1):
        path = settings["dataset_dir"] / f"partition={label}" / "part-0.parquet"
        table = pq.read_table(path, columns=[user_col, time_col]).to_pandas()
        # « 2023-05-01 05:06:28 +0000 » : le jour UTC est le début de la chaîne
        table["date"] = pd.to_datetime(table[time_col].astype(str).str[:10], errors="coerce", utc=True)
        counts = table.groupby([user_col, "date"]).size().rename("raw_points").reset_index()
        rows.append(counts)
        if verbose and (number % 8 == 0 or number == len(labels)):
            print(f"[diagnostic] points bruts {number}/{len(labels)} partitions — {time.perf_counter() - started:.0f} s", flush=True)
    raw = pd.concat(rows, ignore_index=True)
    raw["source_user_id"] = normalize_user_ids(raw[user_col].astype(object))
    return raw.groupby(["source_user_id", "date"], as_index=False)["raw_points"].sum()


def load_legs(config: dict[str, Any]) -> pd.DataFrame:
    import pyarrow.parquet as pq

    path = output_path(config, "data", "legs_enriched.parquet")
    if not path.exists():
        raise FileNotFoundError(f"{path} introuvable (enrichissement des legs désactivé ?).")
    columns = [c for c in ("leg_id", "source_user_id", "leg_started_at", "leg_finished_at", "effective_mode")
               if c in pq.read_schema(path).names]
    legs = pd.read_parquet(path, columns=columns)
    legs["date"] = pd.to_datetime(legs["leg_started_at"], utc=True).dt.floor("D")
    return legs


def diagnose_unmatched_legs(
    legs: pd.DataFrame, per_day: pd.DataFrame, per_leg: pd.DataFrame, raw_days: pd.DataFrame | None
) -> pd.DataFrame:
    """Ajoute ``matched`` et, pour les legs non appariées, ``unmatched_reason``."""
    legs = legs.merge(per_leg, on="leg_id", how="left")
    legs[["matched_points", "overlapping_points"]] = legs[["matched_points", "overlapping_points"]].fillna(0).astype(int)
    legs["matched"] = legs["matched_points"] > 0

    users_with_points = set(per_day["source_user_id"]) | (set(raw_days["source_user_id"]) if raw_days is not None else set())
    clean_day = per_day.set_index(["source_user_id", "date"])["clean_points"]
    key = pd.MultiIndex.from_frame(legs[["source_user_id", "date"]])
    clean_that_day = clean_day.reindex(key).fillna(0).to_numpy() > 0
    if raw_days is not None:
        raw_that_day = raw_days.set_index(["source_user_id", "date"])["raw_points"].reindex(key).fillna(0).to_numpy() > 0
    else:
        raw_that_day = clean_that_day  # sans les bruts, « nettoyés » et « absents » ne se distinguent pas

    reason = pd.Series(pd.NA, index=legs.index, dtype="object")
    unmatched = ~legs["matched"]
    rules = [
        ("user_without_waypoints", ~legs["source_user_id"].isin(users_with_points)),
        ("no_raw_waypoints_that_day", ~raw_that_day),
        ("all_waypoints_cleaned_that_day", raw_that_day & ~clean_that_day),
        ("only_overlapping_points", legs["overlapping_points"] > 0),
        ("no_waypoint_in_interval", pd.Series(True, index=legs.index)),
    ]
    for name, mask in rules:
        reason[unmatched & reason.isna() & mask] = name
    legs["unmatched_reason"] = reason
    return legs


def daily_table(legs: pd.DataFrame, per_day: pd.DataFrame, raw_days: pd.DataFrame | None) -> pd.DataFrame:
    """Par jour UTC : legs, legs appariées, waypoints nettoyés, appariés (et bruts si disponibles)."""
    leg_days = legs.groupby("date").agg(legs=("leg_id", "size"), matched_legs=("matched", "sum"))
    point_days = per_day.groupby("date")[["clean_points", "matched_points"]].sum()
    table = leg_days.join(point_days, how="outer")
    if raw_days is not None:
        table = table.join(raw_days.groupby("date")["raw_points"].sum(), how="outer")
    table = table.fillna(0).astype(int).sort_index()
    table["matched_legs_share"] = table["matched_legs"] / table["legs"].where(table["legs"] > 0)
    table["matched_points_share"] = table["matched_points"] / table["clean_points"].where(table["clean_points"] > 0)
    table.index = table.index.tz_convert(None).date if hasattr(table.index, "tz_convert") else table.index
    return table.rename_axis("date_utc")


def window_check(
    daily: pd.DataFrame, date_from: str, date_to: str, *,
    min_matched_legs_share: float | None = None, min_matched_points_share: float | None = None,
) -> dict[str, Any]:
    """Totaux sur les jours ``date_from`` à ``date_to`` inclus ; assert si des seuils sont fixés."""
    days = pd.to_datetime(pd.Series(daily.index)).dt.date
    mask = (days >= pd.Timestamp(date_from).date()) & (days <= pd.Timestamp(date_to).date())
    window = daily[mask.to_numpy()]
    totals = window.drop(columns=["matched_legs_share", "matched_points_share"]).sum()
    result = {
        "date_from": date_from, "date_to": date_to, "days": int(len(window)),
        **{k: int(v) for k, v in totals.items()},
        "matched_legs_share": float(totals["matched_legs"] / totals["legs"]) if totals["legs"] else None,
        "matched_points_share": float(totals["matched_points"] / totals["clean_points"]) if totals["clean_points"] else None,
    }
    if min_matched_legs_share is not None:
        assert result["matched_legs_share"] is not None and result["matched_legs_share"] >= min_matched_legs_share, (
            f"Part de legs appariées {result['matched_legs_share']:.1%} < seuil {min_matched_legs_share:.0%} "
            f"sur {date_from} → {date_to}"
        )
    if min_matched_points_share is not None:
        assert result["matched_points_share"] is not None and result["matched_points_share"] >= min_matched_points_share, (
            f"Part de waypoints appariés {result['matched_points_share']:.1%} < seuil {min_matched_points_share:.0%} "
            f"sur {date_from} → {date_to}"
        )
    return result


def plot_daily_coverage(daily: pd.DataFrame, date_from: str | None = None, date_to: str | None = None):
    """Deux panneaux : legs (toutes / appariées) et waypoints (bruts / nettoyés / appariés) par jour."""
    import matplotlib.pyplot as plt

    from .indicators import PALETTE, _style

    x = pd.to_datetime(pd.Series(daily.index))
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)
    ax = axes[0]
    ax.bar(x, daily["legs"], color=PALETTE[3][0], edgecolor=PALETTE[3][1], lw=0.5, label="legs")
    ax.bar(x, daily["matched_legs"], color=PALETTE[1][0], edgecolor=PALETTE[1][1], lw=0.5, label="legs appariées (≥ 1 waypoint)")
    ax.set_ylabel("legs par jour")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax2 = axes[1]
    if "raw_points" in daily:
        ax2.plot(x, daily["raw_points"] / 1e6, color=PALETTE[2][1], lw=1.6, label="waypoints bruts")
    ax2.plot(x, daily["clean_points"] / 1e6, color=PALETTE[3][1], lw=1.6, label="waypoints nettoyés")
    ax2.plot(x, daily["matched_points"] / 1e6, color=PALETTE[0][1], lw=1.6, label="waypoints appariés")
    ax2.set_ylabel("millions de waypoints")
    ax2.legend(frameon=False, fontsize=8, loc="upper left")
    for axis in axes:
        _style(axis)
        if date_from and date_to:
            axis.axvspan(pd.Timestamp(date_from), pd.Timestamp(date_to) + pd.Timedelta(days=1), color="#e5e7eb", alpha=0.5, lw=0)
    fig.suptitle("Couverture journalière (jours UTC) — legs vs waypoints ; zone grise : fenêtre contrôlée", fontsize=10)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def run_matching_diagnostics(config: dict[str, Any], *, scan_raw: bool = True, recompute: bool = False) -> dict[str, Any]:
    """Scanne les sorties (mise en cache dans ``reports/01_diagnostics/``) et classe les legs non appariées."""
    cache = output_path(config, "reports", "01_diagnostics")
    cache.mkdir(parents=True, exist_ok=True)
    files = {name: cache / f"{name}.parquet" for name in ("per_day", "per_leg", "raw_days")}
    if recompute or not (files["per_day"].exists() and files["per_leg"].exists()):
        per_day, per_leg = scan_assignments(config)
        per_day.to_parquet(files["per_day"], index=False)
        per_leg.to_parquet(files["per_leg"], index=False)
    else:
        per_day, per_leg = pd.read_parquet(files["per_day"]), pd.read_parquet(files["per_leg"])
    raw_days = None
    if scan_raw:
        if recompute or not files["raw_days"].exists():
            raw_days = scan_raw_days(config)
            if raw_days is not None:
                raw_days.to_parquet(files["raw_days"], index=False)
        else:
            raw_days = pd.read_parquet(files["raw_days"])
    legs = diagnose_unmatched_legs(load_legs(config), per_day, per_leg, raw_days)
    unmatched = legs[~legs["matched"]]
    car_modes = set(config["leg_matching"].get("car_modes", ["Mode::Car", "Mode::Ecar"]))
    reasons = pd.DataFrame({
        "legs": unmatched["unmatched_reason"].value_counts(),
        "dont_voiture": unmatched.loc[unmatched["effective_mode"].isin(car_modes), "unmatched_reason"].value_counts(),
    }).reindex(REASONS).fillna(0).astype(int)
    reasons["part_des_non_appariees"] = reasons["legs"] / max(len(unmatched), 1)
    legs.loc[~legs["matched"], ["leg_id", "source_user_id", "date", "effective_mode", "unmatched_reason"]].to_parquet(
        cache / "unmatched_legs.parquet", index=False)
    return {
        "legs": legs, "reasons": reasons, "daily": daily_table(legs, per_day, raw_days),
        "summary": {"legs": int(len(legs)), "matched_legs": int(legs["matched"].sum()),
                    "unmatched_legs": int(len(unmatched)), "raw_scanned": raw_days is not None,
                    "legs_period_utc": [str(legs["date"].min().date()), str(legs["date"].max().date())],
                    "waypoints_period_utc": [str(per_day["date"].min().date()), str(per_day["date"].max().date())]},
    }
