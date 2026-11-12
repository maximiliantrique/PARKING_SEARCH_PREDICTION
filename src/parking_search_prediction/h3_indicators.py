"""Étape 41 (section H3) — indicateurs de cruising et de stationnement par hexagone H3.

Sources
-------
- ``data/relut_predictions.parquet`` (étape 32) : points des segments prédits,
  avec ``y_hat_labels`` (``searching`` / ``driving``) → indicateurs 1 à 4 ;
- ``data/waypoints_caronly.parquet`` (étape 01) : waypoints voiture des legs du
  périmètre → indicateurs 5 et 6 (fin de leg = dernier waypoint voiture, même
  définition que le filtre de périmètre) ;
- optionnel : offre de stationnement (``OTC_PARKING.gpkg``) → couche des
  parkings et places publiques par hexagone.

Indicateurs par hexagone (résolution ``resolution``, 8 par défaut)
------------------------------------------------------------------
1. temps de cruising cumulé (min) ;
2. distance de cruising cumulée (km) ;
3. legs avec cruising dans l'hexagone ;
4. part du temps voiture passée en cruising ;
5. fins de legs (stationnements) ;
6. fins de legs / legs qui traversent l'hexagone.

Chaque point porte le temps et la distance **jusqu'au point suivant** de son
segment (attribués à l'hexagone du point) : la somme sur les points en recherche
redonne la durée et la distance de ``relut_by_leg``. Les intervalles sont
plafonnés à ``max_point_interval_s`` (un trou de signal ne compte pas comme du
temps passé). **Compter des points biaiserait le résultat vers les traces
échantillonnées finement** : on somme donc des durées, pas des points.

Confidentialité et stabilité : une valeur n'est publiée que si au moins
``min_persons`` personnes distinctes **et** ``min_legs`` legs y contribuent ;
sinon elle est masquée (NaN) dans la table et absente de la carte.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ensure_output_dirs, output_path, resolve_path
from .utils import haversine_m, write_json

# Palettes pastel séquentielles (clair → soutenu) par famille d'indicateurs
PALETTES = {
    "rose": ["#fde7ee", "#f9c8d7", "#f2a5bd", "#e47f9f", "#c95a80"],
    "vert": ["#e6f5ea", "#c4e6cf", "#9dd3b0", "#6fbb8d", "#46996a"],
    "jaune": ["#fdf6d8", "#f9e8a6", "#f2d676", "#e3bd4b", "#c39a2a"],
    "bleu": ["#e6f0fb", "#c4daf3", "#9cc0e8", "#6fa0d8", "#4a7fbf"],
}

INDICATORS = {
    # clé : (libellé, unité, palette, colonne des personnes, colonne des legs)
    "cruising_time_min": ("Temps de cruising cumulé", "min", "rose", "n_persons_cruising", "n_legs_cruising"),
    "cruising_distance_km": ("Distance de cruising cumulée", "km", "rose", "n_persons_cruising", "n_legs_cruising"),
    "n_legs_cruising": ("Legs avec cruising dans l'hexagone", "legs", "jaune", "n_persons_cruising", "n_legs_cruising"),
    "cruising_time_share": ("Part du temps voiture passée en cruising", "%", "jaune", "n_persons_car", "n_legs_car"),
    "n_leg_ends": ("Fins de legs (stationnements)", "legs", "bleu", "n_persons_ends", "n_leg_ends"),
    "leg_end_rate": ("Fins de legs / legs qui traversent l'hexagone", "%", "vert", "n_persons_traversing", "n_legs_traversing"),
}


def _settings(config: dict[str, Any]) -> dict[str, Any]:
    cfg = config.get("h3_indicators", {}) or {}
    return {
        "resolution": int(cfg.get("resolution", 8)),
        "min_persons": int(cfg.get("min_persons", 3)),
        "min_legs": int(cfg.get("min_legs", 3)),
        "max_point_interval_s": float(cfg.get("max_point_interval_s", 120)),
        "parking_supply_file": cfg.get("parking_supply_file"),
        "parking_supply_layer": cfg.get("parking_supply_layer"),
        "parking_capacity_column": cfg.get("parking_capacity_column", "nb_pl_publiques"),
        "parking_name_column": cfg.get("parking_name_column", "nom"),
    }


def _h3():
    try:
        import h3  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Le paquet `h3` est nécessaire : python -m pip install -r requirements.txt") from exc
    return h3


def cells_for(lat: np.ndarray, lon: np.ndarray, resolution: int) -> np.ndarray:
    """Cellule H3 de chaque point (calculée une fois par position distincte arrondie à 1e-6°)."""
    h3 = _h3()
    to_cell = h3.latlng_to_cell if hasattr(h3, "latlng_to_cell") else h3.geo_to_h3
    keys = pd.MultiIndex.from_arrays([np.round(lat, 6), np.round(lon, 6)])
    unique = keys.unique()
    lookup = pd.Series([to_cell(a, b, resolution) for a, b in unique], index=unique)
    return lookup.reindex(keys).to_numpy()


def _parts(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(path.glob("*.parquet"))
    if path.exists():
        return [path]
    raise FileNotFoundError(f"{path} introuvable.")


# --------------------------------------------------------------------------- #
# Agrégation
# --------------------------------------------------------------------------- #


def aggregate_cruising(config: dict[str, Any], settings: dict[str, Any]) -> pd.DataFrame:
    """Indicateurs 1 à 4, à partir des prédictions point par point (lues partie par partie)."""
    cap = settings["max_point_interval_s"]
    sums, pairs = [], []
    columns = ["trip_candidate_id", "leg_id", "source_user_id", "timestamp", "lat", "lon", "y_hat_labels"]
    for part in _parts(output_path(config, "data", "relut_predictions.parquet")):
        df = pd.read_parquet(part, columns=columns)
        if df.empty:
            continue
        df = df.sort_values(["trip_candidate_id", "timestamp"], kind="mergesort").reset_index(drop=True)
        grouped = df.groupby("trip_candidate_id", sort=False)
        next_time = grouped["timestamp"].shift(-1)
        df["dt_s"] = (next_time - df["timestamp"]).dt.total_seconds().clip(upper=cap).fillna(0.0)
        df["step_m"] = pd.Series(
            haversine_m(df["lat"], df["lon"], grouped["lat"].shift(-1), grouped["lon"].shift(-1)), index=df.index
        ).fillna(0.0)
        df["searching"] = df["y_hat_labels"].eq("searching")
        df["hex"] = cells_for(df["lat"].to_numpy(), df["lon"].to_numpy(), settings["resolution"])
        df["cruise_s"] = df["dt_s"].where(df["searching"], 0.0)
        df["cruise_m"] = df["step_m"].where(df["searching"], 0.0)
        sums.append(df.groupby("hex")[["dt_s", "cruise_s", "cruise_m"]].sum())
        pairs.append(df[["hex", "leg_id", "source_user_id", "searching"]].drop_duplicates())

    if not sums:
        return pd.DataFrame()
    totals = pd.concat(sums).groupby(level=0).sum()
    pairs = pd.concat(pairs, ignore_index=True)
    car = pairs.drop_duplicates(["hex", "leg_id"]).groupby("hex").agg(
        n_legs_car=("leg_id", "nunique"), n_persons_car=("source_user_id", "nunique")
    )
    cruising = pairs[pairs["searching"]].groupby("hex").agg(
        n_legs_cruising=("leg_id", "nunique"), n_persons_cruising=("source_user_id", "nunique")
    )
    out = totals.join(car, how="outer").join(cruising, how="left")
    out["cruising_time_min"] = out["cruise_s"] / 60
    out["cruising_distance_km"] = out["cruise_m"] / 1000
    out["car_time_min"] = out["dt_s"] / 60
    out["cruising_time_share"] = np.where(out["dt_s"] > 0, out["cruise_s"] / out["dt_s"], np.nan)
    for column in ("n_legs_cruising", "n_persons_cruising"):
        out[column] = out[column].fillna(0).astype(int)
    return out.drop(columns=["dt_s", "cruise_s", "cruise_m"])


def aggregate_parking(config: dict[str, Any], settings: dict[str, Any]) -> pd.DataFrame:
    """Indicateurs 5 et 6, à partir des waypoints voiture de l'étape 01."""
    ends, pairs = [], []
    for part in _parts(output_path(config, "data", "waypoints_caronly.parquet")):
        df = pd.read_parquet(part, columns=["leg_id", "source_user_id", "timestamp", "lat", "lon"])
        if df.empty:
            continue
        df["hex"] = cells_for(df["lat"].to_numpy(), df["lon"].to_numpy(), settings["resolution"])
        pairs.append(df[["hex", "leg_id", "source_user_id"]].drop_duplicates())
        last = df.sort_values(["leg_id", "timestamp"], kind="mergesort").groupby("leg_id").tail(1)
        ends.append(last[["hex", "leg_id", "source_user_id"]])
    if not pairs:
        return pd.DataFrame()
    pairs = pd.concat(pairs, ignore_index=True).drop_duplicates(["hex", "leg_id"])
    # Une leg peut être coupée entre deux tranches ? Non : un utilisateur est entier dans une tranche.
    ends = pd.concat(ends, ignore_index=True).drop_duplicates("leg_id")
    traversing = pairs.groupby("hex").agg(n_legs_traversing=("leg_id", "nunique"), n_persons_traversing=("source_user_id", "nunique"))
    parked = ends.groupby("hex").agg(n_leg_ends=("leg_id", "nunique"), n_persons_ends=("source_user_id", "nunique"))
    out = traversing.join(parked, how="outer")
    for column in ("n_leg_ends", "n_persons_ends", "n_legs_traversing", "n_persons_traversing"):
        out[column] = out[column].fillna(0).astype(int)
    out["leg_end_rate"] = np.where(out["n_legs_traversing"] > 0, out["n_leg_ends"] / out["n_legs_traversing"], np.nan)
    return out


def aggregate_supply(config: dict[str, Any], settings: dict[str, Any]):
    """Parkings (points) et places publiques par hexagone, si un fichier d'offre est configuré."""
    if not settings["parking_supply_file"]:
        return None, None
    import geopandas as gpd

    path = resolve_path(config, settings["parking_supply_file"])
    if not path.exists():
        print(f"[h3] offre de stationnement introuvable : {path} — couche ignorée.")
        return None, None
    parkings = gpd.read_file(path, layer=settings["parking_supply_layer"])
    points = parkings.copy()
    points["geometry"] = points.geometry.centroid if points.geom_type.str.contains("Polygon").any() else points.geometry
    points = points.explode(index_parts=False).to_crs("EPSG:4326")
    points = points[~points.geometry.is_empty]
    points["lat"], points["lon"] = points.geometry.y, points.geometry.x
    points["hex"] = cells_for(points["lat"].to_numpy(), points["lon"].to_numpy(), settings["resolution"])
    capacity = settings["parking_capacity_column"]
    points[capacity] = pd.to_numeric(points.get(capacity), errors="coerce")
    supply = points.groupby("hex").agg(n_parkings=("hex", "size"), parking_public_spaces=(capacity, "sum"))
    return points.drop(columns="geometry"), supply


def apply_thresholds(table: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    """Masque (NaN) chaque indicateur là où trop peu de personnes ou de legs y contribuent."""
    table = table.copy()
    for key, (_, _, _, persons_col, legs_col) in INDICATORS.items():
        if key not in table:
            continue
        ok = table[persons_col].fillna(0).ge(settings["min_persons"]) & table[legs_col].fillna(0).ge(settings["min_legs"])
        table[f"{key}_published"] = ok
        table[key] = table[key].where(ok)
    return table


# --------------------------------------------------------------------------- #
# Carte HTML
# --------------------------------------------------------------------------- #


def _classes(values: pd.Series, n: int = 5) -> np.ndarray:
    """Bornes par quantiles (classes d'effectifs comparables), dédoublonnées."""
    edges = np.unique(np.nanquantile(values.to_numpy(dtype=float), np.linspace(0, 1, n + 1)))
    return edges if len(edges) >= 2 else np.array([values.min(), values.max() + 1e-9])


def _format(value: float, unit: str) -> str:
    if unit == "%":
        return f"{value * 100:.1f} %"
    if unit == "legs":
        return f"{value:.0f}"
    return f"{value:.1f} {unit}" if value < 100 else f"{value:,.0f} {unit}".replace(",", "'")


def write_h3_map(table: pd.DataFrame, path: Path, settings: dict[str, Any], *, title: str,
                 parking_points: pd.DataFrame | None = None) -> Path:
    """Carte folium : une couche par indicateur (cases à cocher), légende propre à chaque couche."""
    import folium
    from branca.element import Element, MacroElement, Template

    h3 = _h3()
    boundary = h3.cell_to_boundary if hasattr(h3, "cell_to_boundary") else h3.h3_to_geo_boundary
    shapes = {cell: [[lng, lat] for lat, lng in boundary(cell)] for cell in table.index}
    centre = np.mean([np.mean(np.array(s), axis=0) for s in shapes.values()], axis=0) if shapes else [6.14, 46.2]

    fmap = folium.Map(location=[centre[1], centre[0]], zoom_start=11, tiles=None, control_scale=True)
    from .map_style import add_basemaps

    add_basemaps(fmap)

    legends, layer_names = [], []
    layers = [k for k in INDICATORS if k in table] + (["parking_public_spaces"] if "parking_public_spaces" in table else [])
    for position, key in enumerate(layers):
        if key == "parking_public_spaces":
            label, unit, palette = "Places publiques en parking (offre OTC)", "places", "bleu"
        else:
            label, unit, palette, _, _ = INDICATORS[key]
        values = table[key].dropna()
        values = values[values > 0] if key in ("n_leg_ends", "parking_public_spaces") else values
        if values.empty:
            continue
        edges = _classes(values)
        colors = PALETTES[palette][-(len(edges) - 1):] if len(edges) - 1 < 5 else PALETTES[palette]
        name = f"{label}"
        group = folium.FeatureGroup(name=name, overlay=True, show=position == 0)
        features = []
        for cell, value in values.items():
            index = min(int(np.searchsorted(edges, value, side="right")) - 1, len(colors) - 1)
            row = table.loc[cell]
            tooltip = f"<b>{label}</b> : {_format(value, unit)}"
            if key in INDICATORS:
                persons_col, legs_col = INDICATORS[key][3], INDICATORS[key][4]
                tooltip += f"<br>{int(row[legs_col])} legs, {int(row[persons_col])} personnes"
            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [shapes[cell] + [shapes[cell][0]]]},
                "properties": {"color": colors[max(index, 0)], "tooltip": tooltip, "hex": cell},
            })
        folium.GeoJson(
            {"type": "FeatureCollection", "features": features},
            style_function=lambda f: {"fillColor": f["properties"]["color"], "color": "#ffffff", "weight": 0.6, "fillOpacity": 0.78},
            highlight_function=lambda f: {"weight": 2, "color": "#374151"},
            tooltip=folium.GeoJsonTooltip(fields=["tooltip", "hex"], aliases=["", "hexagone"], labels=True, sticky=True),
        ).add_to(group)
        group.add_to(fmap)
        layer_names.append(name)
        rows = "".join(
            f'<div><span style="display:inline-block;width:14px;height:12px;background:{c};margin-right:6px;'
            f'border:1px solid #ddd"></span>{_format(edges[i], unit)} – {_format(edges[i + 1], unit)}</div>'
            for i, c in enumerate(colors[: len(edges) - 1])
        )
        legends.append(
            f'<div class="h3-legend" data-layer="{name}" style="display:{"block" if position == 0 else "none"}">'
            f"<b>{label}</b>{rows}</div>"
        )

    if parking_points is not None and len(parking_points):
        group = folium.FeatureGroup(name="Parkings publics (OTC)", overlay=True, show=False)
        capacity = settings["parking_capacity_column"]
        name_col = settings["parking_name_column"]
        for row in parking_points.itertuples():
            places = getattr(row, capacity, np.nan)
            radius = 3 + 2 * np.sqrt(places / 50) if pd.notna(places) else 3
            folium.CircleMarker(
                [row.lat, row.lon], radius=float(min(radius, 14)), color="#4a7fbf", weight=1,
                fill=True, fill_color="#9cc0e8", fill_opacity=0.85,
                tooltip=f"{getattr(row, name_col, '')} — {places:.0f} places publiques" if pd.notna(places) else str(getattr(row, name_col, "")),
            ).add_to(group)
        group.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.get_root().html.add_child(Element(
        '<style>.h3-legend{position:fixed;bottom:24px;left:12px;z-index:9999;background:#ffffffee;padding:8px 10px;'
        'border-radius:6px;font:12px sans-serif;box-shadow:0 1px 4px #0003;margin-top:6px}'
        '#h3-legends{position:fixed;bottom:24px;left:12px;z-index:9999;display:flex;flex-direction:column-reverse;gap:6px}'
        '#h3-legends .h3-legend{position:static}'
        '.h3-title{position:fixed;top:10px;left:55px;z-index:9999;background:#ffffffee;padding:6px 10px;border-radius:6px;'
        'font:13px sans-serif;box-shadow:0 1px 4px #0003;max-width:55%}</style>'
        f'<div class="h3-title">{title}</div><div id="h3-legends">{"".join(legends)}</div>'
    ))
    # Afficher la légende d'une couche quand on la coche, la masquer quand on la décoche
    macro = MacroElement()
    macro._template = Template(
        "{% macro script(this, kwargs) %}"
        "var m = {{ this._parent.get_name() }};"
        "function legend(name, show){document.querySelectorAll('.h3-legend').forEach(function(d){"
        "if(d.dataset.layer===name){d.style.display=show?'block':'none';}});}"
        "m.on('overlayadd', function(e){legend(e.name, true);});"
        "m.on('overlayremove', function(e){legend(e.name, false);});"
        "{% endmacro %}"
    )
    fmap.add_child(macro)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(path)
    return path


# --------------------------------------------------------------------------- #
# Point d'entrée
# --------------------------------------------------------------------------- #


def run_h3_indicators(config: dict[str, Any]) -> dict[str, Any]:
    ensure_output_dirs(config)
    settings = _settings(config)
    cruising = aggregate_cruising(config, settings)
    parking = aggregate_parking(config, settings)
    points, supply = aggregate_supply(config, settings)
    table = cruising.join(parking, how="outer")
    if supply is not None:
        table = table.join(supply, how="outer")
    table.index.name = "h3_cell"
    raw = table.copy()
    table = apply_thresholds(table, settings)

    run_path = output_path(config, "reports", "prediction_global_summary.json")
    scope = json.loads(run_path.read_text(encoding="utf-8")).get("scope") if run_path.exists() else None
    reports = output_path(config, "reports", "h3")
    reports.mkdir(parents=True, exist_ok=True)
    table.reset_index().to_csv(reports / f"41_h3_r{settings['resolution']}_indicateurs.csv", index=False)

    title = (
        f"Indicateurs H3 (résolution {settings['resolution']}) — prédictions : {scope or 'run inconnu'} — "
        f"publiés si ≥ {settings['min_persons']} personnes et ≥ {settings['min_legs']} legs"
    )
    map_path = write_h3_map(
        table, output_path(config, "maps", f"41_h3_r{settings['resolution']}_indicateurs.html"), settings,
        title=title, parking_points=points,
    )
    published = {key: int(table[f"{key}_published"].sum()) for key in INDICATORS if f"{key}_published" in table}
    summary = {
        "resolution": settings["resolution"],
        "min_persons": settings["min_persons"],
        "min_legs": settings["min_legs"],
        "max_point_interval_s": settings["max_point_interval_s"],
        "predictions_scope": scope,
        "hexagons_total": int(len(table)),
        "hexagons_published_by_indicator": published,
        "totals_before_masking": {
            "cruising_time_h": float(raw["cruising_time_min"].sum() / 60) if "cruising_time_min" in raw else None,
            "cruising_distance_km": float(raw["cruising_distance_km"].sum()) if "cruising_distance_km" in raw else None,
            "leg_ends": int(raw["n_leg_ends"].sum()) if "n_leg_ends" in raw else None,
        },
        "parking_supply": settings["parking_supply_file"] if supply is not None else None,
        "map": str(map_path),
        "table": str(reports / f"41_h3_r{settings['resolution']}_indicateurs.csv"),
    }
    write_json(reports / f"41_h3_r{settings['resolution']}_summary.json", summary)
    return {"summary": summary, "table": table}
