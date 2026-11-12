"""Map matching des traces de la fenêtre de destination sur le réseau routier (service OSRM ``/match``).

Pourquoi pas ``mappymatch.OsrmMatcher`` directement ?
-----------------------------------------------------
``OsrmMatcher`` (mappymatch 0.5) interroge le même service OSRM ``/match``, mais
ne renvoie **ni la géométrie ni la longueur** du trajet apparié (``geom=None``,
``distance=inf``, marqué TODO dans le code), ne garde que le premier tronçon
apparié, n'envoie ni horodatages ni rayons GPS et décale les correspondances
point ↔ segment d'un indice. On appelle donc ``/match`` nous-mêmes, avec les
options utiles : ``timestamps``, ``radiuses`` (précision GPS), ``gaps=split``,
``geometries=geojson``, ``overview=full``. On obtient la longueur appariée, la
géométrie et, pour chaque waypoint, sa position recalée sur la route.

Points envoyés : le point d'entrée interpolé (même origine que le routage), puis
les waypoints de la fenêtre, sous-échantillonnés à ``max_points`` (limite par
défaut d'OSRM : 100 coordonnées par requête), premier et dernier toujours gardés.

Sorties : ``data/map_matching_by_leg.parquet`` (une ligne par leg appariée),
``data/map_matching_geometries.parquet`` (GeoParquet) et
``data/map_matching_tracepoints.parquet`` (waypoints bruts et recalés, pour la carte).
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from .config import output_path

DEFAULTS = {
    "osrm_url": "http://127.0.0.1:5001",   # port du conteneur OSRM (docker compose ps)
    "profile": "driving",                  # segment d'URL ; un serveur OSRM sert un seul profil
    "max_points": 100,                     # limite --max-matching-size d'OSRM (100 par défaut)
    "radius_min_m": 5.0,                   # rayon de recherche = précision GPS, bornée
    "radius_max_m": 50.0,
    "entry_radius_m": 50.0,                # rayon pour le point d'entrée interpolé
    "timeout_seconds": 30,
}


def matching_settings(config: dict[str, Any]) -> dict[str, Any]:
    settings = dict(DEFAULTS)
    settings.update(config.get("map_matching") or {})
    return settings


def _get(url: str, timeout: float) -> dict[str, Any]:
    opener = (urllib.request.build_opener(urllib.request.ProxyHandler({}))
              if urlparse(url).hostname in {"127.0.0.1", "localhost", "::1"} else urllib.request.build_opener())
    try:
        with opener.open(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:          # OSRM renvoie 400 + JSON pour NoMatch, TooBig…
        return json.loads(exc.read().decode("utf-8") or "{}")


def check_osrm(config: dict[str, Any]) -> dict[str, Any]:
    """Vérifie le conteneur OSRM : service joignable (``/nearest``), puis appariement test.

    Les points test sont d'abord recalés sur la route (``/nearest``) : un appariement
    de points posés au hasard peut échouer (« NoMatch ») sans que le service soit en
    cause. Un échec de l'appariement test n'est qu'un avertissement.
    """
    s = matching_settings(config)
    base = s["osrm_url"].rstrip("/")
    points = [(6.1424, 46.2101), (6.1437, 46.2070)]          # Cornavin → rue de Lausanne (Genève)
    try:
        snapped = []
        for lon, lat in points:
            nearest = _get(f"{base}/nearest/v1/{s['profile']}/{lon:.6f},{lat:.6f}?number=1", 15)
            if nearest.get("code") != "Ok":
                raise RuntimeError(f"OSRM /nearest répond {nearest.get('code')} {nearest.get('message')}")
            snapped.append(tuple(nearest["waypoints"][0]["location"]))
    except urllib.error.URLError as exc:
        raise ConnectionError(
            f"Le conteneur OSRM ne répond pas sur {s['osrm_url']} ({exc.reason}). Vérifier le port avec "
            "`docker compose ps` (colonne PORTS, ex. 0.0.0.0:5001->5000/tcp) et l'adapter dans OSRM_MATCH_URL. "
            "Attention : c'est le port du conteneur OSRM, pas celui de l'API xyt (8010)."
        ) from exc
    coords = ";".join(f"{lon:.6f},{lat:.6f}" for lon, lat in snapped)
    match = _get(f"{base}/match/v1/{s['profile']}/{coords}?overview=false&timestamps=0;60&radiuses=25;25", 15)
    result = {"osrm_url": s["osrm_url"], "nearest": "Ok", "test_match": match.get("code"),
              "test_matched_distance_m": (match.get("matchings") or [{}])[0].get("distance")}
    if match.get("code") != "Ok":
        print(f"[matching] ATTENTION : le service répond, mais l'appariement test renvoie {match.get('code')} "
              f"({match.get('message')}). On continue ; regarder le taux d'appariement des vraies legs.")
    return result


def _downsample(n: int, max_points: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    return np.unique(np.round(np.linspace(0, n - 1, max_points)).astype(int))


def match_points(lon: np.ndarray, lat: np.ndarray, t: np.ndarray, radius: np.ndarray, s: dict[str, Any]) -> dict[str, Any]:
    """Appariement d'une suite de points ; renvoie longueur, confiance, géométrie et points recalés."""
    keep = _downsample(len(lon), int(s["max_points"]))
    lon, lat, t, radius = lon[keep], lat[keep], t[keep], radius[keep]
    t = np.maximum.accumulate(np.round(t).astype(np.int64))          # horodatages croissants (exigé par OSRM)
    coords = ";".join(f"{a:.6f},{b:.6f}" for a, b in zip(lon, lat))
    query = (f"?geometries=geojson&overview=full&gaps=split&tidy=true"
             f"&timestamps={';'.join(map(str, t))}&radiuses={';'.join(f'{r:.1f}' for r in radius)}")
    url = f"{s['osrm_url'].rstrip('/')}/match/v1/{s['profile']}/{coords}{query}"
    response = _get(url, float(s["timeout_seconds"]))
    out: dict[str, Any] = {"points_sent": int(len(lon)), "osrm_code": response.get("code")}
    if response.get("code") != "Ok" or not response.get("matchings"):
        out.update({"matching_status": "no_match", "points_matched": 0, "n_matchings": 0,
                    "detour_max": np.nan, "detour_steps_gt3": 0,
                    "matched_distance_m": np.nan, "confidence": np.nan, "geometry": None, "tracepoints": None})
        return out
    matchings = response["matchings"]
    lines = [m["geometry"]["coordinates"] for m in matchings if m.get("geometry")]
    tracepoints = response.get("tracepoints") or []
    snapped = [(tp["location"][0], tp["location"][1]) if tp else (np.nan, np.nan) for tp in tracepoints]
    ratios = detour_ratios(response)
    out.update({
        "detour_max": max(ratios) if ratios else np.nan,
        "detour_steps_gt3": int(sum(r > 3 for r in ratios)),
        "matching_status": "matched",
        "points_matched": int(sum(tp is not None for tp in tracepoints)),
        "n_matchings": len(matchings),
        "matched_distance_m": float(sum(m.get("distance", 0.0) for m in matchings)),
        "confidence": float(np.mean([m.get("confidence", np.nan) for m in matchings])),
        "geometry": lines,
        "tracepoints": pd.DataFrame({"lon": lon, "lat": lat, "snapped_lon": [p[0] for p in snapped],
                                     "snapped_lat": [p[1] for p in snapped]}),
    })
    return out


def window_points(config: dict[str, Any], legs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Point d'entrée interpolé + waypoints de la fenêtre, par leg (lecture partie par partie)."""
    from .spatial_waypoints import _parts

    accuracy_col = config["raw_waypoints"].get("accuracy_col", "accuracy")
    info = legs.set_index("leg_id")[["spatial_entry_at", "spatial_entry_lon", "spatial_entry_lat"]]
    wanted = list(info.index)
    out: dict[str, pd.DataFrame] = {}
    for part in _parts(output_path(config, "data", "waypoints_caronly.parquet")):
        columns = ["leg_id", "timestamp", "lat", "lon"]
        frame = pd.read_parquet(part, filters=[("leg_id", "in", wanted)])
        frame = frame[[c for c in [*columns, accuracy_col] if c in frame.columns]]
        for leg_id, group in frame.groupby("leg_id", sort=False):
            entry_at = pd.to_datetime(info.loc[leg_id, "spatial_entry_at"], utc=True)
            group = group.sort_values("timestamp")
            group = group[pd.to_datetime(group["timestamp"], utc=True) >= entry_at]
            if pd.notna(info.loc[leg_id, "spatial_entry_lon"]):
                entry = pd.DataFrame({"leg_id": [leg_id], "timestamp": [entry_at], "lat": [info.loc[leg_id, "spatial_entry_lat"]],
                                      "lon": [info.loc[leg_id, "spatial_entry_lon"]], accuracy_col: [np.nan], "is_entry": [True]})
                group = pd.concat([entry, group.assign(is_entry=False)], ignore_index=True)
            out[leg_id] = group
    return out


def run_map_matching(config: dict[str, Any], legs: pd.DataFrame, leg_ids: list[str]) -> pd.DataFrame:
    """Apparie les fenêtres des legs demandées et met à jour les tables d'appariement (mise à jour par leg)."""
    import geopandas as gpd
    from shapely.geometry import LineString, MultiLineString

    s = matching_settings(config)
    accuracy_col = config["raw_waypoints"].get("accuracy_col", "accuracy")
    subset = legs[legs["leg_id"].isin(leg_ids)]
    if "spatial_eligible" in subset:
        excluded = int((~subset["spatial_eligible"].astype(bool)).sum())
        if excluded:
            print(f"[matching] {excluded} legs demandées sont exclues par les règles actuelles : elles ne sont pas appariées.")
        subset = subset[subset["spatial_eligible"].astype(bool)]
    points = window_points(config, subset)
    rows, geometries, tracepoints = [], [], []
    started = time.perf_counter()
    for number, (leg_id, group) in enumerate(points.items(), start=1):
        if len(group) < 2:
            rows.append({"leg_id": leg_id, "matching_status": "too_few_points", "points_sent": int(len(group))})
            continue
        t = (pd.to_datetime(group["timestamp"], utc=True) - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()
        accuracy = pd.to_numeric(group.get(accuracy_col, pd.Series(np.nan, index=group.index)), errors="coerce")
        radius = accuracy.clip(s["radius_min_m"], s["radius_max_m"]).fillna(s["radius_max_m"])
        radius = np.where(group["is_entry"].to_numpy(dtype=bool), s["entry_radius_m"], radius)
        try:
            result = match_points(group["lon"].to_numpy(float), group["lat"].to_numpy(float), t, np.asarray(radius, float), s)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            result = {"matching_status": "error", "osrm_code": type(exc).__name__}
        geometry = result.pop("geometry", None)
        tp = result.pop("tracepoints", None)
        rows.append({"leg_id": leg_id, **result})
        if geometry:
            parts = [LineString(c) for c in geometry if len(c) >= 2]
            if parts:
                geometries.append({"leg_id": leg_id, "geometry": parts[0] if len(parts) == 1 else MultiLineString(parts)})
        if tp is not None:
            tracepoints.append(tp.assign(leg_id=leg_id))
        if number % 200 == 0 or number == len(points):
            print(f"[matching] {number}/{len(points)} legs — {time.perf_counter() - started:.0f} s", flush=True)

    table = pd.DataFrame(rows)
    table = table.merge(subset[["leg_id", "spatial_window_distance_m"]], on="leg_id", how="left")
    table["matched_minus_raw_m"] = table["matched_distance_m"] - table["spatial_window_distance_m"]
    table["matched_minus_raw_share"] = table["matched_minus_raw_m"] / table["spatial_window_distance_m"]
    table["matched_at"] = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")

    def upsert(path, new: pd.DataFrame, geo: bool = False) -> None:
        if path.exists():
            old = gpd.read_parquet(path) if geo else pd.read_parquet(path)
            new = pd.concat([old[~old["leg_id"].isin(set(new["leg_id"]))], new], ignore_index=True)
        new.to_parquet(path, index=False)

    upsert(output_path(config, "data", "map_matching_by_leg.parquet"), table)
    if geometries:
        upsert(output_path(config, "data", "map_matching_geometries.parquet"),
               gpd.GeoDataFrame(geometries, geometry="geometry", crs="EPSG:4326"), geo=True)
    if tracepoints:
        upsert(output_path(config, "data", "map_matching_tracepoints.parquet"), pd.concat(tracepoints, ignore_index=True))
    return table


def load_matching(config: dict[str, Any]) -> pd.DataFrame | None:
    path = output_path(config, "data", "map_matching_by_leg.parquet")
    return pd.read_parquet(path).set_index("leg_id") if path.exists() else None


# --------------------------------------------------------------------------- #
# Figures et carte
# --------------------------------------------------------------------------- #


def plot_matching_distributions(table: pd.DataFrame, legs: pd.DataFrame):
    """Écart longueur appariée − longueur brute (m et %), et distance excessive brute vs appariée."""
    import matplotlib.pyplot as plt

    from .indicators import PALETTE, REFERENCE_COLOR, _style

    # Seulement les legs éligibles avec les exclusions actuelles : la table d'appariement
    # s'accumule d'un run à l'autre et peut contenir des legs exclues depuis.
    eligible_ids = set(legs.loc[legs["spatial_eligible"].astype(bool), "leg_id"]) if "spatial_eligible" in legs else set(legs["leg_id"])
    legs = legs[legs["leg_id"].isin(eligible_ids)]
    m = table[table["matching_status"].eq("matched") & table["leg_id"].isin(eligible_ids)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for ax, (column, label, scale) in zip(axes[:2], [("matched_minus_raw_m", "appariée − brute (m)", 1),
                                                     ("matched_minus_raw_share", "appariée − brute (% de la brute)", 100)]):
        values = (m[column] * scale).dropna()
        if len(values):
            low, high = values.quantile([0.01, 0.99])
            ax.hist(values.clip(low, high), bins=40, weights=np.full(len(values), 100 / len(values)),
                    color=PALETTE[1][0], edgecolor=PALETTE[1][1], lw=0.5)
            ax.axvline(values.median(), color=PALETTE[1][1], ls=":", lw=1.5, label=f"médiane {values.median():.1f}")
            ax.axvline(0, color=REFERENCE_COLOR, lw=1)
            ax.legend(frameon=False, fontsize=8)
        ax.set_xlabel(label)
        ax.set_ylabel("% des legs appariées")
        ax.set_title(f"Longueur de la fenêtre : appariée sur le réseau − waypoints bruts\n(n = {len(values)})", fontsize=9)
        _style(ax)
    ax = axes[2]
    both = legs.dropna(subset=["spatial_dist_excess_m", "spatial_dist_excess_matched_m"]) if "spatial_dist_excess_matched_m" in legs else legs.iloc[0:0]
    if len(both):
        values = pd.concat([both["spatial_dist_excess_m"], both["spatial_dist_excess_matched_m"]])
        low, high = values.quantile([0.01, 0.99])
        bins = np.linspace(low, high, 41)
        for column, colors, label in (("spatial_dist_excess_m", PALETTE[2], "brute (waypoints)"),
                                      ("spatial_dist_excess_matched_m", PALETTE[3], "appariée (réseau)")):
            data = both[column].clip(low, high)
            negative = (both[column] < 0).mean()
            ax.hist(data, bins=bins, weights=np.full(len(data), 100 / len(data)), histtype="stepfilled", alpha=0.6,
                    color=colors[0], edgecolor=colors[1], label=f"{label} — {negative:.0%} négatives")
        ax.axvline(0, color=REFERENCE_COLOR, lw=1)
        ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel("distance excessive (m) = distance effective − plus court chemin")
    ax.set_ylabel("% des legs")
    ax.set_title(f"Distance excessive : trace brute vs trace appariée (n = {len(both)})", fontsize=9)
    _style(ax)
    fig.tight_layout()
    return fig


def write_matching_map(config: dict[str, Any], leg_ids: list[str]):
    """Waypoints bruts (points), leur recalage (traits fins), trace appariée (vert) et plus court chemin (bleu)."""
    import folium
    import geopandas as gpd

    tp_path = output_path(config, "data", "map_matching_tracepoints.parquet")
    geo_path = output_path(config, "data", "map_matching_geometries.parquet")
    if not tp_path.exists():
        return None
    tracepoints = pd.read_parquet(tp_path)
    tracepoints = tracepoints[tracepoints["leg_id"].isin(leg_ids)]
    matched = gpd.read_parquet(geo_path).set_index("leg_id") if geo_path.exists() else None
    routed_path = output_path(config, "data", "routing_geometries.parquet")
    routed = gpd.read_parquet(routed_path).drop_duplicates("leg_id").set_index("leg_id") if routed_path.exists() else None
    table = load_matching(config)
    if tracepoints.empty:
        return None
    fmap = folium.Map(location=[tracepoints["lat"].median(), tracepoints["lon"].median()], zoom_start=14, tiles=None)
    from .map_style import add_basemaps

    add_basemaps(fmap)

    def lines_of(geometry):
        return list(geometry.geoms) if hasattr(geometry, "geoms") else [geometry]

    for leg_id, points in tracepoints.groupby("leg_id", sort=False):
        info = table.loc[leg_id] if table is not None and leg_id in table.index else None
        label = leg_id[:8]
        if info is not None and pd.notna(info.get("matched_minus_raw_m")):
            label += f" — appariée − brute {info['matched_minus_raw_m']:+.0f} m"
        group = folium.FeatureGroup(name=label, show=True)
        if routed is not None and leg_id in routed.index:
            for line in lines_of(routed.loc[leg_id, "geometry"]):
                folium.PolyLine([(y, x) for x, y in line.coords], color="#5b8fcb", weight=3, dash_array="6,6",
                                opacity=0.9, tooltip="plus court chemin").add_to(group)
        if matched is not None and leg_id in matched.index:
            for line in lines_of(matched.loc[leg_id, "geometry"]):
                folium.PolyLine([(y, x) for x, y in line.coords], color="#46996a", weight=4, opacity=0.9,
                                tooltip="trace appariée sur le réseau").add_to(group)
        for row in points.itertuples():
            if pd.notna(row.snapped_lat):
                folium.PolyLine([(row.lat, row.lon), (row.snapped_lat, row.snapped_lon)], color="#6b7280",
                                weight=1, opacity=0.8).add_to(group)
                folium.CircleMarker([row.snapped_lat, row.snapped_lon], radius=2, color="#46996a", fill=True,
                                    fill_opacity=1, weight=0).add_to(group)
            folium.CircleMarker([row.lat, row.lon], radius=3, color="#c96a1b", fill=True, fill_color="#f2b36b",
                                fill_opacity=0.9, weight=1,
                                tooltip="waypoint brut" + ("" if pd.notna(row.snapped_lat) else " (non apparié)")).add_to(group)
        group.add_to(fmap)
    legend = ('<div style="position:fixed;bottom:24px;left:12px;z-index:9999;background:#ffffffee;padding:8px 10px;'
              'border-radius:6px;font:12px sans-serif;box-shadow:0 1px 4px #0003;line-height:1.6">'
              '<span style="color:#c96a1b">●</span> waypoint brut (envoyé à OSRM) &nbsp; '
              '<span style="color:#6b7280">—</span> recalage &nbsp; <span style="color:#46996a">●</span> position recalée<br>'
              '<span style="color:#46996a">━</span> trace appariée &nbsp; <span style="color:#5b8fcb">┅</span> plus court chemin</div>')
    fmap.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=True).add_to(fmap)
    path = output_path(config, "maps", "22_map_matching_sample.html")
    path.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(path)
    return path


# --------------------------------------------------------------------------- #
# Diagnostic : d'où viennent les détours de la trace appariée ?
# --------------------------------------------------------------------------- #


def _hav(a: tuple[float, float], b: tuple[float, float]) -> float:
    from .utils import haversine_m

    return float(haversine_m(a[1], a[0], b[1], b[0]))


def detour_ratios(response: dict[str, Any]) -> list[float]:
    """Pour chaque pas apparié (entre deux points recalés consécutifs) : distance sur le réseau / vol d'oiseau.

    Un rapport > 3 sur un pas de quelques dizaines de mètres signale un détour forcé :
    sens unique, interdiction de tourner, rue absente du graphe ou recalage sur la mauvaise chaussée.
    """
    ratios: list[float] = []
    tracepoints = [tp for tp in (response.get("tracepoints") or []) if tp]
    for k, matching in enumerate(response.get("matchings") or []):
        points = sorted((tp for tp in tracepoints if tp.get("matchings_index") == k), key=lambda tp: tp["waypoint_index"])
        for leg, a, b in zip(matching.get("legs", []), points[:-1], points[1:]):
            straight = _hav(tuple(a["location"]), tuple(b["location"]))
            if straight > 5:
                ratios.append(leg.get("distance", np.nan) / straight)
    return ratios


VARIANTS = {
    "actuel": {},
    "rayon 15–50 m": {"radius_min_m": 15.0},
    "rayon 25–80 m": {"radius_min_m": 25.0, "radius_max_m": 80.0},
    "sans horodatages": {"use_timestamps": False},
    "sans tidy": {"tidy": False},
}


def diagnose_matching(config: dict[str, Any], legs: pd.DataFrame, leg_ids: list[str],
                      variants: dict[str, dict[str, Any]] | None = None) -> pd.DataFrame:
    """Rejoue l'appariement de quelques legs avec plusieurs réglages et mesure les détours.

    Une variante peut aussi changer ``osrm_url`` (ex. le conteneur « car » le plus rapide
    plutôt que « car-distance »). Colonnes : longueur appariée, confiance, tronçons, part
    des points appariés, rapport de détour maximal et nombre de pas avec rapport > 3.
    """
    variants = variants or VARIANTS
    base = matching_settings(config)
    accuracy_col = config["raw_waypoints"].get("accuracy_col", "accuracy")
    points = window_points(config, legs[legs["leg_id"].isin(leg_ids)])
    rows = []
    for leg_id, group in points.items():
        if len(group) < 2:
            continue
        t = (pd.to_datetime(group["timestamp"], utc=True) - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()
        accuracy = pd.to_numeric(group.get(accuracy_col, pd.Series(np.nan, index=group.index)), errors="coerce")
        raw = legs.set_index("leg_id").loc[leg_id, "spatial_window_distance_m"]
        for name, change in variants.items():
            s = {**base, **change}
            radius = accuracy.clip(s["radius_min_m"], s["radius_max_m"]).fillna(s["radius_max_m"])
            radius = np.where(group["is_entry"].to_numpy(dtype=bool), s["entry_radius_m"], radius)
            keep = _downsample(len(group), int(s["max_points"]))
            lon, lat = group["lon"].to_numpy(float)[keep], group["lat"].to_numpy(float)[keep]
            ts, rad = np.maximum.accumulate(np.round(t[keep]).astype(np.int64)), np.asarray(radius, float)[keep]
            query = f"?geometries=geojson&overview=false&gaps=split&tidy={'true' if s.get('tidy', True) else 'false'}"
            query += f"&radiuses={';'.join(f'{r:.1f}' for r in rad)}"
            if s.get("use_timestamps", True):
                query += f"&timestamps={';'.join(map(str, ts))}"
            coords = ";".join(f"{a:.6f},{b:.6f}" for a, b in zip(lon, lat))
            response = _get(f"{s['osrm_url'].rstrip('/')}/match/v1/{s['profile']}/{coords}{query}", float(s["timeout_seconds"]))
            ok = response.get("code") == "Ok"
            ratios = detour_ratios(response) if ok else []
            matched = sum(m.get("distance", 0.0) for m in response.get("matchings", [])) if ok else np.nan
            rows.append({
                "leg_id": leg_id, "variante": name, "code": response.get("code"),
                "longueur_brute_m": raw, "longueur_appariee_m": matched,
                "appariee_moins_brute_m": matched - raw if ok else np.nan,
                "confiance": float(np.mean([m.get("confidence", np.nan) for m in response.get("matchings", [])])) if ok else np.nan,
                "troncons": len(response.get("matchings", [])) if ok else 0,
                "part_points_apparies": float(np.mean([tp is not None for tp in response.get("tracepoints", [])])) if ok else 0.0,
                "detour_max": max(ratios) if ratios else np.nan,
                "pas_detour_sup_3": int(sum(r > 3 for r in ratios)),
            })
    return pd.DataFrame(rows)
