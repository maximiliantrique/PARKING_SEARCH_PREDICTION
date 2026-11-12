"""Étape 27 — routage des demandes de la méthode spatiale avec le service local ``xyt_routing``.

Entrée : ``data/routing_requests.parquet`` (écrite par l'étape 22) — une paire
point d'entrée dans le cercle → destination par leg, avec la distance observée
entre waypoints.

Le service ``xyt_routing`` (API HTTP locale, devant des conteneurs OSRM) reçoit
``POST {base_url}/routes`` avec ``{"routes": [{route_id, origin: [lon, lat],
destination: [lon, lat], mode, weighting, geometry}]}`` et répond
``{"results": [{route_id, status, distance_m, duration_s, geometry, …}]}``.

Sorties
-------
- ``data/routing_cache.parquet`` : résultats par ``route_id`` (déterministe :
  demande + mode + pondération). Un nouveau run ne route que les paires absentes ;
  les erreurs ne sont pas mises en cache (elles sont retentées) ;
- ``data/routing_geometries.parquet`` : géométries routées (GeoParquet, EPSG:4326),
  hors de la collection ;
- ``legs/routing_by_leg.parquet`` : une ligne par leg (collection), relue par
  l'étape 22 pour la distance excessive ;
- ``maps/27_routing_comparison.html`` : traces observées vs itinéraires routés.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from .config import ensure_output_dirs, output_path
from .utils import write_json

OK_STATUSES = {"ok", "OK", "Ok"}
N_WAYPOINTS_BEFORE_ENTRY = 4    # carte : derniers waypoints avant l'entrée dans le cercle, en tirets

DEFAULTS = {
    "base_url": "http://127.0.0.1:8010",
    "mode": "car_ch_extended_200km",   # alias du graphe ch_extended_200km dans la config du service
    "weighting": "distance",           # plus court chemin (et non le plus rapide)
    "batch_size": 500,
    "timeout_seconds": 120,
    "include_geometry": True,
    "max_routes": None,                # limiter pour un essai (None = toutes les demandes)
    "force_reroute": False,            # True : ignorer le cache
    "comparison_map_legs": 30,         # legs affichées sur la carte de comparaison
}


def routing_settings(config: dict[str, Any]) -> dict[str, Any]:
    settings = dict(DEFAULTS)
    settings.update(config.get("routing") or {})
    return settings


def route_id(request_id: str, mode: str, weighting: str) -> str:
    return hashlib.sha1(f"{request_id}|{mode}|{weighting}".encode()).hexdigest()[:16]


def _post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    from urllib.parse import urlparse

    request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
    # Service local : ne jamais passer par un proxy (variables HTTP_PROXY d'un réseau d'entreprise)
    if urlparse(url).hostname in {"127.0.0.1", "localhost", "::1"}:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    else:
        opener = urllib.request.build_opener()
    with opener.open(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _normalise(result: dict[str, Any]) -> dict[str, Any]:
    out = dict(result)
    out["routed_distance_m"] = out.get("routed_distance_m", out.get("distance_m"))
    out["routed_duration_s"] = out.get("routed_duration_seconds", out.get("duration_s", out.get("duration_seconds")))
    geometry = out.pop("geometry", None)
    out["routed_geometry_geojson"] = json.dumps(geometry) if geometry is not None else None
    for key in ("origin_snapped", "destination_snapped", "extra"):
        if key in out and out[key] is not None and not isinstance(out[key], str):
            out[key] = json.dumps(out[key])
    keep = ["route_id", "status", "routed_distance_m", "routed_duration_s", "routed_geometry_geojson", "engine",
            "engine_version", "profile", "weighting", "network_id", "network_version", "error_code", "error_message"]
    return {k: out.get(k) for k in keep}


# --------------------------------------------------------------------------- #
# Vérification du service
# --------------------------------------------------------------------------- #


def check_service(config: dict[str, Any]) -> dict[str, Any]:
    """Envoie une route test dans Genève (gare Cornavin → Plainpalais) et explique l'erreur sinon."""
    s = routing_settings(config)
    url = f"{s['base_url'].rstrip('/')}/routes"
    payload = {"routes": [{"route_id": "test", "origin": [6.1424, 46.2101], "destination": [6.1408, 46.1989],
                           "mode": s["mode"], "weighting": s["weighting"], "geometry": False}]}
    try:
        response = _post(url, payload, timeout=30)
    except urllib.error.URLError as exc:
        raise ConnectionError(
            f"Le service de routage ne répond pas sur {url} ({exc.reason}).\n"
            "  1. Docker Desktop est-il lancé ? (icône de la baleine dans la barre de menus)\n"
            "  2. Les conteneurs OSRM tournent-ils ? → docker compose ps (dans le dossier xyt_routing)\n"
            "  3. L'API est-elle lancée ? → XYT_ROUTING_SERVICE_CONFIG=configs/service.ch_extended_200km.yaml .venv/bin/xyt-routing-api"
        ) from exc
    results = response.get("results", [])
    if not results:
        raise RuntimeError(f"Réponse inattendue du service : {response}")
    result = _normalise(results[0])
    if result["status"] not in OK_STATUSES:
        raise RuntimeError(
            f"Le service répond mais la route test échoue : {result.get('error_code')} {result.get('error_message')}. "
            f"Vérifier que le mode {s['mode']!r} existe dans la config du service et que le conteneur OSRM "
            "« car-distance » est démarré."
        )
    return {"url": url, "mode": s["mode"], "weighting": s["weighting"],
            "test_route_distance_m": result["routed_distance_m"], "network_id": result.get("network_id"),
            "network_version": result.get("network_version"), "engine": result.get("engine")}


# --------------------------------------------------------------------------- #
# Routage
# --------------------------------------------------------------------------- #


def load_requests(config: dict[str, Any]) -> pd.DataFrame:
    path = output_path(config, "data", "routing_requests.parquet")
    if not path.exists():
        raise FileNotFoundError(f"{path} introuvable : exécuter la cellule « Demande de routage » du notebook 22.")
    return pd.read_parquet(path)


def run_routing(config: dict[str, Any]) -> dict[str, Any]:
    ensure_output_dirs(config)
    s = routing_settings(config)
    started = time.perf_counter()
    requests = load_requests(config)
    requests["route_id"] = [route_id(r, s["mode"], s["weighting"]) for r in requests["routing_request_id"]]

    cache_path = output_path(config, "data", "routing_cache.parquet")
    cache_columns = ["route_id", "status", "routed_distance_m", "routed_duration_s", "routed_geometry_geojson",
                     "engine", "engine_version", "profile", "weighting", "network_id", "network_version",
                     "error_code", "error_message", "routed_at"]
    cache = pd.read_parquet(cache_path) if cache_path.exists() and not s["force_reroute"] else pd.DataFrame(columns=cache_columns)
    cache = cache.reindex(columns=list(dict.fromkeys([*cache_columns, *cache.columns])))
    todo = requests[~requests["route_id"].isin(set(cache["route_id"]))].drop_duplicates("route_id")
    if s["max_routes"] is not None:
        todo = todo.head(int(s["max_routes"]))

    url = f"{s['base_url'].rstrip('/')}/routes"
    fresh, errors = [], []
    batch_size = int(s["batch_size"])
    for start in range(0, len(todo), batch_size):
        batch = todo.iloc[start: start + batch_size]
        payload = {"routes": [
            {"route_id": row.route_id, "origin": [float(row.origin_lon), float(row.origin_lat)],
             "destination": [float(row.destination_lon), float(row.destination_lat)],
             "mode": s["mode"], "weighting": s["weighting"], "geometry": bool(s["include_geometry"])}
            for row in batch.itertuples()
        ]}
        try:
            response = _post(url, payload, float(s["timeout_seconds"]))
            for result in response.get("results", []):
                normalised = _normalise(result)
                (fresh if normalised["status"] in OK_STATUSES or normalised.get("error_code") == "NoRoute" else errors).append(normalised)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            errors.extend({"route_id": rid, "status": "error", "error_code": type(exc).__name__, "error_message": str(exc)}
                          for rid in batch["route_id"])
        print(f"[routage] {min(start + batch_size, len(todo))}/{len(todo)} routes — {time.perf_counter() - started:.0f} s", flush=True)

    if fresh:
        fresh_frame = pd.DataFrame(fresh)
        fresh_frame["routed_at"] = datetime.now().isoformat(timespec="seconds")
        kept = cache[~cache["route_id"].isin(fresh_frame["route_id"])]
        cache = pd.concat([kept, fresh_frame], ignore_index=True) if len(kept) else fresh_frame
        cache.to_parquet(cache_path, index=False)

    results = requests.merge(cache.drop(columns=["weighting"], errors="ignore"), on="route_id", how="left")
    errors_frame = pd.DataFrame(errors)
    if len(errors_frame):
        errors_frame = errors_frame.drop_duplicates("route_id")
        results = results.set_index("route_id")
        err = errors_frame.set_index("route_id")
        mask = results.index.isin(err.index) & results["status"].isna()
        results.loc[mask, "status"] = "error"
        results.loc[mask, "error_message"] = err.loc[results.index[mask], "error_message"].to_numpy()
        results = results.reset_index()

    write_geometries(config, results)
    by_leg, run_id = build_routing_by_leg(config, results, s)
    from .leg_collection import write_leg_table

    table_path = write_leg_table(by_leg, config, "routing_by_leg", run_id=run_id, params={
        k: s[k] for k in ("base_url", "mode", "weighting", "include_geometry", "max_routes")
    })
    ok = results["status"].isin(OK_STATUSES)
    summary = {
        "requests": int(len(requests)),
        "routed_this_run": int(len(fresh)),
        "from_cache": int(len(requests) - len(todo)) if not s["force_reroute"] else 0,
        "ok": int(ok.sum()),
        "errors": int(results["status"].eq("error").sum()),
        "not_yet_routed": int(results["status"].isna().sum()),
        "status_counts": {str(k): int(v) for k, v in by_leg["routing_status"].value_counts().items()},
        "status_counts_matched_od": {str(k): int(v) for k, v in by_leg["routing_matched_od_status"].value_counts().items()},
        "dist_excess_m": by_leg["routing_dist_excess_m"].describe(percentiles=[0.25, 0.5, 0.75, 0.9]).round(1).to_dict(),
        "dist_excess_matched_od_m": by_leg["routing_matched_od_dist_excess_m"].describe(percentiles=[0.25, 0.5, 0.75, 0.9]).round(1).to_dict(),
        "network": sorted(results["network_id"].dropna().astype(str).unique().tolist()) if "network_id" in results else [],
        "collection_table": str(table_path),
        "seconds": round(time.perf_counter() - started, 1),
    }
    write_json(output_path(config, "reports", "27_routing_summary.json"), summary)
    return {"summary": summary, "results": results, "by_leg": by_leg}


def write_geometries(config: dict[str, Any], results: pd.DataFrame) -> None:
    """Géométries routées en GeoParquet (hors collection)."""
    import geopandas as gpd
    from shapely.geometry import shape

    if "routed_geometry_geojson" not in results:
        return
    rows = results.dropna(subset=["routed_geometry_geojson"])
    if rows.empty:
        return
    geometries = [shape(json.loads(g)) for g in rows["routed_geometry_geojson"]]
    gdf = gpd.GeoDataFrame(rows[["route_id", "leg_id", "route_type", "routed_distance_m"]].copy(), geometry=geometries, crs="EPSG:4326")
    gdf.to_parquet(output_path(config, "data", "routing_geometries.parquet"), index=False)


# type de route → préfixe des colonnes de routing_by_leg (une ligne par leg, un jeu de colonnes par type)
ROUTE_TYPE_PREFIXES = {
    "buffer_entry_to_leg_end": "routing_",                 # OD brute
    "matched_entry_to_matched_end": "routing_matched_od_",  # OD corrigée (positions recalées)
}


def _columns_for(table: pd.DataFrame, prefix: str, s: dict[str, Any]) -> pd.DataFrame:
    status = pd.Series("not_requested", index=table.index, dtype="object")
    requested = table["route_id"].notna()
    status[requested & table["status"].isin(OK_STATUSES)] = "routed"
    status[requested & table["status"].notna() & ~table["status"].isin(OK_STATUSES)] = "failed"
    status[requested & table["status"].isna()] = "not_yet_routed"
    routed = status.eq("routed")
    out = pd.DataFrame(index=table.index)
    out[f"{prefix}status"] = status
    out[f"{prefix}observed_distance_m"] = table["observed_distance_m"].where(requested)
    out[f"{prefix}routed_distance_m"] = pd.to_numeric(table["routed_distance_m"], errors="coerce").where(routed)
    out[f"{prefix}dist_excess_m"] = (out[f"{prefix}observed_distance_m"] - out[f"{prefix}routed_distance_m"]).where(routed)
    out[f"{prefix}routed_duration_s"] = pd.to_numeric(table["routed_duration_s"], errors="coerce").where(routed)
    out[f"{prefix}route_type"] = table["route_type"].where(requested)
    out[f"{prefix}route_id"] = table["route_id"].where(requested)
    out[f"{prefix}graph"] = table.get("network_id", pd.Series(pd.NA, index=table.index)).where(routed).fillna(s["mode"]).where(requested)
    out[f"{prefix}weighting"] = np.where(requested, s["weighting"], None)
    out[f"{prefix}error"] = table.get("error_message", pd.Series(pd.NA, index=table.index)).where(status.eq("failed"))
    for column in (f"{prefix}status", f"{prefix}route_type", f"{prefix}route_id", f"{prefix}graph", f"{prefix}weighting", f"{prefix}error"):
        out[column] = out[column].astype("string")
    return out


def build_routing_by_leg(config: dict[str, Any], results: pd.DataFrame, s: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    """Une ligne par leg ; colonnes ``routing_*`` (OD brute) et ``routing_matched_od_*`` (OD corrigée)."""
    base = pd.read_parquet(output_path(config, "data", "legs_enriched.parquet"), columns=["leg_id"])
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S") + "_routing"
    blocks = [base]
    for route_type, prefix in ROUTE_TYPE_PREFIXES.items():
        r = results[results["route_type"].eq(route_type)].drop_duplicates("leg_id").set_index("leg_id")
        table = base.join(r, on="leg_id")
        if "route_id" not in table:
            table["route_id"] = pd.NA
        for column in ("status", "observed_distance_m", "routed_distance_m", "routed_duration_s", "route_type"):
            if column not in table:
                table[column] = np.nan
        blocks.append(_columns_for(table, prefix, s))
    out = pd.concat(blocks, axis=1)
    out["routing_run_id"] = run_id
    return out, run_id


# --------------------------------------------------------------------------- #
# Carte de comparaison
# --------------------------------------------------------------------------- #


def write_comparison_map(config: dict[str, Any], by_leg: pd.DataFrame, n_legs: int = 30, random_state: int = 42):
    """Trace observée (waypoints, orange) et les deux plus courts chemins : OD brute (bleu) et OD corrigée (violet).

    Ronds : origine (vert) et destination (rose) de la trace ; losanges : origine
    (clair) et destination (foncé) de chaque plus court chemin.
    """
    import folium
    import geopandas as gpd

    geometries_path = output_path(config, "data", "routing_geometries.parquet")
    detail_path = output_path(config, "data", "spatial_waypoints_candidates.parquet")
    if not geometries_path.exists() or not detail_path.exists():
        return None
    excess = by_leg.set_index("leg_id")[[c for c in ("routing_dist_excess_m", "routing_matched_od_dist_excess_m") if c in by_leg]]
    routed = excess.dropna(how="all")
    if routed.empty:
        return None
    score = routed.abs().max(axis=1)
    top = score.nlargest(n_legs // 2).index
    rest = score.index.difference(top)
    sample = list(top) + list(pd.Series(rest).sample(min(n_legs - len(top), len(rest)), random_state=random_state))
    geometries = gpd.read_parquet(geometries_path)
    if "route_type" not in geometries:
        geometries["route_type"] = "buffer_entry_to_leg_end"
    detail = pd.read_parquet(detail_path, columns=["leg_id", "spatial_entry_at", "spatial_entry_lon", "spatial_entry_lat"]).set_index("leg_id")
    entry = detail["spatial_entry_at"]

    from .spatial_waypoints import _parts

    wanted = set(sample)
    points = pd.concat([
        frame[frame["leg_id"].isin(wanted)]
        for frame in (pd.read_parquet(p, columns=["leg_id", "timestamp", "lat", "lon"])
                      for p in _parts(output_path(config, "data", "waypoints_caronly.parquet")))
    ])
    fmap = folium.Map(location=[points["lat"].median(), points["lon"].median()], zoom_start=13, tiles=None)
    from .map_style import add_basemaps

    add_basemaps(fmap)
    styles = {
        "buffer_entry_to_leg_end": ("#5b8fcb", "#B7D3F2", "#2b5c99", "plus court chemin — OD brute"),
        "matched_entry_to_matched_end": ("#8259b8", "#dccbef", "#5b3a8a", "plus court chemin — OD corrigée"),
    }
    for leg_id in sample:
        raw = excess.loc[leg_id].get("routing_dist_excess_m", np.nan)
        corrected = excess.loc[leg_id].get("routing_matched_od_dist_excess_m", np.nan)
        group = folium.FeatureGroup(name=f"{leg_id[:8]} — détour brut {raw:.0f} m / corrigé {corrected:.0f} m", show=True)
        trace = points[points["leg_id"].eq(leg_id)].sort_values("timestamp")
        start = entry.get(leg_id)
        before = trace.iloc[0:0]
        if pd.notna(start):
            inside = pd.to_datetime(trace["timestamp"], utc=True) >= pd.to_datetime(start, utc=True)
            before = trace[~inside].tail(N_WAYPOINTS_BEFORE_ENTRY)
            trace = trace[inside]
        entry_lon, entry_lat = (detail.loc[leg_id, "spatial_entry_lon"], detail.loc[leg_id, "spatial_entry_lat"]) if leg_id in detail.index else (np.nan, np.nan)
        # Approche : les derniers waypoints avant le cercle, en tirets, jusqu'au point d'entrée interpolé
        if len(before):
            approach = before[["lat", "lon"]].values.tolist() + ([[entry_lat, entry_lon]] if pd.notna(entry_lat) else [])
            folium.PolyLine(approach, color="#e3a35f", weight=3, opacity=0.8, dash_array="4,6",
                            tooltip=f"approche : {len(before)} derniers waypoints avant le cercle").add_to(group)
            for row in before.itertuples():
                folium.CircleMarker([row.lat, row.lon], radius=3, color="#c96a1b", weight=1, fill=True,
                                    fill_color="#ffffff", fill_opacity=1, tooltip="waypoint avant le cercle").add_to(group)
        if pd.notna(entry_lat):
            folium.CircleMarker([entry_lat, entry_lon], radius=6, color="#4b5563", weight=2, fill=True, fill_color="#d1d5db",
                                fill_opacity=1, tooltip="point d'entrée interpolé sur le cercle (origine de l'OD brute)").add_to(group)
        if len(trace) >= 2:
            folium.PolyLine(trace[["lat", "lon"]].values.tolist(), color="#e3a35f", weight=4, opacity=0.9,
                            tooltip="trace GPS (waypoints)").add_to(group)
            first, last = trace.iloc[0], trace.iloc[-1]
            folium.CircleMarker([first["lat"], first["lon"]], radius=6, color="#2f7a4f", weight=2, fill=True,
                                fill_color="#B9E3C6", fill_opacity=1, tooltip="trace GPS — origine").add_to(group)
            folium.CircleMarker([last["lat"], last["lon"]], radius=6, color="#b0476a", weight=2, fill=True,
                                fill_color="#F6B6C8", fill_opacity=1, tooltip="trace GPS — destination").add_to(group)
        for row in geometries[geometries["leg_id"].eq(leg_id)].itertuples():
            line_color, start_fill, end_fill, label = styles.get(row.route_type, styles["buffer_entry_to_leg_end"])
            lines = list(row.geometry.geoms) if hasattr(row.geometry, "geoms") else [row.geometry]
            for line in lines:
                folium.PolyLine([(y, x) for x, y in line.coords], color=line_color, weight=3, opacity=0.9, dash_array="6,6",
                                tooltip=f"{label} : {row.routed_distance_m:.0f} m").add_to(group)
            (sx, sy), (ex, ey) = lines[0].coords[0], lines[-1].coords[-1]
            for (lat, lon), fill, text in (((sy, sx), start_fill, "origine"), ((ey, ex), end_fill, "destination")):
                folium.RegularPolygonMarker([lat, lon], number_of_sides=4, rotation=45, radius=7, color=line_color, weight=2,
                                            fill=True, fill_color=fill, fill_opacity=1, tooltip=f"{label} — {text}").add_to(group)
        group.add_to(fmap)
    legend = ('<div style="position:fixed;bottom:24px;left:12px;z-index:9999;background:#ffffffee;padding:8px 10px;'
              'border-radius:6px;font:12px sans-serif;box-shadow:0 1px 4px #0003;line-height:1.6">'
              '<b>Trace GPS</b> (orange) : <span style="color:#2f7a4f">●</span> 1er waypoint dans le cercle <span style="color:#b0476a">●</span> destination '
              '— tirets : derniers waypoints avant le cercle<br>'
              '<span style="color:#6b7280">●</span> point d\'entrée interpolé sur le cercle (origine OD brute ; recalé = origine OD corrigée)<br>'
              '<b>OD brute</b> (bleu, tirets) : <span style="color:#B7D3F2">◆</span> origine <span style="color:#2b5c99">◆</span> destination<br>'
              '<b>OD corrigée</b> (violet, tirets) : <span style="color:#dccbef">◆</span> origine <span style="color:#5b3a8a">◆</span> destination</div>')
    fmap.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=True).add_to(fmap)
    path = output_path(config, "maps", "27_routing_comparison.html")
    path.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(path)
    return path
