from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import resolve_path
from .utils import haversine_m


def normalize_user_ids(values: pd.Series) -> pd.Series:
    """Identifiants utilisateur en texte comparable entre waypoints et legs.

    Corrige les représentations qui ne se voient pas à l'œil mais empêchent
    toute jointure : UUID stockés en binaire (16 octets) ou en objets
    `uuid.UUID`, octets UTF-8, espaces en début/fin. La casse n'est pas
    modifiée.
    """
    import uuid as _uuid

    if values.dtype == object:
        sample = values.dropna()
        sample = sample.iloc[0] if len(sample) else None
        if isinstance(sample, (bytes, bytearray)):
            values = values.map(
                lambda v: str(_uuid.UUID(bytes=bytes(v)))
                if isinstance(v, (bytes, bytearray)) and len(v) == 16
                else (bytes(v).decode("utf-8", "replace") if isinstance(v, (bytes, bytearray)) else v)
            )
        elif isinstance(sample, _uuid.UUID):
            values = values.map(lambda v: str(v) if isinstance(v, _uuid.UUID) else v)
    return values.astype("string").str.strip()


RELUT_COLUMNS = ["trip_candidate_id", "lon", "lat", "timestamp", "speed_kmh"]

LEG_METADATA_COLUMNS = [
    "waypoint_id",
    "leg_id",
    "storyline_id",
    "trip_id",
    "leg_started_at",
    "leg_finished_at",
    "mode",
    "detected_mode",
    "effective_mode",
    "mode_niv1",
    "mode_niv2",
    "mode_mrmt",
    "leg_match_status",
]

LEG_ASSIGNMENT_COLUMNS = [
    "leg_id",
    "storyline_id",
    "trip_id",
    "leg_started_at",
    "leg_finished_at",
    "mode",
    "detected_mode",
    "effective_mode",
    "mode_niv1",
    "mode_niv2",
    "mode_mrmt",
]


DELIMITER_CANDIDATES = (",", ";", "\t", "|")


def resolve_delimiter(path: Path, configured: str | None) -> str:
    """Délimiteur CSV : celui de la config, ou détecté sur l'en-tête si ``auto``.

    La détection compte chaque candidat (virgule, point-virgule, tabulation,
    barre verticale) dans la première ligne et garde le plus fréquent.
    """
    if configured not in (None, "", "auto"):
        return "\t" if configured in ("\\t", "tab") else configured
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        header = handle.readline()
    counts = {candidate: header.count(candidate) for candidate in DELIMITER_CANDIDATES}
    best = max(counts, key=counts.get)
    if counts[best] == 0:
        raise ValueError(
            f"{path.name} : impossible de détecter le délimiteur sur l'en-tête {header[:120]!r}. "
            "Renseigner raw_waypoints.delimiter."
        )
    return best


def discover_waypoint_files(config: dict[str, Any]) -> list[Path]:
    root = Path(config["_project_root"])
    files = sorted(root.glob(config["paths"]["waypoint_glob"]))
    max_files = config["run"].get("max_files")
    if max_files is not None:
        files = files[: int(max_files)]
    return files


def read_raw_waypoints(config: dict[str, Any]) -> pd.DataFrame:
    files = discover_waypoint_files(config)
    raw_cfg = config["raw_waypoints"]
    frames = []
    for path in files:
        frame = pd.read_csv(path, sep=resolve_delimiter(path, raw_cfg.get("delimiter", "auto")))
        frame["source_file"] = path.name
        # CSV line numbers start at 2 because line 1 contains the header. The
        # resulting identifier remains stable as long as the source export is
        # not reordered.
        frame["source_row_number"] = np.arange(2, len(frame) + 2, dtype=np.int64)
        frame["waypoint_id"] = (
            path.name + ":" + frame["source_row_number"].astype(str).str.zfill(9)
        )
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(
            f"No waypoint files matched {config['paths']['waypoint_glob']!r}"
        )

    raw = pd.concat(frames, ignore_index=True)
    max_users = config["run"].get("max_users")
    if max_users is not None:
        users = sorted(raw[raw_cfg["user_col"]].dropna().unique())[: int(max_users)]
        raw = raw[raw[raw_cfg["user_col"]].isin(users)].copy()
    return raw


def read_storyline_tracks(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read storyline Track rows and normalize them as transport legs.

    ``started_at`` and ``finished_at`` are interpreted as UTC. This matches the
    observed Déclic Mobilité export, where the timezone-adjusted values are
    supplied separately in ``*_in_timezone`` columns.
    """

    cfg = config["leg_matching"]
    path = resolve_path(config, config["paths"]["storyline_file"])
    if not path.exists():
        raise FileNotFoundError(f"Storyline file not found: {path}")

    storyline = pd.read_csv(path)
    column_map = {
        "user": cfg["user_col"],
        "type": cfg["type_col"],
        "start": cfg["start_col"],
        "finish": cfg["finish_col"],
        "storyline_id": cfg["storyline_id_col"],
        "trip_id": cfg["trip_id_col"],
        "mode": cfg["mode_col"],
        "detected_mode": cfg["detected_mode_col"],
    }
    missing = sorted(set(column_map.values()) - set(storyline.columns))
    if missing:
        raise ValueError(f"Storyline is missing required columns: {missing}")

    tracks = storyline[storyline[column_map["type"]].eq(cfg.get("track_value", "Track"))].copy()
    tracks["source_user_id"] = normalize_user_ids(tracks[column_map["user"]])
    tracks["storyline_id"] = tracks[column_map["storyline_id"]].astype("string")
    tracks["leg_id"] = tracks["storyline_id"]
    tracks["trip_id"] = tracks[column_map["trip_id"]].astype("string")
    tracks["mode"] = tracks[column_map["mode"]].astype("string")
    tracks["detected_mode"] = tracks[column_map["detected_mode"]].astype("string")
    tracks["effective_mode"] = tracks["mode"].fillna(tracks["detected_mode"])
    tracks["mode_niv1"] = pd.NA
    tracks["mode_niv2"] = pd.NA
    tracks["mode_mrmt"] = pd.NA
    tracks["leg_started_at"] = pd.to_datetime(
        tracks[column_map["start"]], errors="coerce", utc=True
    )
    tracks["leg_finished_at"] = pd.to_datetime(
        tracks[column_map["finish"]], errors="coerce", utc=True
    )

    valid = (
        tracks["source_user_id"].notna()
        & tracks["storyline_id"].notna()
        & tracks["leg_started_at"].notna()
        & tracks["leg_finished_at"].notna()
        & tracks["leg_finished_at"].gt(tracks["leg_started_at"])
    )
    invalid_rows = int((~valid).sum())
    tracks = tracks[valid].copy()

    duplicate_ids = tracks["storyline_id"].duplicated(keep=False)
    if duplicate_ids.any():
        examples = tracks.loc[duplicate_ids, "storyline_id"].drop_duplicates().head(5).tolist()
        raise ValueError(f"Storyline contains duplicated storyline_id values: {examples}")

    tracks = tracks.sort_values(["source_user_id", "leg_started_at", "leg_finished_at"])
    tracks["previous_max_finished_at"] = tracks.groupby("source_user_id")[
        "leg_finished_at"
    ].transform(lambda values: values.shift().cummax())
    tracks["overlaps_previous"] = tracks["leg_started_at"].lt(
        tracks["previous_max_finished_at"]
    )

    report = {
        "storyline_path": str(path),
        "storyline_rows": int(len(storyline)),
        "track_rows": int(len(tracks)),
        "invalid_track_rows": invalid_rows,
        "track_users": int(tracks["source_user_id"].nunique()),
        "tracks_with_previous_overlap": int(tracks["overlaps_previous"].sum()),
    }
    return tracks.reset_index(drop=True), report


LEGS_PARQUET_SOURCES = {
    # source -> la colonne de mode détecté est-elle obligatoire ?
    # (le format de fichier, CSV ou Parquet, est déduit de l'extension)
    "declic_legs_parquet": True,
    "declic_legs_export": True,
    "pl23_legs_parquet": False,
}

# Export de legs Déclic (ex. declic-mobility-vague1, avec géométrie WKT) :
# colonnes appliquées automatiquement pour la source "declic_legs_export",
# surchargeables via leg_matching.declic_export_columns. La géométrie et les
# ~80 autres colonnes ne sont pas lues. `started_at` / `finished_at` portent
# déjà leur fuseau (+00:00).
DECLIC_EXPORT_LEG_COLUMNS = {
    "user_col": "user_id",
    "type_col": "type",
    "track_value": "Track",
    "start_col": "started_at",
    "finish_col": "finished_at",
    "leg_id_col": "leg_id",
    "mode_col": "mode",
    "detected_mode_col": "detected_mode",
    "storyline_id_col": "storyline_id",
    "trip_id_col": "trip_id",
    "mode_niv1_col": "mode_niv1",
    "mode_niv2_col": "mode_niv2",
    "mode_mrmt_col": "mode_mrmt",
    "length_col": "length_leg",
    "duration_col": "duration",
}

# Colonnes du dump Situee / PL23 (241120_legs_nogeom). Appliquées automatiquement
# pour la source "pl23_legs_parquet" afin de ne pas toucher aux colonnes Déclic de
# la config ; surchargeables via leg_matching.pl23_columns. `started_at` /
# `finished_at` sont en UTC sans fuseau (vérifié : égaux à *_in_timezone).
PL23_LEG_COLUMNS = {
    "user_col": "user_id_fors",
    "type_col": "type",
    "track_value": "Track",
    "start_col": "started_at",
    "finish_col": "finished_at",
    "leg_id_col": "leg_id",
    "mode_col": "mode",
    "detected_mode_col": "detected_mode",
    "storyline_id_col": None,
    "trip_id_col": None,
    "mode_niv1_col": None,
    "mode_niv2_col": None,
    "mode_mrmt_col": None,
    "length_col": "length_leg",
    "duration_col": "duration",
}


def _legs_column_settings(config: dict[str, Any], source: str) -> dict[str, Any]:
    cfg = dict(config["leg_matching"])
    if source == "pl23_legs_parquet":
        cfg.update(PL23_LEG_COLUMNS)
        cfg.update(config["leg_matching"].get("pl23_columns") or {})
    elif source == "declic_legs_export":
        cfg.update(DECLIC_EXPORT_LEG_COLUMNS)
        cfg.update(config["leg_matching"].get("declic_export_columns") or {})
    return cfg


def _read_legs_table(path: Path, wanted_columns: list[str]) -> pd.DataFrame:
    """Lit la table de legs en Parquet ou en CSV (selon l'extension).

    En CSV, seules les colonnes utiles sont lues et les identifiants restent du
    texte, pour limiter la mémoire et éviter les conversions de type.
    """
    if path.suffix.lower() != ".csv":
        return pd.read_parquet(path)
    header = list(pd.read_csv(path, nrows=0).columns)
    usecols = [column for column in header if column in set(wanted_columns)]
    if not usecols:
        usecols = header
    return pd.read_csv(path, usecols=usecols, dtype="string")


def read_declic_legs(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read Déclic Mobilité cleaned legs from parquet and normalize them.

    The cleaned base already stores UTC-aware ``started_at`` and ``finished_at``
    values. Each waypoint is later matched to the half-open interval containing
    its timestamp, by user.
    """
    return read_legs_parquet(config, source="declic_legs_parquet")


USER_ID_MAP_SOURCES = {"pl23_legs_parquet"}


def apply_user_id_map(
    tracks: pd.DataFrame, config: dict[str, Any], *, source: str = "pl23_legs_parquet"
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Traduit l'identifiant utilisateur des legs dans celui des waypoints.

    Utile quand les legs et les waypoints n'utilisent pas le même identifiant,
    par exemple pour le Panel Lémanique 2023 : les legs portent l'identifiant du
    panel (``user_id_fors``, ex. ``CH3181``) et les waypoints l'identifiant
    Motiontag (UUID). Configuration ::

        leg_matching:
          user_id_map:
            file: Data/GPS/pl23/20241120_dump_situee/241120_user_statistics_nogeom.csv
            legs_id_col: user_id_fors           # identifiant présent dans les legs
            waypoints_id_col: user_id_motiontag # identifiant présent dans les waypoints

    ``source_user_id`` prend la valeur Motiontag ; l'identifiant d'origine est
    conservé dans ``legs_user_id_original``. Les legs dont l'utilisateur n'est
    pas dans la table de correspondance sont retirées et comptées. Une
    correspondance ambiguë (un identifiant associé à plusieurs autres) arrête le
    run, car elle rattacherait des waypoints à la mauvaise personne.
    """
    map_cfg = config["leg_matching"].get("user_id_map") or {}
    if not map_cfg or not map_cfg.get("file"):
        return tracks, {"user_id_map_applied": False}
    if source not in USER_ID_MAP_SOURCES:
        # Déclic : legs et waypoints partagent le même identifiant (rétrocompatibilité).
        print(
            f"[legs] leg_matching.user_id_map ignoré pour la source {source!r} : "
            "même identifiant dans les legs et les waypoints.",
            flush=True,
        )
        return tracks, {"user_id_map_applied": False, "user_id_map_ignored_for_source": source}

    path = resolve_path(config, map_cfg["file"])
    if not path.exists():
        raise FileNotFoundError(f"leg_matching.user_id_map.file introuvable : {path}")
    legs_col = map_cfg.get("legs_id_col", "user_id_fors")
    waypoints_col = map_cfg.get("waypoints_id_col", "user_id_motiontag")

    if path.suffix.lower() == ".csv":
        header = list(pd.read_csv(path, nrows=0).columns)
        missing = [column for column in (legs_col, waypoints_col) if column not in header]
        if missing:
            raise KeyError(
                f"Colonnes {missing} absentes de {path.name}. Colonnes disponibles : {header}"
            )
        mapping = pd.read_csv(path, usecols=[legs_col, waypoints_col], dtype="string")
    else:
        mapping = pd.read_parquet(path, columns=[legs_col, waypoints_col])

    mapping = pd.DataFrame(
        {
            "legs_id": normalize_user_ids(mapping[legs_col]),
            "waypoints_id": normalize_user_ids(mapping[waypoints_col]),
        }
    ).dropna().drop_duplicates()

    for column, label in (("legs_id", legs_col), ("waypoints_id", waypoints_col)):
        ambiguous = mapping[mapping[column].duplicated(keep=False)]
        if len(ambiguous):
            examples = ambiguous.sort_values(column).head(6).to_dict("records")
            raise ValueError(
                f"Correspondance ambiguë dans {path.name} : des valeurs de {label!r} sont "
                f"associées à plusieurs identifiants. Exemples : {examples}"
            )

    lookup = dict(zip(mapping["legs_id"], mapping["waypoints_id"]))
    users_before = tracks["source_user_id"].dropna().unique()
    tracks = tracks.copy()
    tracks["legs_user_id_original"] = tracks["source_user_id"]
    tracks["source_user_id"] = tracks["source_user_id"].map(lookup).astype("string")
    unmapped = tracks["source_user_id"].isna() & tracks["legs_user_id_original"].notna()
    unmapped_users = tracks.loc[unmapped, "legs_user_id_original"].dropna().unique()
    tracks = tracks[~unmapped].copy()

    report = {
        "user_id_map_applied": True,
        "user_id_map_file": str(path),
        "user_id_map_columns": f"{legs_col} -> {waypoints_col}",
        "user_id_map_rows": int(len(mapping)),
        "user_id_map_legs_users": int(len(users_before)),
        "user_id_map_legs_users_mapped": int(len(users_before) - len(unmapped_users)),
        "user_id_map_legs_users_unmapped": int(len(unmapped_users)),
        "user_id_map_legs_rows_dropped": int(unmapped.sum()),
        "user_id_map_unmapped_examples": [str(value) for value in unmapped_users[:5]],
    }
    return tracks, report


def read_legs_parquet(
    config: dict[str, Any], *, source: str = "declic_legs_parquet"
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Lit une table de legs (Parquet ou CSV ; Déclic ou dump Situee / PL23) et la normalise.

    Les noms de colonnes viennent de ``leg_matching.*_col``. Pour
    ``pl23_legs_parquet``, la colonne de mode détecté est facultative : si elle
    est absente, ``detected_mode`` vaut NA et ``effective_mode`` = ``mode``.
    Les dates sans fuseau horaire sont interprétées comme UTC.
    """

    cfg = _legs_column_settings(config, source)
    path = resolve_path(config, config["paths"]["legs_file"])
    if not path.exists():
        raise FileNotFoundError(f"Legs file not found: {path}")
    require_detected_mode = LEGS_PARQUET_SOURCES[source]
    legs_source = source  # `source` est réutilisé plus bas comme nom de colonne

    wanted = [
        value
        for key, value in cfg.items()
        if key.endswith("_col") and isinstance(value, str)
    ]
    legs = _read_legs_table(path, wanted)
    column_map = {
        "user": cfg["user_col"],
        "type": cfg.get("type_col"),
        "start": cfg["start_col"],
        "finish": cfg["finish_col"],
        "leg_id": cfg["leg_id_col"],
        "storyline_id": cfg.get("storyline_id_col"),
        "trip_id": cfg.get("trip_id_col"),
        "mode": cfg["mode_col"],
        "detected_mode": cfg.get("detected_mode_col"),
        "mode_niv1": cfg.get("mode_niv1_col"),
        "mode_niv2": cfg.get("mode_niv2_col"),
        "mode_mrmt": cfg.get("mode_mrmt_col"),
    }
    required = {
        column_map["user"],
        column_map["start"],
        column_map["finish"],
        column_map["leg_id"],
        column_map["mode"],
    }
    if require_detected_mode:
        required.add(column_map["detected_mode"])
    missing = sorted(str(column) for column in required if column not in legs.columns)
    if missing:
        # En CSV seules les colonnes utiles ont été lues : on relit l'en-tête complet.
        all_columns = (
            list(pd.read_csv(path, nrows=0).columns)
            if path.suffix.lower() == ".csv"
            else list(legs.columns)
        )
        preset = {
            "pl23_legs_parquet": "PL23_LEG_COLUMNS (+ leg_matching.pl23_columns)",
            "declic_legs_export": "DECLIC_EXPORT_LEG_COLUMNS (+ leg_matching.declic_export_columns)",
        }.get(legs_source, "leg_matching.*_col de la config")
        used = {
            key: cfg.get(key)
            for key in ("user_col", "start_col", "finish_col", "leg_id_col", "mode_col", "detected_mode_col")
        }
        raise ValueError(
            f"Legs table ({legs_source}, {path.name}) is missing required columns: {missing}.\n"
            f"  Colonnes demandées (source : {preset}) : {used}\n"
            f"  Colonnes du fichier : {all_columns}\n"
            "  Les colonnes des legs se règlent indépendamment de celles des waypoints "
            "(raw_waypoints.user_col) ; si les identifiants diffèrent, utiliser "
            "leg_matching.user_id_map."
        )

    type_col = column_map["type"]
    if type_col and type_col in legs.columns:
        tracks = legs[legs[type_col].eq(cfg.get("track_value", "Track"))].copy()
    else:
        tracks = legs.copy()

    tracks["source_user_id"] = normalize_user_ids(tracks[column_map["user"]])
    tracks, user_map_report = apply_user_id_map(tracks, config, source=legs_source)
    tracks["leg_id"] = tracks[column_map["leg_id"]].astype("string")
    for target in ["storyline_id", "trip_id", "mode_niv1", "mode_niv2", "mode_mrmt"]:
        source = column_map.get(target)
        if source and source in tracks.columns:
            tracks[target] = tracks[source].astype("string")
        else:
            tracks[target] = pd.NA
    tracks["mode"] = tracks[column_map["mode"]].astype("string")
    detected_col = column_map["detected_mode"]
    if detected_col and detected_col in tracks.columns:
        tracks["detected_mode"] = tracks[detected_col].astype("string")
    else:
        tracks["detected_mode"] = pd.Series(pd.NA, index=tracks.index, dtype="string")
    tracks["effective_mode"] = tracks["mode"].fillna(tracks["detected_mode"])
    # Longueur et durée de la leg (facultatives ; utilisées par leg_enrichment.py)
    for target, key in (("leg_length_m", "length_col"), ("leg_duration_seconds", "duration_col")):
        source = cfg.get(key)
        tracks[target] = (
            pd.to_numeric(tracks[source], errors="coerce") if source and source in tracks.columns else np.nan
        )
    tracks["leg_started_at"] = pd.to_datetime(
        tracks[column_map["start"]], errors="coerce", utc=True
    )
    tracks["leg_finished_at"] = pd.to_datetime(
        tracks[column_map["finish"]], errors="coerce", utc=True
    )

    valid = (
        tracks["source_user_id"].notna()
        & tracks["leg_id"].notna()
        & tracks["leg_started_at"].notna()
        & tracks["leg_finished_at"].notna()
        & tracks["leg_finished_at"].gt(tracks["leg_started_at"])
    )
    invalid_rows = int((~valid).sum())
    tracks = tracks[valid].copy()

    duplicate_ids = tracks["leg_id"].duplicated(keep=False)
    if duplicate_ids.any():
        examples = tracks.loc[duplicate_ids, "leg_id"].drop_duplicates().head(5).tolist()
        raise ValueError(f"Legs parquet contains duplicated leg_id values: {examples}")

    tracks = tracks.sort_values(["source_user_id", "leg_started_at", "leg_finished_at"])
    tracks["previous_max_finished_at"] = tracks.groupby("source_user_id")[
        "leg_finished_at"
    ].transform(lambda values: values.shift().cummax())
    tracks["overlaps_previous"] = tracks["leg_started_at"].lt(
        tracks["previous_max_finished_at"]
    )

    report = {
        "legs_source": legs_source,
        "legs_path": str(path),
        "legs_rows": int(len(legs)),
        "track_rows": int(len(tracks)),
        "invalid_track_rows": invalid_rows,
        **user_map_report,
        "track_users": int(tracks["source_user_id"].nunique()),
        "tracks_with_previous_overlap": int(tracks["overlaps_previous"].sum()),
    }
    return tracks.reset_index(drop=True), report


def read_transport_legs(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = config["leg_matching"].get("source", "storyline_csv")
    if source in LEGS_PARQUET_SOURCES:
        return read_legs_parquet(config, source=source)
    if source == "storyline_csv":
        return read_storyline_tracks(config)
    raise ValueError(
        "leg_matching.source must be one of "
        f"{sorted([*LEGS_PARQUET_SOURCES, 'storyline_csv'])}, got {source!r}"
    )


def attach_waypoints_to_legs(
    waypoints: pd.DataFrame,
    tracks: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach each waypoint to the Track interval containing its timestamp.

    Intervals are half-open: ``leg_started_at <= timestamp < leg_finished_at``.
    If Track intervals overlap, the most recently started Track is retained but
    the point is marked ``overlapping_tracks`` and excluded from car filtering.
    """

    required_waypoints = {"source_user_id", "timestamp", "waypoint_id"}
    missing_waypoints = sorted(required_waypoints - set(waypoints.columns))
    if missing_waypoints:
        raise ValueError(f"Waypoints are missing matching columns: {missing_waypoints}")

    out = waypoints.copy()
    n_rows = len(out)
    tracks = tracks.copy()
    if "leg_id" not in tracks.columns and "storyline_id" in tracks.columns:
        tracks["leg_id"] = tracks["storyline_id"]
    for column in LEG_ASSIGNMENT_COLUMNS:
        if column not in tracks.columns:
            tracks[column] = pd.NA
    # L'appariement suppose les legs de chaque utilisateur triées par début (searchsorted) :
    # on trie toujours, quel que soit l'ordre reçu, et on recalcule le chevauchement
    # dans cet ordre (une table enrichie peut arriver triée par trip).
    tracks = tracks.sort_values(["source_user_id", "leg_started_at", "leg_finished_at"], kind="mergesort")
    tracks["previous_max_finished_at"] = tracks.groupby("source_user_id")[
        "leg_finished_at"
    ].transform(lambda values: values.shift().cummax())

    assignment_columns = LEG_ASSIGNMENT_COLUMNS
    assignments = {column: np.full(n_rows, pd.NA, dtype=object) for column in assignment_columns}
    statuses = np.full(n_rows, "outside_track", dtype=object)

    tracks_by_user = {
        str(user_id): group.reset_index(drop=True)
        for user_id, group in tracks.groupby("source_user_id", sort=False)
    }

    for user_id, point_positions in out.groupby("source_user_id", sort=False).indices.items():
        user_tracks = tracks_by_user.get(str(user_id))
        if user_tracks is None or user_tracks.empty:
            statuses[point_positions] = "user_without_tracks"
            continue

        point_times = out.iloc[point_positions]["timestamp"].array.asi8
        starts = user_tracks["leg_started_at"].array.asi8
        finishes = user_tracks["leg_finished_at"].array.asi8
        previous_max_finishes = user_tracks["previous_max_finished_at"].array.asi8

        candidates = np.searchsorted(starts, point_times, side="right") - 1
        has_candidate = candidates >= 0
        contained = np.zeros(len(point_positions), dtype=bool)
        contained[has_candidate] = (
            point_times[has_candidate] < finishes[candidates[has_candidate]]
        )

        # A later overlapping Track can finish before an earlier, longer Track.
        # In that uncommon case, fall back to the most recently started interval
        # that still contains the point.
        running_max_finishes = np.maximum.accumulate(finishes)
        can_fallback = np.zeros(len(point_positions), dtype=bool)
        can_fallback[has_candidate] = (
            point_times[has_candidate]
            < running_max_finishes[candidates[has_candidate]]
        )
        fallback_positions = np.flatnonzero(has_candidate & ~contained & can_fallback)
        for point_position in fallback_positions:
            track_position = candidates[point_position] - 1
            while track_position >= 0:
                if point_times[point_position] < finishes[track_position]:
                    candidates[point_position] = track_position
                    contained[point_position] = True
                    break
                track_position -= 1

        matched_positions = np.flatnonzero(contained)
        if not len(matched_positions):
            continue

        matched_tracks = candidates[matched_positions]
        ambiguous = (
            point_times[matched_positions]
            < previous_max_finishes[matched_tracks]
        )
        statuses[point_positions[matched_positions]] = np.where(
            ambiguous, "overlapping_tracks", "matched"
        )

        for column in assignment_columns:
            values = user_tracks[column].to_numpy(dtype=object)
            assignments[column][point_positions[matched_positions]] = values[matched_tracks]

    for column, values in assignments.items():
        out[column] = values
    out["leg_match_status"] = statuses

    status_counts = out["leg_match_status"].value_counts(dropna=False).to_dict()
    unambiguous_matches = int(status_counts.get("matched", 0))
    report = {
        "waypoint_rows": n_rows,
        "waypoint_users": int(out["source_user_id"].nunique()),
        "matched_waypoints": unambiguous_matches,
        "matched_share": float(unambiguous_matches / n_rows) if n_rows else 0.0,
        "status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "matched_legs": int(
            out.loc[out["leg_match_status"].eq("matched"), _leg_id_column(out)].nunique()
        ),
    }
    return out, report


def filter_car_waypoints(
    waypoints_with_legs: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Keep unambiguously matched waypoints belonging to configured car modes."""

    car_modes = set(config["leg_matching"].get("car_modes", ["Mode::Car"]))
    matched = waypoints_with_legs["leg_match_status"].eq("matched")
    is_car = waypoints_with_legs["effective_mode"].isin(car_modes)
    car = waypoints_with_legs[matched & is_car].copy()

    matched_modes = (
        waypoints_with_legs.loc[matched, "effective_mode"]
        .fillna("[missing]")
        .value_counts(dropna=False)
        .to_dict()
    )
    report = {
        "car_modes": sorted(car_modes),
        "matched_waypoints_by_mode": {
            str(key): int(value) for key, value in matched_modes.items()
        },
        "car_waypoints": int(len(car)),
        "car_legs": int(car[_leg_id_column(car)].nunique()) if len(car) else 0,
        "car_users": int(car["source_user_id"].nunique()),
    }
    return car.reset_index(drop=True), report


def _leg_id_column(df: pd.DataFrame) -> str:
    for column in ["leg_id", "storyline_id", "trip_id"]:
        if column in df.columns and df[column].notna().any():
            return column
    return "source_user_id"


def clean_waypoints(raw: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw_cfg = config["raw_waypoints"]
    clean_cfg = config["cleaning"]

    col_user = raw_cfg["user_col"]
    col_time = raw_cfg["time_col"]
    col_lat = raw_cfg["lat_col"]
    col_lon = raw_cfg["lon_col"]
    col_speed = raw_cfg["speed_col"]
    col_accuracy = raw_cfg["accuracy_col"]

    df = raw.copy()
    report: dict[str, Any] = {
        "raw_rows": int(len(df)),
        "raw_users": int(df[col_user].nunique(dropna=True)),
    }

    for col in [col_lat, col_lon, col_speed, col_accuracy]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["timestamp"] = pd.to_datetime(df[col_time], errors="coerce", utc=True)
    df["source_user_id"] = normalize_user_ids(df[col_user])
    df["lat"] = df[col_lat]
    df["lon"] = df[col_lon]

    valid = (
        df["source_user_id"].notna()
        & df["timestamp"].notna()
        & df["lat"].between(-90, 90)
        & df["lon"].between(-180, 180)
    )
    report["dropped_required_or_bounds"] = int((~valid).sum())
    df = df[valid].copy()

    # Filtre de précision du corpus commun (leg_filters.max_accuracy_m), sinon cleaning.max_accuracy_m
    max_accuracy = (config.get("leg_filters") or {}).get("max_accuracy_m", clean_cfg.get("max_accuracy_m"))
    if max_accuracy is not None and col_accuracy in df.columns:
        before = len(df)
        df = df[(df[col_accuracy].isna()) | (df[col_accuracy] <= float(max_accuracy))].copy()
        report["dropped_accuracy"] = int(before - len(df))

    if clean_cfg.get("drop_duplicate_points", True):
        before = len(df)
        df = df.drop_duplicates(
            subset=["source_user_id", "timestamp", "lat", "lon"], keep="first"
        ).copy()
        report["dropped_duplicates"] = int(before - len(df))

    df = df.sort_values(["source_user_id", "timestamp"]).reset_index(drop=True)

    df["dt_seconds"] = df.groupby("source_user_id")["timestamp"].diff().dt.total_seconds()
    df["prev_lat"] = df.groupby("source_user_id")["lat"].shift()
    df["prev_lon"] = df.groupby("source_user_id")["lon"].shift()
    df["distance_from_previous_m"] = haversine_m(
        df["prev_lat"], df["prev_lon"], df["lat"], df["lon"]
    )
    computed_speed = (df["distance_from_previous_m"] / df["dt_seconds"]) * 3.6
    computed_speed = computed_speed.replace([np.inf, -np.inf], np.nan)
    computed_speed[df["dt_seconds"] <= 0] = np.nan
    df["computed_speed_kmh"] = computed_speed
    df["distance_from_previous_m"] = df["distance_from_previous_m"].fillna(0)

    raw_speed = df[col_speed].copy()
    for value in clean_cfg.get("unknown_speed_values", []):
        raw_speed = raw_speed.mask(raw_speed == value)

    if raw_cfg.get("speed_unit") == "mps":
        raw_speed_kmh = raw_speed * 3.6
    elif raw_cfg.get("speed_unit") == "kmh":
        raw_speed_kmh = raw_speed
    else:
        raise ValueError("raw_waypoints.speed_unit must be 'mps' or 'kmh'")

    df["raw_speed_kmh"] = raw_speed_kmh
    df["speed_kmh"] = df["raw_speed_kmh"].fillna(df["computed_speed_kmh"]).fillna(0)
    df["speed_kmh"] = df["speed_kmh"].clip(lower=0)

    max_speed = clean_cfg.get("max_speed_kmh")
    if max_speed is not None:
        df["speed_above_cap"] = df["speed_kmh"] > float(max_speed)
        df["speed_kmh"] = df["speed_kmh"].clip(upper=float(max_speed))
    else:
        df["speed_above_cap"] = False

    report["clean_rows"] = int(len(df))
    report["clean_users"] = int(df["source_user_id"].nunique(dropna=True))
    report["speed_missing_or_unknown_rows"] = int(raw_speed_kmh.isna().sum())
    report["speed_capped_rows"] = int(df["speed_above_cap"].sum())

    return df, report


SEGMENT_STAT_COLUMNS = [
    "trip_candidate_id",
    "source_user_id",
    "leg_id",
    "storyline_id",
    "trip_id",
    "effective_mode",
    "mode_niv1",
    "mode_niv2",
    "mode_mrmt",
    "candidate_sequence",
    "point_count",
    "started_at",
    "finished_at",
    "duration_seconds",
    "distance_m",
    "median_interval_seconds",
    "kept",
]


def segment_waypoints(
    clean: pd.DataFrame, config: dict[str, Any], *, allow_empty: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Découpe les waypoints voiture en segments candidats.

    ``allow_empty=True`` est utilisé par le mode partitionné : une tranche
    d'utilisateurs peut ne contenir aucun segment retenu sans que ce soit une
    erreur pour l'ensemble du jeu. On renvoie alors des tables vides (avec les
    bonnes colonnes) et les candidats rejetés restent dans les statistiques.
    """
    seg_cfg = config["segmentation"]
    grouping_columns = ["source_user_id"]
    leg_group_column = None
    for candidate in ["leg_id", "storyline_id"]:
        if candidate in clean.columns and clean[candidate].notna().any():
            leg_group_column = candidate
            break
    if leg_group_column is not None:
        grouping_columns.append(leg_group_column)

    df = clean.sort_values(grouping_columns + ["timestamp"]).copy()
    gap_seconds = float(seg_cfg["gap_minutes"]) * 60

    df["gap_split"] = df.groupby(grouping_columns)["timestamp"].diff().dt.total_seconds().gt(
        gap_seconds
    )
    df["candidate_sequence"] = df.groupby(grouping_columns)["gap_split"].cumsum().astype(int)

    # Recompute kinematics within candidate segments. This prevents the first
    # point of a leg, or the first point after a large gap, from inheriting a
    # distance and imputed speed from another transport leg.
    candidate_grouping = grouping_columns + ["candidate_sequence"]
    df["dt_seconds"] = df.groupby(candidate_grouping)["timestamp"].diff().dt.total_seconds()
    df["prev_lat"] = df.groupby(candidate_grouping)["lat"].shift()
    df["prev_lon"] = df.groupby(candidate_grouping)["lon"].shift()
    df["distance_from_previous_m"] = haversine_m(
        df["prev_lat"], df["prev_lon"], df["lat"], df["lon"]
    )
    computed_speed = (df["distance_from_previous_m"] / df["dt_seconds"]) * 3.6
    computed_speed = computed_speed.replace([np.inf, -np.inf], np.nan)
    computed_speed[df["dt_seconds"] <= 0] = np.nan
    df["computed_speed_kmh"] = computed_speed
    if "raw_speed_kmh" in df.columns:
        df["speed_kmh"] = df["raw_speed_kmh"].fillna(df["computed_speed_kmh"]).fillna(0)
    else:
        df["speed_kmh"] = df["speed_kmh"].fillna(df["computed_speed_kmh"]).fillna(0)
    df["distance_from_previous_m"] = df["distance_from_previous_m"].fillna(0)
    max_speed = config["cleaning"].get("max_speed_kmh")
    df["speed_kmh"] = df["speed_kmh"].clip(lower=0, upper=max_speed)

    segmented_frames = []
    stats = []
    for group_key, group in df.groupby(candidate_grouping, sort=True):
        if leg_group_column is not None:
            user_id, leg_group_id, seq = group_key
        else:
            user_id, seq = group_key
            leg_group_id = None
        group = group.sort_values("timestamp").copy()
        group = _trim_static_tail(group, seg_cfg)
        if group.empty:
            continue

        id_root = leg_group_id if leg_group_id is not None else user_id
        candidate_id = f"{id_root}__seg_{seq:04d}"
        group["trip_candidate_id"] = candidate_id

        duration_s = (group["timestamp"].iloc[-1] - group["timestamp"].iloc[0]).total_seconds()
        distance_m = float(group["distance_from_previous_m"].fillna(0).sum())
        point_count = int(len(group))
        median_interval = float(group["dt_seconds"].dropna().median()) if point_count > 1 else np.nan

        keep = (
            point_count >= int(seg_cfg["min_points"])
            and point_count <= int(seg_cfg["max_points_per_segment"])
            and duration_s >= float(seg_cfg["min_duration_minutes"]) * 60
            and duration_s <= float(seg_cfg["max_duration_minutes"]) * 60
            and distance_m >= float(seg_cfg["min_distance_m"])
        )

        stats.append(
            {
                "trip_candidate_id": candidate_id,
                "source_user_id": user_id,
                "leg_id": group["leg_id"].iloc[0] if "leg_id" in group.columns else None,
                "storyline_id": (
                    group["storyline_id"].iloc[0] if "storyline_id" in group.columns else None
                ),
                "trip_id": group["trip_id"].iloc[0] if "trip_id" in group.columns else None,
                "effective_mode": (
                    group["effective_mode"].iloc[0] if "effective_mode" in group.columns else None
                ),
                "mode_niv1": group["mode_niv1"].iloc[0] if "mode_niv1" in group.columns else None,
                "mode_niv2": group["mode_niv2"].iloc[0] if "mode_niv2" in group.columns else None,
                "mode_mrmt": group["mode_mrmt"].iloc[0] if "mode_mrmt" in group.columns else None,
                "candidate_sequence": int(seq),
                "point_count": point_count,
                "started_at": group["timestamp"].iloc[0],
                "finished_at": group["timestamp"].iloc[-1],
                "duration_seconds": duration_s,
                "distance_m": distance_m,
                "median_interval_seconds": median_interval,
                "kept": bool(keep),
            }
        )
        if keep:
            segmented_frames.append(group)

    stats_frame = pd.DataFrame(stats) if stats else pd.DataFrame(
        columns=SEGMENT_STAT_COLUMNS
    )
    if not segmented_frames:
        if not allow_empty:
            raise ValueError("No waypoint segment satisfied the configured segmentation filters.")
        empty = df.iloc[0:0].copy()
        empty["trip_candidate_id"] = pd.Series(dtype="object")
        return empty.reset_index(drop=True), stats_frame

    segmented = pd.concat(segmented_frames, ignore_index=True)
    max_segments = config["run"].get("max_segments")
    if max_segments is not None:
        keep_ids = sorted(segmented["trip_candidate_id"].unique())[: int(max_segments)]
        segmented = segmented[segmented["trip_candidate_id"].isin(keep_ids)].copy()

    return segmented.reset_index(drop=True), stats_frame


def _trim_static_tail(group: pd.DataFrame, seg_cfg: dict[str, Any]) -> pd.DataFrame:
    if not seg_cfg.get("trim_static_tail", True) or len(group) < 3:
        return group

    static_speed = float(seg_cfg["static_speed_kmh"])
    tail_seconds = float(seg_cfg["static_tail_minutes"]) * 60
    static_mask = group["speed_kmh"].le(static_speed).fillna(False).to_numpy()
    if not static_mask[-1]:
        return group

    start_pos = len(group) - 1
    while start_pos > 0 and static_mask[start_pos - 1]:
        start_pos -= 1

    tail_duration = (
        group["timestamp"].iloc[-1] - group["timestamp"].iloc[start_pos]
    ).total_seconds()
    if tail_duration >= tail_seconds and start_pos > 0:
        return group.iloc[: start_pos + 1].copy()
    return group


def to_relut_points(segmented: pd.DataFrame) -> pd.DataFrame:
    relut = segmented.copy()
    relut = relut.sort_values(["trip_candidate_id", "timestamp"]).reset_index(drop=True)
    metadata_columns = [column for column in LEG_METADATA_COLUMNS if column in relut.columns]
    return relut[RELUT_COLUMNS + metadata_columns + [
        "source_user_id",
        "source_file",
        "source_row_number",
        "distance_from_previous_m",
        "computed_speed_kmh",
        "raw_speed_kmh",
    ]].copy()


def write_prepared_outputs(
    relut_points: pd.DataFrame,
    segment_stats: pd.DataFrame,
    clean_report: dict[str, Any],
    config: dict[str, Any],
) -> None:
    from .config import output_path
    from .utils import write_json

    data_dir = output_path(config, "data")
    reports_dir = output_path(config, "reports")
    relut_points.to_parquet(data_dir / "prepared_relut_points.parquet", index=False)
    relut_points.to_csv(data_dir / "prepared_relut_points.csv", index=False)
    segment_stats.to_csv(data_dir / "segment_candidates.csv", index=False)
    write_json(reports_dir / "cleaning_report.json", clean_report)
