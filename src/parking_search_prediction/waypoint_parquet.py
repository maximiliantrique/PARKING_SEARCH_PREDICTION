"""Conversion des waypoints CSV en dataset Parquet partitionné par utilisateur.

Principe
--------
Chaque utilisateur est affecté à une partition stable ::

    partition = crc32(user_id en UTF-8) % n_partitions

`zlib.crc32` est déterministe (contrairement à `hash()` de Python, randomisé à
chaque session). Tous les waypoints d'un utilisateur tombent donc dans la même
partition, quel que soit le CSV d'origine : chaque partition peut être traitée
seule par `prepare()` sans changer le résultat.

La conversion est une **copie fidèle** : aucune ligne n'est filtrée, aucune
valeur n'est nettoyée. Les colonnes texte (identifiant utilisateur et timestamp
par défaut) restent du texte, toutes les autres sont lues en float64. Le
nettoyage reste entièrement dans `clean_waypoints`.

Deux colonnes de traçabilité sont ajoutées, calculées exactement comme dans
`read_raw_waypoints` (lecture pandas, numéro de ligne = position + 2) pour que
les `waypoint_id` restent identiques aux runs CSV :

- `source_file` : nom du CSV d'origine ;
- `source_row_number` : numéro de ligne CSV (la ligne 1 est l'en-tête).

Déroulé en deux passes
----------------------
1. **Fragments** (reprenable) : chaque CSV est lu puis écrit en un fichier par
   partition sous ``_fragments/<csv>/partition=XX.parquet``. Un marqueur
   ``_SUCCESS.json`` (taille + date de modification du CSV) permet de sauter un
   CSV déjà converti si la conversion est relancée après un plantage.
2. **Compaction** : chaque partition est relue, triée par
   (utilisateur, source_file, source_row_number) — c'est-à-dire l'ordre
   d'origine restreint à chaque utilisateur — et réécrite en un seul fichier
   ``partition=XX/part-0.parquet``.

Les lignes sans identifiant utilisateur vont dans ``partition=null`` : jamais
traitée, mais comptée dans le manifeste (elles seraient de toute façon
rejetées par `clean_waypoints`).

Les noms commençant par ``_`` (``_fragments``, ``_conversion_manifest.json``)
sont ignorés par pyarrow à la lecture du dataset.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import resolve_path
from .data_prep import resolve_delimiter

MANIFEST_NAME = "_conversion_manifest.json"
FRAGMENTS_DIR = "_fragments"
NULL_PARTITION = "null"
DEFAULT_N_PARTITIONS = 64
DEFAULT_COMPRESSION = "zstd"


# --------------------------------------------------------------------------- #
# Paramètres
# --------------------------------------------------------------------------- #


def conversion_settings(config: dict[str, Any]) -> dict[str, Any]:
    raw_cfg = config["raw_waypoints"]
    conv_cfg = config.get("parquet_conversion", {}) or {}
    dataset_dir = config.get("paths", {}).get("waypoint_parquet_dir")
    if not dataset_dir:
        raise KeyError(
            "paths.waypoint_parquet_dir doit être défini pour utiliser le dataset Parquet."
        )
    text_columns = conv_cfg.get("text_columns") or [raw_cfg["user_col"], raw_cfg["time_col"]]
    return {
        "dataset_dir": resolve_path(config, dataset_dir),
        "n_partitions": int(conv_cfg.get("n_partitions", DEFAULT_N_PARTITIONS)),
        "compression": conv_cfg.get("compression", DEFAULT_COMPRESSION),
        "keep_fragments": bool(conv_cfg.get("keep_fragments", True)),
        "delimiter": raw_cfg.get("delimiter", "auto"),
        "user_col": raw_cfg["user_col"],
        "text_columns": list(text_columns),
    }


def _fmt(n: int) -> str:
    return f"{n:,}".replace(",", "'")


def partition_label(index: int, n_partitions: int) -> str:
    width = max(2, len(str(n_partitions - 1)))
    return f"{index:0{width}d}"


def user_partition_labels(users: pd.Series, n_partitions: int) -> pd.Series:
    """Label de partition par ligne (calcul du crc32 une seule fois par utilisateur)."""
    labels = pd.Series(NULL_PARTITION, index=users.index, dtype="object")
    present = users.notna()
    if present.any():
        unique_users = pd.unique(users[present])
        mapping = {
            user: partition_label(zlib.crc32(str(user).encode("utf-8")) % n_partitions, n_partitions)
            for user in unique_users
        }
        labels[present] = users[present].map(mapping)
    return labels


# --------------------------------------------------------------------------- #
# Lecture d'un CSV avec schéma fixe
# --------------------------------------------------------------------------- #


def _csv_dtypes(path: Path, settings: dict[str, Any]) -> tuple[dict[str, str], list[str], list[str]]:
    header = pd.read_csv(path, sep=resolve_delimiter(path, settings["delimiter"]), nrows=0)
    columns = list(header.columns)
    missing = [column for column in settings["text_columns"] if column not in columns]
    if missing:
        raise KeyError(
            f"{path.name} : colonnes texte absentes de l'en-tête : {missing}. "
            f"Colonnes trouvées : {columns}"
        )
    text_columns = [column for column in columns if column in settings["text_columns"]]
    float_columns = [column for column in columns if column not in settings["text_columns"]]
    dtypes = {column: "object" for column in text_columns}
    dtypes.update({column: "float64" for column in float_columns})
    return dtypes, text_columns, float_columns


def read_csv_with_schema(path: Path, settings: dict[str, Any]) -> tuple[pd.DataFrame, list[str], list[str]]:
    dtypes, text_columns, float_columns = _csv_dtypes(path, settings)
    try:
        frame = pd.read_csv(path, sep=resolve_delimiter(path, settings["delimiter"]), dtype=dtypes)
    except ValueError as exc:
        raise ValueError(
            f"{path.name} : une colonne déclarée float64 contient une valeur non numérique "
            f"({exc}). Ajouter la colonne à parquet_conversion.text_columns si elle doit "
            "rester du texte."
        ) from exc
    # Identique à read_raw_waypoints : la ligne 1 du CSV est l'en-tête.
    frame["source_file"] = path.name
    frame["source_row_number"] = pd.Series(range(2, len(frame) + 2), index=frame.index, dtype="int64")
    return frame, text_columns, float_columns


def _enforce_schema(frame: pd.DataFrame, text_columns: list[str], float_columns: list[str]) -> pd.DataFrame:
    for column in text_columns:
        if column in frame.columns:
            frame[column] = frame[column].astype("object")
    for column in float_columns:
        if column in frame.columns:
            frame[column] = frame[column].astype("float64")
    frame["source_file"] = frame["source_file"].astype("object")
    frame["source_row_number"] = frame["source_row_number"].astype("int64")
    return frame


# --------------------------------------------------------------------------- #
# Passe 1 : fragments
# --------------------------------------------------------------------------- #


def _fragment_marker(csv_path: Path, settings: dict[str, Any]) -> dict[str, Any]:
    stat = csv_path.stat()
    return {
        "csv": csv_path.name,
        "size_bytes": int(stat.st_size),
        "mtime": float(stat.st_mtime),
        "n_partitions": settings["n_partitions"],
        "text_columns": settings["text_columns"],
    }


def _write_fragments(csv_path: Path, settings: dict[str, Any]) -> dict[str, Any]:
    fragments_root = settings["dataset_dir"] / FRAGMENTS_DIR
    final_dir = fragments_root / csv_path.name
    marker_path = final_dir / "_SUCCESS.json"
    expected = _fragment_marker(csv_path, settings)

    if marker_path.exists():
        done = json.loads(marker_path.read_text(encoding="utf-8"))
        if all(done.get(key) == value for key, value in expected.items()):
            done["skipped"] = True
            return done

    tmp_dir = fragments_root / f".tmp_{csv_path.name}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    started = time.perf_counter()
    frame, text_columns, float_columns = read_csv_with_schema(csv_path, settings)
    labels = user_partition_labels(frame[settings["user_col"]], settings["n_partitions"])

    partition_rows: dict[str, int] = {}
    for label, part in frame.groupby(labels, sort=True):
        part.to_parquet(
            tmp_dir / f"partition={label}.parquet",
            index=False,
            compression=settings["compression"],
        )
        partition_rows[str(label)] = int(len(part))

    marker = {
        **expected,
        "delimiter": resolve_delimiter(csv_path, settings["delimiter"]),
        "rows": int(len(frame)),
        "null_user_rows": int(partition_rows.get(NULL_PARTITION, 0)),
        "columns": list(frame.columns[:-2]),
        "text_columns_found": text_columns,
        "float_columns": float_columns,
        "partition_rows": partition_rows,
        "seconds": round(time.perf_counter() - started, 2),
    }
    (tmp_dir / "_SUCCESS.json").write_text(json.dumps(marker, indent=2), encoding="utf-8")
    if final_dir.exists():
        shutil.rmtree(final_dir)
    os.replace(tmp_dir, final_dir)
    marker["skipped"] = False
    return marker


# --------------------------------------------------------------------------- #
# Passe 2 : compaction
# --------------------------------------------------------------------------- #


def _compact_partition(
    label: str,
    csv_names: list[str],
    settings: dict[str, Any],
    text_columns: list[str],
    float_columns: list[str],
) -> dict[str, Any]:
    fragments_root = settings["dataset_dir"] / FRAGMENTS_DIR
    paths = [
        fragments_root / name / f"partition={label}.parquet"
        for name in csv_names
        if (fragments_root / name / f"partition={label}.parquet").exists()
    ]
    frames = [pd.read_parquet(path) for path in paths]
    frame = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    frame = _enforce_schema(frame, text_columns, float_columns)
    # Ordre d'origine (fichiers triés, puis lignes) restreint à chaque utilisateur :
    # c'est l'ordre dans lequel read_raw_waypoints présentait ces lignes.
    frame = frame.sort_values(
        [settings["user_col"], "source_file", "source_row_number"],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)

    out_dir = settings["dataset_dir"] / f"partition={label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = out_dir / ".part-0.parquet.tmp"
    frame.to_parquet(tmp_path, index=False, compression=settings["compression"])
    os.replace(tmp_path, out_dir / "part-0.parquet")
    return {
        "rows": int(len(frame)),
        "users": int(frame[settings["user_col"]].nunique(dropna=True)),
        "source_files": int(len(paths)),
        "parquet_bytes": int((out_dir / "part-0.parquet").stat().st_size),
    }


# --------------------------------------------------------------------------- #
# Point d'entrée
# --------------------------------------------------------------------------- #


def _manifest_id(files: list[dict[str, Any]], settings: dict[str, Any]) -> str:
    payload = {
        "files": [
            {key: item[key] for key in ("csv", "size_bytes", "mtime", "rows")} for item in files
        ],
        "n_partitions": settings["n_partitions"],
        "text_columns": settings["text_columns"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def convert_waypoints_to_parquet(
    config: dict[str, Any],
    *,
    files: list[Path] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Convertit les CSV de waypoints en dataset Parquet partitionné.

    ``files`` permet de convertir une liste explicite (utilisé par le test
    d'équivalence) ; par défaut tous les fichiers de ``paths.waypoint_glob``
    sont convertis, sans tenir compte de ``run.max_files``.
    """
    settings = conversion_settings(config)
    dataset_dir: Path = settings["dataset_dir"]
    if files is None:
        root = Path(config["_project_root"])
        files = sorted(root.glob(config["paths"]["waypoint_glob"]))
    files = sorted(Path(path) for path in files)
    if not files:
        raise FileNotFoundError(
            f"Aucun CSV trouvé pour paths.waypoint_glob = {config['paths']['waypoint_glob']!r} "
            f"(cherché depuis {config['_project_root']}). Utiliser '**/*.csv' si les "
            "fichiers sont dans des sous-dossiers."
        )

    dataset_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    def log(message: str) -> None:
        if verbose:
            print(f"[conversion] {message}", flush=True)

    total_bytes = sum(path.stat().st_size for path in files)
    log(f"{len(files)} CSV, {total_bytes / 1e9:.2f} GB → {dataset_dir}")
    log(f"{settings['n_partitions']} partitions, compression {settings['compression']}")

    # ---- Passe 1 ----------------------------------------------------------- #
    markers = []
    done_bytes = 0
    for index, path in enumerate(files, start=1):
        marker = _write_fragments(path, settings)
        markers.append(marker)
        done_bytes += path.stat().st_size
        status = "déjà converti" if marker.get("skipped") else f"{marker['seconds']:.0f} s"
        log(
            f"fragments {index}/{len(files)} {path.name} : {_fmt(marker['rows'])} lignes "
            f"({status}) — {done_bytes / total_bytes:.0%}"
        )

    # Cohérence des schémas entre fichiers
    reference = markers[0]
    text_columns = reference["text_columns_found"]
    float_columns = list(reference["float_columns"])
    schema_warnings = []
    for marker in markers[1:]:
        if marker["columns"] != reference["columns"]:
            schema_warnings.append(
                f"{marker['csv']} : colonnes {marker['columns']} ≠ {reference['columns']}"
            )
            for column in marker["float_columns"]:
                if column not in float_columns:
                    float_columns.append(column)
    for warning in schema_warnings:
        log(f"ATTENTION schéma : {warning}")

    manifest_id = _manifest_id(markers, settings)
    manifest_path = dataset_dir / MANIFEST_NAME
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("manifest_id") == manifest_id and all(
            (dataset_dir / f"partition={label}" / "part-0.parquet").exists()
            for label in previous.get("partitions", {})
        ):
            log(f"dataset déjà à jour (manifest {manifest_id}), compaction sautée")
            return previous

    # ---- Passe 2 ----------------------------------------------------------- #
    labels = sorted({label for marker in markers for label in marker["partition_rows"]})
    for stale in dataset_dir.glob("partition=*"):
        if stale.is_dir() and stale.name.split("=", 1)[1] not in labels:
            shutil.rmtree(stale)

    csv_names = [marker["csv"] for marker in markers]
    partitions: dict[str, Any] = {}
    for index, label in enumerate(labels, start=1):
        partitions[label] = _compact_partition(label, csv_names, settings, text_columns, float_columns)
        if index == len(labels) or index % 8 == 0:
            log(f"compaction {index}/{len(labels)} partitions")

    # ---- Contrôles et manifeste ------------------------------------------- #
    total_rows = sum(marker["rows"] for marker in markers)
    compacted_rows = sum(item["rows"] for item in partitions.values())
    if compacted_rows != total_rows:
        raise RuntimeError(
            f"Conversion incohérente : {compacted_rows} lignes compactées ≠ {total_rows} lignes CSV."
        )

    null_rows = int(partitions.get(NULL_PARTITION, {}).get("rows", 0))
    user_partitions = {key: value for key, value in partitions.items() if key != NULL_PARTITION}
    manifest = {
        "manifest_id": manifest_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_dir": str(dataset_dir),
        "partitioning": f"crc32(utf-8({settings['user_col']})) % {settings['n_partitions']}",
        "n_partitions": settings["n_partitions"],
        "user_col": settings["user_col"],
        "delimiter": settings["delimiter"],
        "compression": settings["compression"],
        "text_columns": text_columns,
        "float_columns": float_columns,
        "added_columns": {"source_file": "object", "source_row_number": "int64"},
        "schema_warnings": schema_warnings,
        "total_csv_bytes": int(total_bytes),
        "total_rows": int(total_rows),
        "null_user_rows": null_rows,
        "total_users": int(sum(item["users"] for item in user_partitions.values())),
        "non_empty_partitions": len(user_partitions),
        "total_parquet_bytes": int(sum(item["parquet_bytes"] for item in partitions.values())),
        "files": [
            {key: marker[key] for key in ("csv", "size_bytes", "mtime", "rows", "null_user_rows")}
            for marker in markers
        ],
        "partitions": partitions,
        "seconds": round(time.perf_counter() - started, 1),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not settings["keep_fragments"]:
        shutil.rmtree(dataset_dir / FRAGMENTS_DIR, ignore_errors=True)

    log(
        f"terminé en {manifest['seconds'] / 60:.1f} min : {_fmt(total_rows)} lignes, "
        f"{_fmt(manifest['total_users'])} utilisateurs, {_fmt(null_rows)} lignes sans "
        f"utilisateur, {manifest['total_parquet_bytes'] / 1e9:.2f} GB Parquet "
        f"(×{total_bytes / max(manifest['total_parquet_bytes'], 1):.1f} plus petit)"
    )
    return manifest


# --------------------------------------------------------------------------- #
# Lecture pour le traitement
# --------------------------------------------------------------------------- #


def load_manifest(config: dict[str, Any]) -> dict[str, Any]:
    settings = conversion_settings(config)
    manifest_path = settings["dataset_dir"] / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifeste introuvable : {manifest_path}. Lancer d'abord "
            "convert_waypoints_to_parquet(config)."
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def read_waypoint_partitions(config: dict[str, Any], labels: list[str]) -> pd.DataFrame:
    """Lit des partitions et reconstruit `waypoint_id` comme read_raw_waypoints."""
    settings = conversion_settings(config)
    frames = []
    for label in labels:
        path = settings["dataset_dir"] / f"partition={label}" / "part-0.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Partition absente : {path}")
        frames.append(pd.read_parquet(path))
    raw = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    raw["waypoint_id"] = (
        raw["source_file"] + ":" + raw["source_row_number"].astype(str).str.zfill(9)
    )
    return raw
