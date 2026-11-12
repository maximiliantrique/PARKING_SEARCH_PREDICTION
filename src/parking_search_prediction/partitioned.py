"""Mode partitionné de l'étape 01 (``raw_waypoints.source: parquet``).

Le dataset produit par `waypoint_parquet.convert_waypoints_to_parquet` est
traité par **tranches** de ``run.groups_per_run`` partitions. Chaque tranche
passe exactement par le même code que le mode CSV (`_match_and_filter` puis
`_segment_and_validate` dans `pipeline.py`). Comme un utilisateur n'appartient
qu'à une partition et qu'aucune étape ne compare deux utilisateurs, le résultat
global est identique au mode CSV, à l'ordre des lignes près.

Sorties (mêmes chemins qu'en mode CSV)
--------------------------------------
- ``waypoint_leg_assignments.parquet`` et ``waypoints_caronly.parquet`` sont
  des **dossiers** contenant un fichier par tranche (trop gros pour être
  concaténés) ; ``pd.read_parquet(chemin)`` les lit comme un seul fichier.
- ``prepared_relut_points.parquet/.csv``, ``segment_candidates.csv`` et
  ``relut_validation_by_segment.csv`` sont concaténés en un seul fichier :
  ``predict()`` et ``review()`` tournent sans modification.
- Les rapports JSON habituels sont réécrits avec les valeurs agrégées, et
  ``prepare_partitioned_summary.json`` donne la synthèse complète (métadonnées
  de run, contrôles de cohérence, détail par tranche).

Reprise
-------
Chaque tranche terminée écrit ``_DONE.json`` avec une empreinte (manifeste de
conversion, partitions de la tranche, sections de config utilisées). Au
relancement, les tranches dont l'empreinte est identique sont sautées. Une
modification du code n'est pas détectée : utiliser ``run.force_rerun: true``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ensure_output_dirs, output_path
from .data_prep import (
    discover_waypoint_files,
    normalize_user_ids,
    read_transport_legs,
    write_prepared_outputs,
)
from .pipeline import (
    ASSIGNMENT_COLUMNS,
    _enrichment_summary,
    load_enriched_legs,
    _match_and_filter,
    _mode_counts_frame,
    _prepare_summary,
    _progress,
    _remove_path,
    _segment_and_validate,
)
from .utils import write_json
from .waypoint_parquet import (
    NULL_PARTITION,
    conversion_settings,
    convert_waypoints_to_parquet,
    load_manifest,
    read_waypoint_partitions,
)

RUNS_DIR = "_partitioned_runs"
DEFAULT_GROUPS_PER_RUN = 4


# --------------------------------------------------------------------------- #
# Plan et empreinte
# --------------------------------------------------------------------------- #


def _run_settings(config: dict[str, Any]) -> dict[str, Any]:
    run_cfg = config.get("run", {}) or {}
    for key in ("max_users", "max_segments"):
        if run_cfg.get(key) is not None:
            raise ValueError(
                f"run.{key} = {run_cfg[key]!r} n'est pas supporté en mode partitionné "
                "(sa sémantique changerait d'une tranche à l'autre). Le mettre à None "
                "et utiliser run.max_partitions pour un run réduit."
            )
    groups_per_run = int(run_cfg.get("groups_per_run") or DEFAULT_GROUPS_PER_RUN)
    if groups_per_run < 1:
        raise ValueError("run.groups_per_run doit être >= 1.")
    max_partitions = run_cfg.get("max_partitions")
    return {
        "groups_per_run": groups_per_run,
        "max_partitions": int(max_partitions) if max_partitions is not None else None,
        "force_rerun": bool(run_cfg.get("force_rerun", False)),
    }


def _plan_runs(manifest: dict[str, Any], settings: dict[str, Any]) -> list[list[str]]:
    labels = sorted(label for label in manifest["partitions"] if label != NULL_PARTITION)
    if settings["max_partitions"] is not None:
        labels = labels[: settings["max_partitions"]]
    size = settings["groups_per_run"]
    return [labels[start : start + size] for start in range(0, len(labels), size)]


def _fingerprint(config: dict[str, Any], manifest: dict[str, Any], labels: list[str]) -> str:
    payload = {
        "manifest_id": manifest["manifest_id"],
        "partitions": labels,
        "legs_file": config["paths"].get("legs_file"),
        "sections": {
            key: config.get(key)
            for key in (
                "raw_waypoints",
                "cleaning",
                "leg_matching",
                "leg_enrichment",
                "pl23_survey",
                "leg_filters",
                "segmentation",
                "relut_validation",
                "spatial_detection",
            )
        },
    }
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _peak_rss_gb() -> float | None:
    """Pic mémoire du processus depuis son lancement (pas propre à la tranche)."""
    try:
        import resource
    except ImportError:  # Windows
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / 1e9 if sys.platform == "darwin" else peak / 1e6, 2)


# --------------------------------------------------------------------------- #
# Une tranche
# --------------------------------------------------------------------------- #


def _part_name(run_index: int) -> str:
    return f"part-run{run_index:04d}.parquet"


def _run_one(
    run_index: int,
    n_runs: int,
    labels: list[str],
    tracks_all: pd.DataFrame,
    config: dict[str, Any],
    paths: dict[str, Path],
    walk_lookup: pd.DataFrame | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    prefix = f"[prepare][{run_index + 1}/{n_runs}]"
    user_col = config["raw_waypoints"]["user_col"]

    raw = read_waypoint_partitions(config, labels)
    raw_rows = int(len(raw))
    users = normalize_user_ids(pd.Series(pd.unique(raw[user_col].dropna()), dtype=object))
    _progress(
        "1. lecture partitions",
        raw,
        prefix=prefix,
        partitions=f"{labels[0]}–{labels[-1]}",
        utilisateurs=int(len(users)),
    )
    tracks = tracks_all[tracks_all["source_user_id"].isin(users)]
    _progress("3. legs de la tranche", prefix=prefix, legs=len(tracks))

    stage = _match_and_filter(raw, tracks, config, prefix=prefix)
    del raw
    matched = stage.pop("matched")
    if config["leg_matching"].get("write_assignment_table", True):
        columns = [column for column in ASSIGNMENT_COLUMNS if column in matched.columns]
        matched[columns].to_parquet(paths["assignments"] / _part_name(run_index), index=False)
    walk_waypoints = 0
    if walk_lookup is not None and not walk_lookup.empty:
        from .leg_enrichment import post_car_walk_waypoints

        walk = post_car_walk_waypoints(matched, walk_lookup, config)
        walk.to_parquet(paths["walk"] / _part_name(run_index), index=False)
        walk_waypoints = int(len(walk))
        _progress("4b. waypoints marche après voiture", walk, prefix=prefix)
        del walk
    del matched
    car_waypoints = stage.pop("car_waypoints")
    car_waypoints.to_parquet(paths["caronly"] / _part_name(run_index), index=False)

    seg = _segment_and_validate(car_waypoints, config, prefix=prefix, allow_empty=True)
    del car_waypoints

    work_dir = paths["runs"] / f"run_{run_index:04d}"
    work_dir.mkdir(parents=True, exist_ok=True)
    seg["relut_points"].to_parquet(work_dir / "relut_points.parquet", index=False)
    seg["segment_candidates"].to_parquet(work_dir / "segment_candidates.parquet", index=False)
    seg["validation"].to_parquet(work_dir / "validation.parquet", index=False)
    stage["leg_stats"].to_parquet(work_dir / "leg_stats.parquet", index=False)
    pd.DataFrame({"leg_id": sorted(stage["car_legs_with_waypoints"])}).to_parquet(work_dir / "car_legs.parquet", index=False)

    report = {
        "run": run_index,
        "partitions": labels,
        "raw_rows": raw_rows,
        "users": int(len(users)),
        "tracks": int(len(tracks)),
        "cleaning_report": stage["cleaning_report"],
        "matching_report": stage["matching_report"],
        "car_filter_report": stage["car_filter_report"],
        "study_area_report": stage["study_area_report"],
        "loop_filter_report": seg["loop_filter_report"],
        "validation_summary": seg["validation_summary"],
        "prepared_points": int(len(seg["relut_points"])),
        "prepared_segments": int(seg["relut_points"]["trip_candidate_id"].nunique()),
        "post_car_walk_waypoints": walk_waypoints,
        "seconds": round(time.perf_counter() - started, 1),
        "process_peak_rss_gb": _peak_rss_gb(),
    }
    write_json(work_dir / "report.json", report)
    return report


# --------------------------------------------------------------------------- #
# Contrôle préalable : les utilisateurs des waypoints existent-ils dans les legs ?
# --------------------------------------------------------------------------- #


def _waypoint_user_ids(config: dict[str, Any], labels: list[str]) -> pd.Series:
    """Identifiants utilisateur distincts des partitions, sans charger les waypoints."""
    import pyarrow.parquet as pq

    settings = conversion_settings(config)
    user_col = settings["user_col"]
    ids: set[Any] = set()
    for label in labels:
        path = settings["dataset_dir"] / f"partition={label}" / "part-0.parquet"
        column = pq.read_table(path, columns=[user_col], read_dictionary=[user_col]).column(0)
        for chunk in column.chunks:
            values = chunk.dictionary if hasattr(chunk, "dictionary") else chunk.unique()
            ids.update(value for value in values.to_pylist() if value is not None)
    return normalize_user_ids(pd.Series(sorted(ids, key=str), dtype=object)).dropna()


def _check_user_overlap(
    config: dict[str, Any], labels: list[str], tracks_all: pd.DataFrame
) -> dict[str, Any]:
    started = time.perf_counter()
    waypoint_users = _waypoint_user_ids(config, labels)
    legs_users = pd.Series(tracks_all["source_user_id"].dropna().unique(), dtype="string")
    exact = int(waypoint_users.isin(legs_users).sum())
    case_insensitive = int(waypoint_users.str.lower().isin(legs_users.str.lower()).sum())
    stats = {
        "waypoint_users": int(len(waypoint_users)),
        "legs_users": int(len(legs_users)),
        "waypoint_users_with_legs": exact,
        "share_with_legs": round(exact / len(waypoint_users), 4) if len(waypoint_users) else 0.0,
        "matches_if_case_ignored": case_insensitive,
        "seconds": round(time.perf_counter() - started, 1),
    }
    _progress(
        "0. contrôle utilisateurs",
        prefix="[prepare]",
        waypoints=stats["waypoint_users"],
        legs=stats["legs_users"],
        communs=exact,
        part=f"{stats['share_with_legs']:.0%}",
    )
    if exact == 0:
        def show(series: pd.Series) -> str:
            sample = series.head(3).tolist()
            return ", ".join(f"{value!r} (longueur {len(value)})" for value in sample)

        hints = []
        if case_insensitive:
            hints.append(
                f"{case_insensitive} identifiants correspondent si on ignore la casse : "
                "majuscules d'un côté, minuscules de l'autre."
            )
        hints.append(
            f"Vérifier leg_matching.user_col / la source de legs "
            f"(source={config['leg_matching'].get('source')!r}, legs_file="
            f"{config['paths'].get('legs_file')!r}) et raw_waypoints.user_col "
            f"({config['raw_waypoints']['user_col']!r})."
        )
        raise ValueError(
            "Aucun utilisateur des waypoints n'a de leg : le run produirait 0 waypoint apparié.\n"
            f"  waypoints ({stats['waypoint_users']} utilisateurs) : {show(waypoint_users)}\n"
            f"  legs ({stats['legs_users']} utilisateurs) : {show(legs_users)}\n  "
            + "\n  ".join(hints)
        )
    if stats["share_with_legs"] < 0.5:
        print(
            f"[prepare] ATTENTION : seuls {stats['share_with_legs']:.0%} des utilisateurs des "
            "waypoints ont des legs. Vérifier que les deux fichiers viennent de la même vague.",
            flush=True,
        )
    return stats


# --------------------------------------------------------------------------- #
# Agrégation des rapports
# --------------------------------------------------------------------------- #

SUM, DICT_SUM, SAME, RECOMPUTE = "sum", "dict_sum", "same", "recompute"

# Règles explicites par rapport. Les compteurs s'additionnent car les
# utilisateurs (donc les legs et les segments) ne se recoupent pas entre
# tranches ; les paramètres doivent être identiques ; les ratios et médianes
# sont recalculés sur les totaux ou les tables assemblées.
AGGREGATION_RULES: dict[str, dict[str, str]] = {
    "cleaning_report": {
        "raw_rows": SUM,
        "raw_users": SUM,
        "dropped_required_or_bounds": SUM,
        "dropped_accuracy": SUM,
        "dropped_duplicates": SUM,
        "clean_rows": SUM,
        "clean_users": SUM,
        "speed_missing_or_unknown_rows": SUM,
        "speed_capped_rows": SUM,
    },
    "matching_report": {
        "waypoint_rows": SUM,
        "waypoint_users": SUM,
        "matched_waypoints": SUM,
        "matched_share": RECOMPUTE,
        "status_counts": DICT_SUM,
        "matched_legs": SUM,
    },
    "car_filter_report": {
        "car_modes": SAME,
        "matched_waypoints_by_mode": DICT_SUM,
        "car_waypoints": SUM,
        "car_legs": SUM,
        "car_users": SUM,
    },
    "study_area_report": {
        "study_area_filter_configured": SAME,
        "study_area_boundary_file": SAME,
        "study_area_boundary_layer": SAME,
        "study_area_total_tracks": RECOMPUTE,
        "study_area_total_car_legs": SUM,
        "study_area_car_legs_with_destination": SUM,
        "study_area_car_legs_in_area": SUM,
        "study_area_car_legs_out_of_area": SUM,
        "study_area_car_legs_missing_destination": SUM,
        "study_area_filter_applied": SAME,
        "study_area_car_waypoints_before_filter": SUM,
        "study_area_car_waypoints_after_filter": SUM,
        "study_area_car_legs_after_filter": SUM,
    },
    "loop_filter_report": {
        "loop_filter_radius_m": SAME,
        "loop_filter_applied": SAME,
        "loop_total_kept_segments": SUM,
        "loop_segments_detected": SUM,
        "loop_segments_after_filter": SUM,
        "loop_points_before_filter": SUM,
        "loop_points_after_filter": SUM,
    },
    "validation_summary": {
        "segments_total": SUM,
        "segments_relut_ready": SUM,
        "segments_not_ready": SUM,
        "points_total": SUM,
        "median_interval_overall_seconds": RECOMPUTE,
    },
}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _sum(values: list[Any]) -> Any:
    present = [value for value in values if value is not None]
    if not present:
        return None
    total = sum(present)
    return int(total) if all(isinstance(value, int) for value in present) else float(total)


def _dict_sum(values: list[Any]) -> dict[str, Any]:
    total: dict[str, Any] = {}
    for value in values:
        for key, count in (value or {}).items():
            total[key] = total.get(key, 0) + count
    return dict(sorted(total.items(), key=lambda item: (-item[1], item[0])))


def _aggregate(name: str, reports: list[dict[str, Any]], warnings: list[str]) -> dict[str, Any]:
    rules = AGGREGATION_RULES[name]
    keys: list[str] = []
    for report in reports:
        keys.extend(key for key in report if key not in keys)
    combined: dict[str, Any] = {}
    for key in keys:
        values = [report[key] for report in reports if key in report]
        rule = rules.get(key)
        if rule is None:
            if all(_is_number(value) or value is None for value in values):
                rule = SUM
            elif all(isinstance(value, dict) for value in values):
                rule = DICT_SUM
            else:
                rule = SAME
            warnings.append(f"{name}.{key} : clé sans règle explicite, agrégée en '{rule}'.")
        if rule == SUM:
            combined[key] = _sum(values)
        elif rule == DICT_SUM:
            combined[key] = _dict_sum(values)
        elif rule == SAME:
            distinct = {json.dumps(value, sort_keys=True, default=str) for value in values}
            if len(distinct) > 1:
                warnings.append(f"{name}.{key} : valeurs différentes entre tranches {sorted(distinct)}.")
            combined[key] = values[0]
        else:  # RECOMPUTE : rempli par l'appelant
            combined[key] = None
    return combined


def _combine_reports(
    reports: list[dict[str, Any]],
    *,
    legs_report: dict[str, Any],
    total_tracks: int,
    null_user_rows: int,
    include_null_rows: bool,
    relut_points: pd.DataFrame,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    warnings: list[str] = []
    combined = {name: _aggregate(name, [r[name] for r in reports], warnings) for name in AGGREGATION_RULES}

    # Lignes sans utilisateur : jamais traitées, mais comptées comme brutes puis
    # rejetées (« required_or_bounds »), exactement comme en mode CSV.
    cleaning = combined["cleaning_report"]
    if include_null_rows and null_user_rows:
        cleaning["raw_rows"] = (cleaning.get("raw_rows") or 0) + null_user_rows
        cleaning["dropped_required_or_bounds"] = (
            cleaning.get("dropped_required_or_bounds") or 0
        ) + null_user_rows

    matching = combined["matching_report"]
    rows = matching.get("waypoint_rows") or 0
    matching["matched_share"] = float((matching.get("matched_waypoints") or 0) / rows) if rows else 0.0

    # Ce compteur porte sur la table des legs entière, pas sur une tranche.
    if "study_area_total_tracks" in combined["study_area_report"]:
        combined["study_area_report"]["study_area_total_tracks"] = int(total_tracks)

    combined["validation_summary"]["median_interval_overall_seconds"] = float(
        relut_points.groupby("trip_candidate_id")["timestamp"].diff().dt.total_seconds().median()
    ) if len(relut_points) else float("nan")

    combined["legs_report"] = legs_report
    return combined, warnings


# --------------------------------------------------------------------------- #
# Assemblage
# --------------------------------------------------------------------------- #


def _concat_runs(paths: list[Path]) -> pd.DataFrame:
    frames = [frame for frame in (pd.read_parquet(path) for path in paths) if len(frame)]
    if not frames:
        return pd.read_parquet(paths[0]) if paths else pd.DataFrame()
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def prepare_partitioned(config: dict[str, Any]) -> dict[str, Any]:
    ensure_output_dirs(config)
    settings = _run_settings(config)
    manifest = load_manifest(config)
    runs = _plan_runs(manifest, settings)
    if not runs:
        raise ValueError("Aucune partition à traiter dans le dataset Parquet.")
    all_partitions_processed = settings["max_partitions"] is None

    started = time.perf_counter()
    _progress(
        "démarrage (source Parquet)",
        prefix="[prepare]",
        partitions=sum(len(run) for run in runs),
        tranches=len(runs),
        groups_per_run=settings["groups_per_run"],
        manifest=manifest["manifest_id"],
    )

    paths = {
        "assignments": output_path(config, "data", "waypoint_leg_assignments.parquet"),
        "caronly": output_path(config, "data", "waypoints_caronly.parquet"),
        "walk": output_path(config, "data", "waypoints_post_car_walk.parquet"),
        "runs": output_path(config, "data", RUNS_DIR),
    }
    for key in ("assignments", "caronly"):
        if paths[key].is_file():  # sortie d'un run CSV précédent
            paths[key].unlink()
        paths[key].mkdir(parents=True, exist_ok=True)
    paths["runs"].mkdir(parents=True, exist_ok=True)

    tracks_all, legs_report, enrichment_report, walk_lookup = load_enriched_legs(config)
    write_walk = not walk_lookup.empty
    if paths["walk"].is_file() or (paths["walk"].exists() and not write_walk):
        _remove_path(paths["walk"])
    if write_walk:
        paths["walk"].mkdir(parents=True, exist_ok=True)
    user_overlap = _check_user_overlap(config, [label for run in runs for label in run], tracks_all)

    write_assignments = config["leg_matching"].get("write_assignment_table", True)
    reports = []
    for run_index, labels in enumerate(runs):
        work_dir = paths["runs"] / f"run_{run_index:04d}"
        done_path = work_dir / "_DONE.json"
        fingerprint = _fingerprint(config, manifest, labels)
        expected_files = [
            work_dir / "report.json",
            work_dir / "relut_points.parquet",
            work_dir / "segment_candidates.parquet",
            work_dir / "validation.parquet",
            work_dir / "leg_stats.parquet",
            work_dir / "car_legs.parquet",
            paths["caronly"] / _part_name(run_index),
        ] + ([paths["assignments"] / _part_name(run_index)] if write_assignments else []) + (
            [paths["walk"] / _part_name(run_index)] if write_walk else []
        )
        if (
            not settings["force_rerun"]
            and done_path.exists()
            and json.loads(done_path.read_text(encoding="utf-8")).get("fingerprint") == fingerprint
            and all(path.exists() for path in expected_files)
        ):
            report = json.loads((work_dir / "report.json").read_text(encoding="utf-8"))
            report["skipped"] = True
            print(f"[prepare][{run_index + 1}/{len(runs)}] déjà calculée, sautée", flush=True)
        else:
            if done_path.exists():
                done_path.unlink()
            report = _run_one(run_index, len(runs), labels, tracks_all, config, paths, walk_lookup)
            write_json(
                done_path,
                {
                    "fingerprint": fingerprint,
                    "partitions": labels,
                    "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                },
            )
            report["skipped"] = False
            elapsed = time.perf_counter() - started
            remaining = elapsed / (run_index + 1) * (len(runs) - run_index - 1)
            print(
                f"[prepare][{run_index + 1}/{len(runs)}] terminée en {report['seconds']:.0f} s "
                f"— pic RAM processus {report['process_peak_rss_gb']} GB "
                f"— reste ~{remaining / 60:.0f} min",
                flush=True,
            )
        reports.append(report)

    # Parties obsolètes (tranches d'un plan précédent)
    valid_parts = {_part_name(index) for index in range(len(runs))}
    for key in ("assignments", "caronly", "walk"):
        if not paths[key].is_dir():
            continue
        for part in paths[key].glob("part-run*.parquet"):
            if part.name not in valid_parts:
                part.unlink()
    for stale in paths["runs"].glob("run_*"):
        if stale.is_dir() and int(stale.name.split("_")[1]) >= len(runs):
            shutil.rmtree(stale)

    # ---- Tables concaténées ------------------------------------------------ #
    _progress("assemblage des tranches", prefix="[prepare]")
    run_dirs = [paths["runs"] / f"run_{index:04d}" for index in range(len(runs))]
    relut_points = _concat_runs([d / "relut_points.parquet" for d in run_dirs])
    relut_points = relut_points.sort_values(["trip_candidate_id", "timestamp"]).reset_index(drop=True)
    segment_candidates = _concat_runs([d / "segment_candidates.parquet" for d in run_dirs])
    if len(segment_candidates):
        segment_candidates = segment_candidates.sort_values(
            "source_user_id", kind="mergesort"
        ).reset_index(drop=True)
    validation = _concat_runs([d / "validation.parquet" for d in run_dirs])
    if len(validation):
        validation = validation.sort_values("trip_candidate_id", kind="mergesort").reset_index(drop=True)

    # ---- Rapports ---------------------------------------------------------- #
    processed_labels = [label for run in runs for label in run]
    null_user_rows = int(manifest.get("null_user_rows", 0))
    combined, warnings = _combine_reports(
        reports,
        legs_report=legs_report,
        total_tracks=len(tracks_all),
        null_user_rows=null_user_rows,
        include_null_rows=all_partitions_processed,
        relut_points=relut_points,
    )

    expected_rows = sum(manifest["partitions"][label]["rows"] for label in processed_labels)
    expected_users = sum(manifest["partitions"][label]["users"] for label in processed_labels)
    processed_rows = sum(report["raw_rows"] for report in reports)
    checks = {
        "raw_rows_match_manifest": {
            "ok": processed_rows == expected_rows,
            "processed": processed_rows,
            "manifest": expected_rows,
        },
        "users_disjoint_between_runs": {
            "ok": sum(report["users"] for report in reports) == expected_users,
            "sum_over_runs": sum(report["users"] for report in reports),
            "manifest": expected_users,
        },
        "segment_ids_unique": {
            "ok": bool(validation["trip_candidate_id"].is_unique) if len(validation) else True,
            "segments": int(len(validation)),
        },
        "waypoint_ids_unique_in_relut_points": {
            "ok": bool(relut_points["waypoint_id"].is_unique) if len(relut_points) else True,
        },
        "segments_match_validation": {
            "ok": combined["validation_summary"]["segments_total"] == int(len(validation)),
            "summed": combined["validation_summary"]["segments_total"],
            "assembled": int(len(validation)),
        },
    }

    # ---- Écriture ---------------------------------------------------------- #
    write_prepared_outputs(relut_points, segment_candidates, combined["cleaning_report"], config)
    from .pipeline import finalize_corpus

    corpus_stats = _concat_runs([d / "leg_stats.parquet" for d in run_dirs])
    car_legs = set(_concat_runs([d / "car_legs.parquet" for d in run_dirs]).get("leg_id", pd.Series(dtype=object)))
    corpus_summary = finalize_corpus(config, tracks_all, corpus_stats, car_legs)
    validation.to_csv(output_path(config, "reports", "relut_validation_by_segment.csv"), index=False)
    write_json(
        output_path(config, "reports", "relut_validation_summary.json"),
        combined["validation_summary"],
    )
    write_json(
        output_path(config, "reports", "spatial_study_area_filter_summary.json"),
        combined["study_area_report"],
    )
    write_json(output_path(config, "reports", "loop_filter_summary.json"), combined["loop_filter_report"])
    write_json(
        output_path(config, "reports", "waypoint_leg_matching_summary.json"),
        {
            **legs_report,
            **combined["matching_report"],
            **combined["car_filter_report"],
            **combined["study_area_report"],
        },
    )
    _mode_counts_frame(combined["car_filter_report"]).to_csv(
        output_path(config, "reports", "waypoint_leg_matching_by_mode.csv"), index=False
    )

    result = _prepare_summary(
        raw_rows=combined["cleaning_report"]["raw_rows"],
        clean_rows=combined["cleaning_report"]["clean_rows"],
        matching_report=combined["matching_report"],
        car_filter_report=combined["car_filter_report"],
        study_area_report=combined["study_area_report"],
        loop_filter_report=combined["loop_filter_report"],
        validation_summary=combined["validation_summary"],
        prepared_points=len(relut_points),
        prepared_segments=relut_points["trip_candidate_id"].nunique() if len(relut_points) else 0,
    )
    synthesis = {
        "mode": "parquet_partitioned",
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total_seconds": round(time.perf_counter() - started, 1),
        "dataset": {
            "dir": manifest["dataset_dir"],
            "manifest_id": manifest["manifest_id"],
            "n_partitions": manifest["n_partitions"],
            "partitioning": manifest["partitioning"],
            "total_rows": manifest["total_rows"],
            "null_user_rows": null_user_rows,
        },
        "plan": {
            "groups_per_run": settings["groups_per_run"],
            "max_partitions": settings["max_partitions"],
            "runs": len(runs),
            "partitions_processed": len(processed_labels),
            "all_partitions_processed": all_partitions_processed,
            "null_user_rows_counted_in_cleaning_report": all_partitions_processed,
        },
        "user_overlap": user_overlap,
        "leg_enrichment": enrichment_report,
        "post_car_walk_waypoints": int(sum(r.get("post_car_walk_waypoints", 0) for r in reports)),
        "checks": checks,
        "aggregation_warnings": warnings,
        "result": result,
        "reports": combined,
        "runs": [
            {
                "run": report["run"],
                "partitions": report["partitions"],
                "skipped": report.get("skipped", False),
                "raw_rows": report["raw_rows"],
                "users": report["users"],
                "car_waypoints": report["car_filter_report"]["car_waypoints"],
                "prepared_segments": report["prepared_segments"],
                "seconds": report["seconds"],
                "process_peak_rss_gb": report["process_peak_rss_gb"],
            }
            for report in reports
        ],
    }
    write_json(output_path(config, "reports", "prepare_partitioned_summary.json"), synthesis)

    for warning in warnings:
        print(f"[prepare] ATTENTION agrégation : {warning}", flush=True)
    failed = [name for name, check in checks.items() if not check["ok"]]
    _progress(
        "10. sorties écrites",
        prefix="[prepare]",
        minutes=round(synthesis["total_seconds"] / 60, 1),
        contrôles="OK" if not failed else f"ÉCHEC {failed}",
    )
    if failed:
        raise RuntimeError(
            f"Contrôles de cohérence en échec : {failed}. Détail dans "
            f"{output_path(config, 'reports', 'prepare_partitioned_summary.json')}"
        )
    return {
        **result,
        **_enrichment_summary(enrichment_report, sum(r.get("post_car_walk_waypoints", 0) for r in reports)),
        "corpus_legs": corpus_summary["in_corpus"],
        "partitioned": synthesis["plan"],
    }


# --------------------------------------------------------------------------- #
# Test d'équivalence CSV ↔ Parquet partitionné
# --------------------------------------------------------------------------- #

_COMPARED_TABLES = [
    # (fichier, sous-dossier, clés de tri)
    ("prepared_relut_points.parquet", "data", ["trip_candidate_id", "timestamp", "waypoint_id"]),
    ("segment_candidates.csv", "data", ["trip_candidate_id"]),
    ("relut_validation_by_segment.csv", "reports", ["trip_candidate_id"]),
    ("waypoints_caronly.parquet", "data", ["waypoint_id"]),
    ("waypoint_leg_assignments.parquet", "data", ["waypoint_id"]),
    ("waypoints_post_car_walk.parquet", "data", ["waypoint_id"]),
    ("legs_enriched.parquet", "data", ["leg_id"]),
    ("corpus_by_leg.parquet", "legs", ["leg_id"]),
]
_COMPARED_REPORTS = [
    "cleaning_report.json",
    "leg_enrichment_summary.json",
    "waypoint_leg_matching_summary.json",
    "spatial_study_area_filter_summary.json",
    "loop_filter_summary.json",
    "relut_validation_summary.json",
]


def _read_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
    # Même instant, résolution différente (µs / ns) selon le chemin de lecture :
    # ce n'est pas une différence de résultat, on aligne avant de comparer.
    for column in frame.columns:
        dtype = frame[column].dtype
        if isinstance(dtype, pd.DatetimeTZDtype):
            frame[column] = frame[column].astype(pd.DatetimeTZDtype("ns", dtype.tz))
        elif pd.api.types.is_datetime64_dtype(dtype):
            frame[column] = frame[column].astype("datetime64[ns]")
    return frame


def _diff_values(a: Any, b: Any, path: str, out: list[str]) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                out.append(f"{path}.{key} : présent d'un seul côté (CSV={a.get(key)!r}, Parquet={b.get(key)!r})")
            else:
                _diff_values(a[key], b[key], f"{path}.{key}", out)
        return
    if _is_number(a) and _is_number(b):
        if (isinstance(a, float) and math.isnan(a)) and (isinstance(b, float) and math.isnan(b)):
            return
        if math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9):
            return
    elif a == b:
        return
    out.append(f"{path} : CSV={a!r} ≠ Parquet={b!r}")


def check_partitioned_equivalence(
    config: dict[str, Any],
    *,
    n_files: int = 2,
    groups_per_run: int | None = None,
    work_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Compare le mode CSV et le mode partitionné sur les ``n_files`` premiers CSV.

    Les deux modes écrivent dans des dossiers séparés sous ``work_dir`` (par
    défaut ``<output_dir>/_equivalence_test``) ; le dataset Parquet de test est
    reconstruit à chaque appel. ``groups_per_run`` vaut par défaut environ un
    tiers des partitions non vides, pour forcer plusieurs tranches et donc
    tester l'agrégation.

    Lève ``AssertionError`` avec la liste des différences si les résultats ne
    sont pas identiques.
    """
    base = Path(work_dir) if work_dir else output_path(config, "_equivalence_test")
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True)

    files_cfg = copy.deepcopy(config)
    files_cfg["run"]["max_files"] = n_files
    files = discover_waypoint_files(files_cfg)
    if not files:
        root = Path(config["_project_root"])
        raise FileNotFoundError(
            "Aucun CSV trouvé pour le test d'équivalence : paths.waypoint_glob = "
            f"{config['paths']['waypoint_glob']!r}, cherché depuis {root}. Vérifier le "
            "chemin (relatif à la racine du projet), l'extension (.csv) et utiliser "
            "'**/*.csv' si les fichiers sont dans des sous-dossiers."
        )
    print(f"[équivalence] {len(files)} CSV : {[path.name for path in files]}", flush=True)

    cfg_csv = copy.deepcopy(config)
    cfg_csv["raw_waypoints"]["source"] = "csv"
    cfg_csv["paths"]["output_dir"] = str(base / "csv_mode")
    cfg_csv["run"].update({"max_files": n_files, "max_users": None, "max_segments": None})

    cfg_pq = copy.deepcopy(config)
    cfg_pq["raw_waypoints"]["source"] = "parquet"
    cfg_pq["paths"]["output_dir"] = str(base / "parquet_mode")
    cfg_pq["paths"]["waypoint_parquet_dir"] = str(base / "dataset")
    cfg_pq.setdefault("parquet_conversion", {})["keep_fragments"] = False
    cfg_pq["run"].update(
        {"max_files": None, "max_users": None, "max_segments": None, "max_partitions": None, "force_rerun": True}
    )

    from .pipeline import prepare_csv

    print("[équivalence] — conversion du sous-ensemble", flush=True)
    manifest = convert_waypoints_to_parquet(cfg_pq, files=files)
    n_nonempty = manifest["non_empty_partitions"]
    cfg_pq["run"]["groups_per_run"] = groups_per_run or max(1, math.ceil(n_nonempty / 3))
    print(
        f"[équivalence] {n_nonempty} partitions non vides, groups_per_run="
        f"{cfg_pq['run']['groups_per_run']}",
        flush=True,
    )

    print("[équivalence] — mode CSV", flush=True)
    summary_csv = prepare_csv(cfg_csv)
    print("[équivalence] — mode Parquet partitionné", flush=True)
    summary_pq = prepare_partitioned(cfg_pq)

    differences: list[str] = []
    _diff_values(summary_csv, {k: summary_pq.get(k) for k in summary_csv}, "prepare()", differences)

    for name in _COMPARED_REPORTS:
        a = json.loads(output_path(cfg_csv, "reports", name).read_text(encoding="utf-8"))
        b = json.loads(output_path(cfg_pq, "reports", name).read_text(encoding="utf-8"))
        _diff_values(a, b, name, differences)

    for name, subdir, keys in _COMPARED_TABLES:
        path_csv = output_path(cfg_csv, subdir, name)
        path_pq = output_path(cfg_pq, subdir, name)
        if not path_csv.exists() or not path_pq.exists():
            if path_csv.exists() != path_pq.exists():
                differences.append(f"{name} : présent d'un seul côté")
            continue
        a, b = _read_table(path_csv), _read_table(path_pq)
        # les identifiants de run (horodatés) diffèrent forcément entre les deux modes
        a = a.drop(columns=[c for c in a.columns if c.endswith("_run_id")])
        b = b.drop(columns=[c for c in b.columns if c.endswith("_run_id")])
        if list(a.columns) != list(b.columns):
            differences.append(f"{name} : colonnes différentes {list(a.columns)} ≠ {list(b.columns)}")
            continue
        if len(a) != len(b):
            differences.append(f"{name} : {len(a)} lignes (CSV) ≠ {len(b)} lignes (Parquet)")
            continue
        sort_keys = [key for key in keys if key in a.columns]
        a = a.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)
        b = b.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)
        try:
            pd.testing.assert_frame_equal(a, b, check_dtype=False, check_exact=False, rtol=1e-9)
        except AssertionError as exc:
            differences.append(f"{name} : {str(exc).splitlines()[0:6]}")

    result = {
        "files": [path.name for path in files],
        "partitions": n_nonempty,
        "runs": summary_pq["partitioned"]["runs"],
        "tables_compared": [name for name, _, _ in _COMPARED_TABLES],
        "reports_compared": _COMPARED_REPORTS,
        "differences": differences,
        "equivalent": not differences,
        "work_dir": str(base),
    }
    write_json(base / "equivalence_result.json", result)
    if differences:
        print("\n❌ ÉQUIVALENCE EN ÉCHEC — différences entre mode CSV et mode Parquet :", flush=True)
        for line in differences:
            print(f"   - {line}", flush=True)
        raise AssertionError(
            f"{len(differences)} différence(s) entre les modes CSV et Parquet. "
            f"Détail : {base / 'equivalence_result.json'}"
        )
    print(
        f"\n✅ ÉQUIVALENCE OK — {len(files)} CSV, {n_nonempty} partitions en "
        f"{result['runs']} tranches : tables et rapports identiques.",
        flush=True,
    )
    return result
