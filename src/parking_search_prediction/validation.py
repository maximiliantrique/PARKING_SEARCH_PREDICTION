from __future__ import annotations

from typing import Any

import pandas as pd


def validate_relut_segments(relut_points: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    cfg = config["relut_validation"]
    rows = []
    for trip_id, group in relut_points.groupby("trip_candidate_id", sort=True):
        group = group.sort_values("timestamp")
        duration = group["timestamp"].iloc[-1] - group["timestamp"].iloc[0]
        intervals = group["timestamp"].diff().dt.total_seconds().dropna()
        median_interval = intervals.median()
        max_gap = intervals.max() if not intervals.empty else 0
        required_missing = group[["lon", "lat", "timestamp", "speed_kmh"]].isna().any().any()
        valid = {
            "trip_candidate_id": trip_id,
            "source_user_id": group["source_user_id"].iloc[0],
            "point_count": int(len(group)),
            "duration_seconds": float(duration.total_seconds()),
            "median_interval_seconds": float(median_interval) if pd.notna(median_interval) else None,
            "max_gap_seconds": float(max_gap) if pd.notna(max_gap) else None,
            "has_missing_required": bool(required_missing),
            "lon_lat_valid": bool(group["lon"].between(-180, 180).all() and group["lat"].between(-90, 90).all()),
            "speed_non_negative": bool(group["speed_kmh"].ge(0).all()),
            "timestamps_sorted": bool(group["timestamp"].is_monotonic_increasing),
        }
        for column in [
            "leg_id",
            "storyline_id",
            "trip_id",
            "mode",
            "detected_mode",
            "effective_mode",
            "mode_niv1",
            "mode_niv2",
            "mode_mrmt",
            "leg_started_at",
            "leg_finished_at",
        ]:
            if column in group.columns:
                valid[column] = group[column].iloc[0]
        valid["rows_ok"] = cfg["min_rows"] <= valid["point_count"] <= cfg["max_rows"]
        valid["duration_ok"] = valid["duration_seconds"] <= cfg["max_duration_days"] * 24 * 3600
        valid["median_interval_ok"] = (
            valid["median_interval_seconds"] is not None
            and valid["median_interval_seconds"] <= cfg["max_median_interval_seconds"]
        )
        valid["gap_ok"] = valid["max_gap_seconds"] is not None and valid["max_gap_seconds"] <= cfg["max_timestamp_gap_seconds"]
        valid["relut_ready"] = all(
            [
                valid["rows_ok"],
                valid["duration_ok"],
                valid["median_interval_ok"],
                valid["gap_ok"],
                not valid["has_missing_required"],
                valid["lon_lat_valid"],
                valid["speed_non_negative"],
                valid["timestamps_sorted"],
            ]
        )
        rows.append(valid)

    validation = pd.DataFrame(rows)
    summary = {
        "segments_total": int(len(validation)),
        "segments_relut_ready": int(validation["relut_ready"].sum()) if len(validation) else 0,
        "segments_not_ready": int((~validation["relut_ready"]).sum()) if len(validation) else 0,
        "points_total": int(len(relut_points)),
        "median_interval_overall_seconds": float(relut_points.groupby("trip_candidate_id")["timestamp"].diff().dt.total_seconds().median()),
    }
    return validation, summary
