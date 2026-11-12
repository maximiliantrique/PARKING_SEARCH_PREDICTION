from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def resolve_config_path(config_path: str | Path) -> Path:
    """Resolve a config file path robustly from common project working dirs.

    The documented usage is to run commands from the repository root. In
    practice, notebooks and ad-hoc terminal sessions are often started from
    `Notebooks/` or `scripts/`. For relative config paths, this resolver first
    tries the current working directory, then the repository root inferred from
    this package location, then parent directories of the current working
    directory.
    """

    path = Path(config_path)
    if path.is_absolute():
        return path.resolve()

    package_root = Path(__file__).resolve().parents[2]
    candidates = [Path.cwd().resolve() / path, package_root / path]
    candidates.extend(parent / path for parent in Path.cwd().resolve().parents)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (Path.cwd().resolve() / path).resolve()


def project_root_from_config(config_path: str | Path) -> Path:
    config_path = resolve_config_path(config_path)
    # config/default.yaml -> project root
    if config_path.parent.name == "config":
        return config_path.parents[1]
    for parent in config_path.parents:
        if (parent / "src" / "parking_search_prediction").exists():
            return parent
    raise ValueError(f"Cannot infer project root from config path: {config_path}")


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = resolve_config_path(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(project_root_from_config(config_path))
    return config


def resolve_path(config: dict[str, Any], path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return Path(config["_project_root"]).resolve() / path


def output_path(config: dict[str, Any], *parts: str) -> Path:
    return resolve_path(config, config["paths"]["output_dir"]).joinpath(*parts)


def leg_filter(config: dict[str, Any], key: str, fallback: Any = None) -> float | None:
    """Critère commun aux méthodes temporelle et spatiale (section ``leg_filters`` du YAML).

    Si la clé n'y est pas, on garde la valeur propre à la méthode (``fallback``).
    """
    value = (config.get("leg_filters") or {}).get(key, fallback)
    return None if value is None else float(value)


def ensure_output_dirs(config: dict[str, Any]) -> None:
    for subdir in ["data", "figures", "maps", "reports"]:
        output_path(config, subdir).mkdir(parents=True, exist_ok=True)
