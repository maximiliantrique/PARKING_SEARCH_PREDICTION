"""Collection de tables au niveau de la leg (clé ``leg_id``) et vue d'assemblage.

Conventions
-----------
- **Une table par producteur**, une ligne par leg (toutes les legs de
  ``legs_enriched``), clé ``leg_id`` unique — vérifiée à l'écriture.
- **Préfixe par source** (``relut_``, ``spatial_``…) et **unité dans le nom**
  (``_s``, ``_m``, ``_at`` = instant UTC).
- **Statut par table** (``<préfixe>_status``). Règle commune : **0 = évalué
  sans effet, NaN = non évalué** (ou non applicable).
- **Traçabilité** : ``legs/_manifest.json`` (producteur, date, run, paramètres,
  nombre de lignes, empreinte de config) ; les mêmes informations sont écrites
  dans les métadonnées du Parquet (clé ``leg_collection``).
- **Dictionnaire généré** : ``legs/_dictionary.csv`` (type, part de valeurs
  manquantes, description tirée de ``COLUMN_DESCRIPTIONS`` ; une colonne sans
  description est signalée).
- **Runs limités** dans ``legs/_samples/<run_id>/``, jamais à la place de la
  table complète.
- **Table des personnes** ``legs/persons.parquet`` pour les pondérations à
  l'échelle de la personne (``Weight`` est un poids de personne).
- **Pas de géométrie** dans la collection.

Emplacements : ``legs_enriched`` est produite par l'étape 01 dans
``data/legs_enriched.parquet`` ; les autres tables sont dans ``legs/``.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .config import output_path
from .utils import write_json

KEY = "leg_id"
COLLECTION_DIR = "legs"
METADATA_KEY = b"leg_collection"

# Nom de table -> chemin relatif au dossier de sortie, producteur
TABLES = {
    "legs_enriched": ("data/legs_enriched.parquet", "étape 01 (leg_enrichment.py)"),
    "corpus_by_leg": ("legs/corpus_by_leg.parquet", "étape 01 (leg_corpus.py) : corpus commun des méthodes"),
    "relut_by_leg": ("legs/relut_by_leg.parquet", "étape 32 (relut_detection.py)"),
    "spatial_by_leg": ("legs/spatial_by_leg.parquet", "étape 22 (spatial_waypoints.py)"),
    "routing_by_leg": ("legs/routing_by_leg.parquet", "étape 27 (routing_xyt.py)"),
    "hybrid_by_leg": ("legs/hybrid_by_leg.parquet", "méthode hybride (à venir)"),
}

RELUT_STATUSES = {
    "evaluated": "dernier segment de la leg prédit par ReLUT",
    "not_car_leg": "leg d'un autre mode : non applicable",
    "no_car_waypoints": "leg voiture sans waypoint voiture apparié",
    "outside_study_area": "destination (dernier waypoint voiture) hors du périmètre d'étude",
    "no_segment": "aucun segment candidat construit pour la leg",
    "last_segment_rejected": "dernier segment écarté par la segmentation (points, durée, distance)",
    "excluded_by_corpus": "leg voiture du périmètre écartée par les filtres du corpus commun (voir corpus_exclusion_reason)",
    "last_segment_loop": "dernier segment retiré comme boucle à l'étape 01 (segmentation.apply_loop_filter)",
    "not_sampled": "run limité : leg non tirée",
    "unknown": "leg voiture non évaluée, raison non déterminable (table d'appariement absente)",
}

COLUMN_DESCRIPTIONS: dict[str, str] = {
    "leg_id": "Identifiant de leg (clé de la collection).",
    # relut_by_leg
    "relut_status": "Statut ReLUT de la leg : " + " ; ".join(f"{k} = {v}" for k, v in RELUT_STATUSES.items()),
    "relut_search_detected": "True si une recherche est prédite sur le dernier segment ; NaN si non évalué.",
    "relut_search_duration_s": "Durée de recherche prédite (bascule → fin du segment), s ; 0 sans recherche ; NaN si non évalué.",
    "relut_search_distance_m": "Distance parcourue le long de la trace depuis la bascule (haversine point à point), m ; 0 sans recherche ; NaN si non évalué.",
    "relut_straight_line_distance_m": "Distance à vol d'oiseau entre le point de bascule et la fin du segment (stationnement supposé), m ; 0 sans recherche ; NaN si non évalué.",
    "relut_search_started_at": "Instant de bascule en recherche (UTC) ; vide sans recherche ou si non évalué.",
    "relut_segment_id": "Segment retenu (dernier segment de la leg, sans repli sur un segment antérieur).",
    "relut_segments_in_leg": "Nombre de segments candidats construits pour la leg.",
    "relut_segment_kept": "True si le dernier segment a passé les filtres de segmentation de l'étape 01.",
    "relut_segment_duration_s": "Durée du segment retenu, s.",
    "relut_segment_distance_m": "Longueur du segment retenu le long de la trace, m.",
    "relut_segment_loop_distance_m": "Distance début-fin du dernier segment (critère des boucles), m.",
    "relut_segment_relut_ready": "True si le segment passe la validation de format ReLUT.",
    "relut_max_probability": "Probabilité de recherche maximale sur le segment.",
    "relut_threshold": "Seuil de probabilité appliqué au segment (auto : selon l'échantillonnage médian).",
    "relut_run_id": "Identifiant du run de l'étape 32 qui a produit la ligne.",
    # spatial_by_leg (descriptions complètes : spatial_waypoints.VARIABLES)
    "spatial_status": "Statut spatial : evaluated, excluded_by_corpus (voir corpus_exclusion_reason), excluded_by_spatial_rule (voir spatial_exclusion_reason), no_destination_window, not_car_leg, no_car_waypoints, outside_study_area.",
    "spatial_run_id": "Identifiant du run de l'étape 22 qui a produit la ligne.",
    # routing_by_leg
    "routing_status": "routed, failed (erreur ou pas de route), not_yet_routed (demande pas encore traitée), not_requested (aucune demande pour la leg).",
    "routing_observed_distance_m": "Distance observée entre waypoints, point d'entrée → destination, m (copiée de la demande).",
    "routing_routed_distance_m": "Plus court chemin sur le réseau entre les mêmes points, m.",
    "routing_dist_excess_m": "Détour = observée − routée, m (NaN si non routée).",
    "routing_routed_duration_s": "Durée de l'itinéraire routé selon le profil, s (indicative avec la pondération distance).",
    "routing_route_type": "Paire routée (buffer_entry_to_leg_end).",
    "routing_route_id": "Identifiant déterministe de la route (demande + mode + pondération), clé du cache.",
    "routing_graph": "Graphe routier utilisé (network_id du service, sinon le mode demandé).",
    "routing_weighting": "Pondération du routage (distance = plus court chemin).",
    "routing_error": "Message d'erreur du service pour les routes en échec.",
    "routing_run_id": "Identifiant du run de l'étape 27.",
}
# OD corrigée (positions recalées par le map matching) : mêmes colonnes, préfixe routing_matched_od_
COLUMN_DESCRIPTIONS.update({
    "routing_matched_od_" + key[len("routing_"):]: description + " — OD corrigée : 1re → dernière position recalée par le map matching ; distance observée = longueur appariée."
    for key, description in list(COLUMN_DESCRIPTIONS.items())
    if key.startswith("routing_") and key != "routing_run_id"
})


def _add_spatial_descriptions() -> None:
    from .leg_corpus import DESCRIPTIONS as CORPUS_DESCRIPTIONS

    for variable, description in CORPUS_DESCRIPTIONS.items():
        COLUMN_DESCRIPTIONS.setdefault(variable, description)
    try:
        from .spatial_waypoints import VARIABLES
    except ImportError:  # pragma: no cover
        return
    for _, variable, description in VARIABLES:
        if variable.startswith("spatial_"):
            COLUMN_DESCRIPTIONS.setdefault(variable, description)


def collection_dir(config: dict[str, Any]) -> Path:
    return output_path(config, COLLECTION_DIR)


def table_path(config: dict[str, Any], name: str, run_id: str | None = None) -> Path:
    relative, _ = TABLES[name]
    if run_id:
        return collection_dir(config) / "_samples" / run_id / Path(relative).name
    return output_path(config, *Path(relative).parts)


def config_fingerprint(config: dict[str, Any], sections: Iterable[str]) -> str:
    payload = {section: config.get(section) for section in sections}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def _manifest_path(config: dict[str, Any], run_id: str | None) -> Path:
    base = collection_dir(config) / "_samples" / run_id if run_id else collection_dir(config)
    return base / "_manifest.json"


def read_manifest(config: dict[str, Any], run_id: str | None = None) -> dict[str, Any]:
    path = _manifest_path(config, run_id)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_leg_table(
    frame: pd.DataFrame,
    config: dict[str, Any],
    name: str,
    *,
    params: dict[str, Any],
    run_id: str | None = None,
    sample: bool = False,
) -> Path:
    """Écrit une table de la collection, met à jour le manifeste et le dictionnaire."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if KEY not in frame.columns:
        raise KeyError(f"{name} : colonne {KEY!r} absente.")
    duplicated = frame[KEY].duplicated()
    if duplicated.any():
        raise ValueError(f"{name} : {int(duplicated.sum())} leg_id en double, la clé doit être unique.")

    path = table_path(config, name, run_id if sample else None)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "table": name,
        "path": str(path),
        "producer": TABLES[name][1],
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": run_id,
        "sample": sample,
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "params": params,
    }
    table = pa.Table.from_pandas(frame, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata[METADATA_KEY] = json.dumps(entry, default=str).encode("utf-8")
    pq.write_table(table.replace_schema_metadata(metadata), path)

    manifest_path = _manifest_path(config, run_id if sample else None)
    manifest = read_manifest(config, run_id if sample else None)
    manifest[name] = entry
    write_json(manifest_path, manifest)
    if not sample:
        write_dictionary(config)
    return path


# --------------------------------------------------------------------------- #
# Table des personnes et dictionnaire
# --------------------------------------------------------------------------- #

PERSON_COLUMNS = ["user_id_fors", "source_user_id", "space_home", "space_work", "zone_domicile",
                  "zone_travail", "KLASSE_ARE_dom", "Genre_actuel", "age", "Weight"]


def write_persons_table(config: dict[str, Any]) -> Path | None:
    """Une ligne par personne (``user_id_fors``), dérivée de ``legs_enriched``."""
    legs_path = table_path(config, "legs_enriched")
    if not legs_path.exists():
        return None
    import pyarrow.parquet as pq

    available = set(pq.read_schema(legs_path).names)
    if "user_id_fors" not in available:
        return None
    columns = [c for c in PERSON_COLUMNS if c in available]
    legs = pd.read_parquet(legs_path, columns=[*columns, "effective_mode"])
    persons = legs.drop(columns="effective_mode").drop_duplicates("user_id_fors").dropna(subset=["user_id_fors"])
    counts = legs.groupby("user_id_fors").agg(n_legs=("effective_mode", "size"))
    persons = persons.merge(counts, on="user_id_fors", how="left")
    path = collection_dir(config) / "persons.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    persons.to_parquet(path, index=False)
    return path


def write_dictionary(config: dict[str, Any]) -> Path:
    """Dictionnaire généré à partir des tables présentes (type, valeurs manquantes, description)."""
    import pyarrow.parquet as pq

    _add_spatial_descriptions()

    rows = []
    for name in TABLES:
        path = table_path(config, name)
        if not path.exists():
            continue
        schema = pq.read_schema(path)
        frame = pd.read_parquet(path)
        for field in schema:
            if field.name.startswith("__index"):
                continue
            rows.append({
                "table": name,
                "variable": field.name,
                "type": str(field.type),
                "part_manquante": float(frame[field.name].isna().mean()) if len(frame) else None,
                "description": COLUMN_DESCRIPTIONS.get(
                    field.name, "voir docs/01_prepare_inputs.md (enrichissement des legs)" if name == "legs_enriched" else ""
                ),
                "documentee": field.name in COLUMN_DESCRIPTIONS,
            })
    dictionary = pd.DataFrame(rows)
    path = collection_dir(config) / "_dictionary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    dictionary.to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------- #
# Vue d'assemblage
# --------------------------------------------------------------------------- #


def load_leg_table(
    config: dict[str, Any],
    tables: Iterable[str] = ("relut_by_leg",),
    *,
    columns: dict[str, list[str]] | None = None,
    sample_run_id: str | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """``legs_enriched`` + jointures gauches sur ``leg_id`` des tables demandées.

    ``columns`` restreint les colonnes lues par table (``leg_id`` toujours lu).
    ``sample_run_id`` lit les tables d'un run limité (``legs/_samples/<run_id>``).
    Une colonne présente dans deux tables arrête l'assemblage (collision).
    La provenance (manifeste) est rangée dans ``result.attrs["leg_collection"]``.
    """
    columns = columns or {}
    tables = list(tables)
    manifest = read_manifest(config, sample_run_id)
    base_path = table_path(config, "legs_enriched")
    if base_path.exists():
        base_columns = columns.get("legs_enriched")
        result = pd.read_parquet(base_path, columns=[KEY, *[c for c in base_columns if c != KEY]] if base_columns else None)
        provenance = {"legs_enriched": {"path": str(base_path), "rows": int(len(result))}}
    else:  # declic : pas de legs_enriched, la première table sert de base
        if not tables:
            raise FileNotFoundError(f"{base_path} introuvable et aucune table demandée.")
        result = pd.DataFrame({KEY: pd.read_parquet(table_path(config, tables[0], sample_run_id), columns=[KEY])[KEY]})
        provenance = {}
        if verbose:
            print(f"[legs] {base_path.name} absent : base = {tables[0]}")

    for name in tables:
        path = table_path(config, name, sample_run_id)
        if not path.exists():
            raise FileNotFoundError(f"Table {name!r} introuvable : {path}")
        wanted = columns.get(name)
        frame = pd.read_parquet(path, columns=[KEY, *[c for c in wanted if c != KEY]] if wanted else None)
        collisions = sorted((set(frame.columns) & set(result.columns)) - {KEY})
        if collisions:
            raise ValueError(f"Colonnes en double entre la vue et {name!r} : {collisions}")
        result = result.merge(frame, on=KEY, how="left", validate="one_to_one")
        provenance[name] = manifest.get(name, {"path": str(path)})
        if verbose:
            entry = provenance[name]
            print(f"[legs] {name} : {entry.get('rows', len(frame))} lignes — run {entry.get('run_id')} — {entry.get('created_at')}")
    result.attrs["leg_collection"] = provenance
    return result
