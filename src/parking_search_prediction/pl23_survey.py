"""Variables de stationnement déclarées dans le questionnaire du Panel Lémanique 2023.

Reprend **à l'identique** la dérivation de ``pl23_spatial._load_pl23_survey_subset``
(étape 23), pour que l'EDA et la détection spatiale utilisent les mêmes
catégories :

- ``space_home`` (domicile) : ``Aucun`` par défaut ; ``Q62_1_R == 1`` → place
  dédiée ; ``Q63 == 1`` → pas de place, pas besoin de cruise ; ``Q63 == 2`` →
  pas de place, besoin de cruise (Q63 l'emporte sur Q62_1_R) ;
- ``space_work`` (travail / études) : bloc ``Q71_R`` / ``Q72`` (1, 2 → pas
  besoin ; 3 → besoin), puis repli ``Q65_R`` / ``Q67`` (1, 3 → pas besoin ;
  2 → besoin) si toujours vide ; sinon NA.

Les lignes ``Weight == "#NULL!"`` sont exclues. La clé est ``IDNO`` (= identifiant
panel ``user_id_fors``).
"""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

from .config import resolve_path

PLACE_DEDIEE = "Place dédiée"
PAS_BESOIN = "Pas de place dédiée, pas besoin de cruise"
BESOIN = "Pas de place dédiée, besoin de cruise"
PARKING_CATEGORIES = [PLACE_DEDIEE, PAS_BESOIN, BESOIN]

SURVEY_COLUMNS = [
    "IDNO",
    "Weight",
    "Q62_1_R",
    "Q63",
    "Q71_R",
    "Q72",
    "Q65_R",
    "Q67",
    "AGGLO_CH_dom",
    "pays_dom",
    "AGGLO_CH_trav",
    "pays_trav",
    "Genre_actuel",
    "age",
    "KLASSE_ARE_dom",
]
OUTPUT_COLUMNS = [
    "user_id_fors",
    "space_work",
    "space_home",
    "zone_travail",
    "zone_domicile",
    "Genre_actuel",
    "age",
    "KLASSE_ARE_dom",
    "Weight",
]


def survey_path(config: dict[str, Any]):
    """Chemin du questionnaire : ``pl23_survey.survey_file``, sinon ``pl23_spatial.survey_file``."""
    for section in ("pl23_survey", "pl23_spatial"):
        survey_file = (config.get(section) or {}).get("survey_file")
        if survey_file:
            return resolve_path(config, survey_file)
    raise KeyError("Renseigner pl23_survey.survey_file dans la config (ex. Data/GPS/pl23/EPFL_vague1_v4.csv).")


def derive_parking_variables(survey: pd.DataFrame) -> pd.DataFrame:
    """Ajoute ``space_home`` / ``space_work`` (et zones) à un questionnaire lu en texte."""
    survey = survey.copy()
    survey["user_id_fors"] = survey["IDNO"].astype("string")

    survey["space_home"] = "Aucun"
    if "Q62_1_R" in survey:
        survey.loc[survey["Q62_1_R"].eq("1"), "space_home"] = PLACE_DEDIEE
    if "Q63" in survey:
        survey.loc[survey["Q63"].eq("1"), "space_home"] = PAS_BESOIN
        survey.loc[survey["Q63"].eq("2"), "space_home"] = BESOIN

    survey["space_work"] = pd.NA
    if "Q71_R" in survey:
        survey.loc[survey["Q71_R"].eq("1"), "space_work"] = PLACE_DEDIEE
    if "Q72" in survey:
        survey.loc[survey["Q72"].isin(["1", "2"]), "space_work"] = PAS_BESOIN
        survey.loc[survey["Q72"].eq("3"), "space_work"] = BESOIN
    missing_work = survey["space_work"].isna()
    if "Q65_R" in survey:
        survey.loc[missing_work & survey["Q65_R"].eq("1"), "space_work"] = PLACE_DEDIEE
    missing_work = survey["space_work"].isna()
    if "Q67" in survey:
        survey.loc[missing_work & survey["Q67"].isin(["1", "3"]), "space_work"] = PAS_BESOIN
        survey.loc[missing_work & survey["Q67"].eq("2"), "space_work"] = BESOIN

    for prefix, target in (("dom", "zone_domicile"), ("trav", "zone_travail")):
        agglo, pays = f"AGGLO_CH_{prefix}", f"pays_{prefix}"
        if agglo in survey or pays in survey:
            empty = pd.Series(pd.NA, index=survey.index)
            survey[target] = survey.get(agglo, empty).fillna(survey.get(pays, empty))
            survey[target] = survey[target].replace({"CH": "CH_autres", "FR": "FR_autres"})
    return survey


def load_pl23_survey_parking(
    config: dict[str, Any], user_ids: Iterable[str] | None = None
) -> pd.DataFrame:
    """Une ligne par participant (``user_id_fors``) avec ``space_home`` / ``space_work``."""
    path = survey_path(config)
    if not path.exists():
        raise FileNotFoundError(f"Questionnaire PL23 introuvable : {path}")
    header = list(pd.read_csv(path, nrows=0).columns)
    if "IDNO" not in header:
        raise KeyError(f"Colonne IDNO absente de {path.name} ; colonnes : {header[:30]}…")
    available = [column for column in SURVEY_COLUMNS if column in header]

    survey = pd.read_csv(path, usecols=available, dtype="string", low_memory=False)
    if "Weight" in survey:
        survey = survey[survey["Weight"].ne("#NULL!")].copy()
    if user_ids is not None:
        survey = survey[survey["IDNO"].astype("string").isin(set(user_ids))].copy()
    survey = derive_parking_variables(survey)
    columns = [column for column in OUTPUT_COLUMNS if column in survey.columns]
    return survey[columns].drop_duplicates("user_id_fors").reset_index(drop=True)


def parking_by_user(config: dict[str, Any], source_user_ids: Iterable[str]) -> pd.DataFrame:
    """``space_home`` / ``space_work`` pour des participants identifiés par leur UUID Motiontag.

    Jointure : ``source_user_id`` (UUID, sorties de l'étape 01) → table de
    correspondance ``leg_matching.user_id_map`` → ``user_id_fors`` → questionnaire
    (``IDNO``). Une ligne par UUID fourni ; ``in_survey`` indique si le
    participant a été retrouvé dans le questionnaire.
    """
    from .data_prep import normalize_user_ids

    users = pd.DataFrame(
        {"source_user_id": normalize_user_ids(pd.Series(sorted({u for u in source_user_ids if u}), dtype=object))}
    )
    map_cfg = config["leg_matching"].get("user_id_map") or {}
    if not map_cfg.get("file"):
        raise KeyError("leg_matching.user_id_map.file est nécessaire pour relier les participants au questionnaire.")
    legs_col = map_cfg.get("legs_id_col", "user_id_fors")
    waypoints_col = map_cfg.get("waypoints_id_col", "user_id_motiontag")
    mapping = pd.read_csv(resolve_path(config, map_cfg["file"]), usecols=[legs_col, waypoints_col], dtype="string")
    mapping = pd.DataFrame(
        {
            "source_user_id": normalize_user_ids(mapping[waypoints_col]),
            "user_id_fors": normalize_user_ids(mapping[legs_col]),
        }
    ).dropna().drop_duplicates("source_user_id")

    table = users.merge(mapping, on="source_user_id", how="left")
    survey = load_pl23_survey_parking(config, table["user_id_fors"].dropna())
    table = table.merge(survey, on="user_id_fors", how="left")
    table["in_survey"] = table["user_id_fors"].isin(survey["user_id_fors"])
    return table
