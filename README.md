# parking_search_prediction

Détection de la recherche de stationnement à partir de traces GPS — source de travail : Panel Lémanique 2023 (Déclic Mobilité disponible en rétrocompatibilité).

- Installation : `python -m pip install -r requirements.txt`
- Étape 01, préparation des entrées : `Notebooks/01_prepare_inputs.ipynb` — documentation dans [`docs/01_prepare_inputs.md`](docs/01_prepare_inputs.md)
- Étape 21, EDA Panel Lémanique (distributions des waypoints, stationnement déclaré) : `Notebooks/21_EDA_panel_lemanique.ipynb` ou `python scripts/21_eda_panel_lemanique.py`
- Collection de tables par leg (`legs/`, clé `leg_id`) : [`docs/legs_collection.md`](docs/legs_collection.md)
- Étape 32, détection temporelle ReLUT (durées prédites, groupes besoin / pas besoin de chercher, cartes) : `Notebooks/32_temporal_detection_relut.ipynb` ou `python scripts/32_temporal_detection.py --max-legs 300`. Modèle fourni dans `models/` (pas besoin de TensorFlow).
- Étape 41, indicateurs de cruising par heure et par motif, pour une ou plusieurs méthodes : `Notebooks/41_indicateurs_cruising.ipynb` ou `python scripts/41_indicators.py`
- Étape 22, détection spatiale sur les waypoints (même population que le temporel, demandes de routage, `legs/spatial_by_leg.parquet`) : `Notebooks/22_spatial_detection_waypoints.ipynb`
- Étape 27, routage des demandes de l'étape 22 avec le service local `xyt_routing` (Docker + OSRM) → `legs/routing_by_leg.parquet` : `Notebooks/27_routing.ipynb`, puis relancer le notebook 22 pour la distance excessive

**Guide des notebooks (ordre, paramètres, sorties, corpus commun, dépannage) : [`docs/notebooks.md`](docs/notebooks.md).** Les filtres de legs communs aux deux méthodes sont dans la section `leg_filters` de `config/full_waypoints.yaml`.
