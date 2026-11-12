# Collection de tables par leg (`outputs/<jeu>/legs/`)

Objectif : rendre interopérables les résultats des différentes méthodes (temporelle ReLUT, spatiale, analyses sociologiques) en les rangeant dans des tables **au niveau de la leg**, toutes avec la clé `leg_id`, assemblées par une vue.

## Tables

| Table | Emplacement | Producteur | Contenu |
|---|---|---|---|
| `legs_enriched` | `data/legs_enriched.parquet` | étape 01 | toutes les legs : trip, motif, moment, questionnaire, marche après la voiture |
| `corpus_by_leg` | `legs/corpus_by_leg.parquet` | étape 01 (`leg_corpus.py`) | toutes les legs : `corpus_status` (`in_corpus` = base commune des méthodes), motifs d'exclusion, statistiques |
| `relut_by_leg` | `legs/relut_by_leg.parquet` | étape 32 | toutes les legs : statut ReLUT et résultats du dernier segment |
| `spatial_by_leg` | `legs/spatial_by_leg.parquet` | étape 22 (`spatial_waypoints.py`) | toutes les legs : statut, motif d'exclusion, détection, distance et durée, métriques de fenêtre |
| `routing_by_leg` | `legs/routing_by_leg.parquet` | notebook de routage (à venir) | lit `data/routing_requests.parquet` écrite par l'étape 22 |
| `persons` | `legs/persons.parquet` | étape 32 (dérivée de `legs_enriched`) | une ligne par personne : questionnaire, `Weight`, nombre de legs |

## Conventions

- **Une table par producteur**, un seul écrivain ; une ligne par leg de `legs_enriched` ; `leg_id` unique (vérifié à l'écriture).
- **Préfixe par source** (`relut_`, `spatial_`…) ; **unité dans le nom** : `_s`, `_m`, `_min`, `_at` (instant UTC avec fuseau).
- **Statut** `<préfixe>_status` dans chaque table ; **0 = évalué sans effet, NaN = non évalué ou non applicable**.
- **Provenance** : `legs/_manifest.json` (producteur, date, `run_id`, paramètres, nombre de lignes) ; mêmes informations dans les métadonnées Parquet (clé `leg_collection`).
- **Dictionnaire généré** : `legs/_dictionary.csv` (type, part de valeurs manquantes, description ; `documentee = False` signale une colonne à décrire dans `leg_collection.COLUMN_DESCRIPTIONS`).
- **Runs limités** (`MAX_LEGS`) dans `legs/_samples/<run_id>/` ; la table complète n'est jamais écrasée par un échantillon.
- **Pas de géométrie** : les traces restent au niveau des waypoints (`waypoints_caronly`, `waypoints_post_car_walk`).
- **Pondération** : `Weight` est un poids de personne ; pour des résultats à l'échelle de la population, passer par `persons`.

## `relut_by_leg`

Règle : pour chaque leg voiture, **le dernier segment** (fin la plus tardive parmi tous les candidats, retenus ou non), **sans repli** sur un segment antérieur.

| `relut_status` | Signification |
|---|---|
| `evaluated` | dernier segment prédit |
| `not_car_leg` | autre mode, non applicable |
| `no_car_waypoints` | leg voiture sans waypoint voiture apparié |
| `outside_study_area` | destination hors du périmètre |
| `no_segment` | aucun segment construit |
| `last_segment_rejected` | dernier segment écarté par la segmentation |
| `excluded_by_corpus` | leg écartée par les filtres du corpus commun (voir `corpus_by_leg`) |
| `last_segment_loop` | dernier segment retiré comme boucle à l'étape 01 (`apply_loop_filter`) |
| `not_sampled` | run limité, leg non tirée |
| `unknown` | non déterminable (table d'appariement absente) |

Mesures (NaN si non évalué, 0 si évalué sans recherche) : `relut_search_duration_s`, `relut_search_distance_m` (le long de la trace depuis la bascule = longueur du segment − distance parcourue avant la bascule), `relut_straight_line_distance_m` (vol d'oiseau bascule → stationnement), `relut_search_started_at`, `relut_search_detected`. Contexte : `relut_segment_*`, `relut_max_probability`, `relut_threshold`, `relut_run_id`.

## Vue d'assemblage

```python
from parking_search_prediction.leg_collection import load_leg_table
legs = load_leg_table(config, ["relut_by_leg"])                       # + "spatial_by_leg" plus tard
legs = load_leg_table(config, ["relut_by_leg"], sample_run_id="…")    # tables d'un run limité
legs.attrs["leg_collection"]                                          # provenance
```

Jointures gauches sur `leg_id` depuis `legs_enriched` ; une colonne présente dans deux tables arrête l'assemblage. Pour Déclic (sans `legs_enriched`), la première table sert de base.

## Contrat d'une méthode (lu par l'étape 41)

Toute table de méthode (`relut_by_leg`, `spatial_by_leg`, `hybrid_by_leg`) doit fournir, avec son préfixe `<p>` :

| Colonne | Contenu |
|---|---|
| `<p>_status` | `evaluated` pour les legs évaluées |
| `<p>_search_detected` | cruising détecté (booléen, NaN si non évalué) |
| `<p>_search_duration_s` | durée du cruising, s (0 sans cruising, NaN si non évalué) |
| `<p>_search_distance_m` | distance du cruising, m (0 sans cruising, NaN si non évalué) |

L'étape 41 produit aussi la **répartition** des legs avec cruising par heure et par motif (somme = 100 %), comparée à la répartition de toutes les legs évaluées. `indicators.min_cruising_duration_s` permet de ne compter que les cruisings d'au moins N secondes.

Une nouvelle méthode s'ajoute dans `leg_collection.TABLES` et `indicators.METHODS` ; ses couleurs sont fixes (rose = temporel, vert = spatial, jaune = hybride, bleu = autre).

## Indicateurs H3 (étape 41, `h3_indicators.py`)

Résolution 8 par défaut. Méthode temporelle : chaque point prédit porte le temps et la distance jusqu'au point suivant de son segment (plafonnés à `max_point_interval_s`), attribués à l'hexagone du point. Les sommes sur les points en recherche redonnent exactement les durées et distances de la synthèse par segment (vérifié).

| Indicateur | Population |
|---|---|
| temps et distance de cruising cumulés, legs avec cruising, part du temps voiture en cruising | segments prédits du dernier run de l'étape 32 |
| fins de legs, fins / legs qui traversent | legs voiture du périmètre (waypoints de l'étape 01) |

Une valeur n'est publiée que si `min_persons` personnes **et** `min_legs` legs y contribuent (3 et 3 par défaut). Sorties : `reports/h3/41_h3_r8_indicateurs.csv`, `maps/41_h3_r8_indicateurs.html` (une couche par indicateur à cocher, légende par couche, offre de stationnement OTC en option).
