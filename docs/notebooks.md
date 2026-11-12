# Guide des notebooks

Projet `parking_search_prediction` — détection de la recherche de stationnement (cruising) sur les traces GPS du Panel Lémanique 2023 (Déclic en rétrocompatibilité).

## Ordre d'exécution

```
01_prepare_inputs ──► 21_EDA_panel_lemanique ──► (ajuster leg_filters, relancer 01 si besoin)
        │
        ├──► 32_temporal_detection_relut ─────────────────────────────┐
        │                                                             ├──► 41_indicateurs_cruising
        └──► 22_spatial_detection_waypoints ──► 27_routing ──► 22 ────┘
```

| Notebook | Rôle | Durée indicative (jeu complet) |
|---|---|---|
| `01_prepare_inputs` | conversion, nettoyage, appariement waypoints ↔ legs, enrichissement, **corpus commun** | conversion ~1 h (une fois), préparation ~30 min |
| `21_EDA_panel_lemanique` | distributions des waypoints, stationnement déclaré, **effet de chaque filtre du corpus** | quelques minutes |
| `32_temporal_detection_relut` | méthode temporelle (modèle ReLUT) | ~10–30 min |
| `22_spatial_detection_waypoints` | méthode spatiale sur waypoints, demandes de routage, map matching | quelques minutes (+ appariement) |
| `27_routing` | plus court chemin (service `xyt_routing`) → distance excessive | selon le nombre de routes (cache) |
| `41_indicateurs_cruising` | indicateurs par heure, par motif, H3, histogrammes filtrés | quelques minutes |

Chaque notebook commence par une **cellule de paramètres** ; tout le reste se lance sans modification. Les paramètres durables sont dans `config/full_waypoints.yaml`, les notebooks ne surchargent que les choix d'exécution (dossier de sortie, échantillon, affichage).

## Le corpus commun de legs (point central)

Les deux méthodes traitent **exactement les mêmes legs** : celles du corpus commun, construit à l'étape 01 et écrit dans `legs/corpus_by_leg.parquet`. Ses filtres sont **uniquement** dans la section `leg_filters` du YAML :

| Paramètre | Effet | Défaut |
|---|---|---|
| `max_accuracy_m` | waypoints plus imprécis retirés au nettoyage (avant appariement) | 100 m |
| `min_leg_points` | waypoints voiture minimum par leg | 20 |
| `min_leg_duration_s` | durée minimale (premier → dernier waypoint) | 300 s |
| `min_leg_length_m` | distance parcourue minimale (somme des pas entre waypoints) | 500 m |
| `min_origin_destination_distance_m` | début et fin trop proches → écarté (boucle, aller-retour) | 100 m |
| `max_window_gap_s` | trou maximal entre waypoints dans le cercle de destination | 120 s |
| `max_trip_duration_s` | durée maximale du trip | 18 000 s |
| `destination_window_radius_m` | rayon du cercle de destination (aussi celui de la méthode spatiale) | 1 000 m |
| `exclude_tunnel_h3_destinations`, `tunnel_h3_cells` | destination dans une cellule H3 de tunnel | activé, 11 cellules |

`null` désactive un filtre. Statuts (`corpus_status`) : `in_corpus` (traitée par les deux méthodes), `excluded` (motif dans `corpus_exclusion_reason`), `not_car_leg`, `no_car_waypoints`, `outside_study_area`.

**Procédure pour régler les filtres** : lancer 01 une fois avec des valeurs larges → regarder le panneau de sensibilité du notebook 21 → fixer les valeurs dans le YAML → relancer 01 avec `FORCE_RERUN = True` (la conversion Parquet n'est pas refaite) → lancer 32 et 22.

> Un seuil de précision **plus strict** se teste dans le notebook 21 sans relancer 01 ; un seuil plus large exige de relancer 01 (les points retirés ne sont plus dans les sorties).

---

## 01 — Préparation des entrées

Détail complet : [`01_prepare_inputs.md`](01_prepare_inputs.md).

**Paramètres principaux** : `DATA_SOURCE` (`pl23` / `declic`), `RUN_FULL`, `RUN_CONVERSION` (une seule fois par jeu de données), `RUN_EQUIVALENCE_TEST`, `FORCE_RERUN` (après une modification du code ou du YAML), `DATASETS[...]` (chemins), enrichissement (`POST_CAR_WALK_*`, `LOCAL_TIMEZONE`), périmètre (`APPLY_STUDY_AREA_FILTER`). Laisser `APPLY_LOOP_FILTER = False` : les boucles sont filtrées au niveau des legs par le corpus.

**Étapes** : conversion CSV → Parquet partitionné par utilisateur ; nettoyage (dont `leg_filters.max_accuracy_m`) ; correspondance `user_id_fors` → UUID ; appariement waypoints ↔ legs (intervalle demi-ouvert, `overlapping_tracks`) ; enrichissement (trip, motif, moment, questionnaire, marche après la voiture) ; filtre voiture et périmètre ; segmentation pour ReLUT ; **corpus commun**.

**Sorties** : `data/waypoints_caronly.parquet`, `data/prepared_relut_points.parquet`, `data/segment_candidates.csv`, `data/legs_enriched.parquet`, `data/waypoints_post_car_walk.parquet`, **`legs/corpus_by_leg.parquet`**, rapports `reports/*.json`.

**Contrôles en fin de notebook** : corpus (statuts, premier filtre non vérifié, statistiques), enrichissement, périmètre, **diagnostic de l'appariement** (motifs des legs non appariées, fenêtre de dates avec assert, couverture journalière ; relit les sorties sans relancer).

## 21 — EDA Panel Lémanique

**Paramètres** : `OUTPUT_DIR`, fichiers du questionnaire et de correspondance.

**Contenu** :
1. distributions des waypoints (précision, vitesse, intervalle, distance au précédent) et parts de stationnement déclaré (domicile, travail) ;
2. **corpus commun** : statuts et motifs d'exclusion ;
3. **panneau de sensibilité** : nombre de legs du corpus selon la valeur de chaque filtre (`SWEEPS`), les autres restant à leur valeur du YAML, valeur actuelle en tirets, axe secondaire en % des legs voiture du périmètre ;
4. **couverture par les segments** : legs du corpus selon le nombre de segments ReLUT qui les couvrent, et part dont le dernier segment est retenu (évaluables par les deux méthodes).

**Sorties** : `figures/21_*.png`, `reports/21_*.csv`.

## 32 — Méthode temporelle (ReLUT)

**Paramètres** : `MAX_LEGS` (`None` = tout le corpus ; sinon tirage aléatoire `RANDOM_SEED`), `P_SEARCH` (`auto` = seuils ReLUT selon l'échantillonnage), `MAX_SEARCH_DURATION_MINUTES`, `USE_SURVEY`, `PARKING_VARIABLE`, nombres de legs des cartes.

**Population** : segments des legs `in_corpus`. Pour chaque leg, seul le **dernier segment** est retenu (sans repli).

**Sorties** : `data/relut_predictions.parquet`, synthèses et figures `32_*`, cartes HTML, **`legs/relut_by_leg.parquet`** (statuts : `evaluated`, `excluded_by_corpus`, `no_segment`, `last_segment_rejected`, `not_sampled`… ; durée, distance le long de la trace, distance à vol d'oiseau ; 0 = pas de recherche, NaN = non évalué). Un run limité écrit dans `legs/_samples/<run_id>/`.

## 22 — Méthode spatiale sur waypoints

**Paramètres** : `STAYPOINTS_FILE` (activité de destination), `EXCLUSIONS` (règles **propres** à la méthode spatiale, toutes désactivées par défaut pour traiter exactement le corpus), `METRICS` (anti-bruit, boucle, passage près de la référence), `DETECTION` (critères combinés par « ou »), `ROUTING_REQUESTS_SCOPE`, filtres du panneau, `OSRM_MATCH_URL` et paramètres du map matching.

**Métriques** (fenêtre = dernière entrée dans le cercle de destination → destination) : distance excessive routée (si `routing_by_leg` existe) ; distance excessive sur trace appariée ; distance et temps dans le cercle, mesurés entre waypoints ; boucle (repassage après excursion) ; recul (déplacements projetés négatifs sur l'axe entrée → destination) ; passage près de l'activité.

**Sections** : dictionnaire des variables ; exclusions propres ; **demande de routage** (`data/routing_requests.parquet`) ; panneau d'histogrammes filtrable par stationnement déclaré et motif ; **map matching** OSRM `/match` (distributions brute vs appariée, carte des recalages, diagnostic des détours).

**Sorties** : `data/spatial_waypoints_candidates.parquet` (détail), **`legs/spatial_by_leg.parquet`**, `data/routing_requests.parquet`, tables d'appariement `data/map_matching_*`, `reports/22_*`.

> Relancer le notebook depuis le haut après le routage (27) ou l'appariement : les distances excessives sont alors intégrées.

## 27 — Routage

**Prérequis** : Docker Desktop, conteneurs OSRM et API `xyt-routing-api` lancés (commandes en tête du notebook).

**Paramètres** : `BASE_URL` (API, port 8010), `MODE`, `WEIGHTING` (`distance` = plus court chemin), `MAX_ROUTES` (essai), `FORCE_REROUTE`.

**Sorties** : `data/routing_cache.parquet` (cache par `route_id` : un nouveau run ne route que les paires nouvelles), `data/routing_geometries.parquet`, **`legs/routing_by_leg.parquet`**, carte de comparaison observé / routé.

## 41 — Indicateurs

**Paramètres** : `METHODS` (`relut`, `spatial`, `hybrid` ; tables absentes ignorées), `LAYOUT` (`panels` / `overlay`), `MIN_LEGS_PER_CLASS`, `PURPOSE_COLUMN`, `MIN_CRUISING_DURATION_S`, section H3 (`H3_RESOLUTION`, `H3_MIN_PERSONS`, `H3_MIN_LEGS`, `PARKING_SUPPLY_FILE`), histogramme filtré (`HIST_*`).

**Contenu** : part de legs avec cruising, durée et distance par heure (classe 9 h = 8:30:00–9:29:59, heure locale de fin de leg) et par motif ; répartition des legs avec cruising (référence grise = toutes les legs évaluées) ; carte H3 à couches (temps et distance de cruising cumulés, legs avec cruising, part du temps voiture en cruising, fins de legs, taux d'arrêt, offre de stationnement) ; histogramme des durées ReLUT filtré par stationnement déclaré et motif.

---

## Collection de tables par leg

Toutes les tables de résultats sont **une ligne par leg**, clé `leg_id`, et s'assemblent avec `load_leg_table(config, [...])` : voir [`legs_collection.md`](legs_collection.md).

| Table | Producteur |
|---|---|
| `data/legs_enriched.parquet` | 01 |
| `legs/corpus_by_leg.parquet` | 01 |
| `legs/relut_by_leg.parquet` | 32 |
| `legs/spatial_by_leg.parquet` | 22 |
| `legs/routing_by_leg.parquet` | 27 |
| `legs/persons.parquet` | 32 (dérivée de `legs_enriched`) |

## Dépannage rapide

| Symptôme | Cause habituelle |
|---|---|
| `KeyError` sur une clé de `DATASETS` | cellule de paramètres d'une ancienne version, ou kernel non redémarré |
| `No module named …` | paquet installé dans un autre environnement : `%pip install …` dans le notebook, puis redémarrer le kernel |
| `corpus_by_leg.parquet introuvable` | relancer l'étape 01 avec cette version du code |
| peu de waypoints appariés | voir le diagnostic d'appariement en fin du notebook 01 |
| service de routage injoignable | Docker Desktop, conteneurs et API (message détaillé dans le notebook 27) |
| OSRM `/match` injoignable | `OSRM_MATCH_URL` = port du conteneur OSRM (`docker compose ps`), pas de l'API |
| modification du code sans effet | redémarrer le kernel ; pour l'étape 01, `FORCE_RERUN = True` |
