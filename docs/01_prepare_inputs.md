# Étape 01 — Préparation des entrées

Notebook : `Notebooks/01_prepare_inputs.ipynb`
Code : `src/parking_search_prediction/`
Configuration : `config/full_waypoints.yaml`

Cette étape transforme les waypoints GPS bruts et les legs (déplacements) en tables prêtes pour les étapes suivantes : exploration spatiale (21), détection spatiale (22) et modèle temporel ReLUT (31–32). Elle ne lance aucun modèle.

Deux jeux de données, choisis par `DATA_SOURCE` dans le notebook :

| | `pl23` — **source de travail** | `declic` — rétrocompatibilité |
|---|---|---|
| Données | Panel Lémanique 2023 (dump Situee) | Déclic Mobilité, traitement d'avant |
| Lecture des waypoints | conversion unique en Parquet partitionné, puis traitement par tranches | CSV lus directement en mémoire |
| Identifiants | legs en `user_id_fors` (panel), waypoints en UUID Motiontag : **correspondance obligatoire** | même identifiant des deux côtés, pas de correspondance |
| Test d'équivalence | oui, avant le premier run complet | sans objet |

---

## 1. Installation

```bash
python -m venv .venv
source .venv/bin/activate            # Windows : .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name parking-search
```

Python 3.10 à 3.12, **pandas 2.x** (pandas 3 n'est pas supporté).

Arborescence attendue :

```
projet/
├── config/full_waypoints.yaml
├── Data/                      # non versionné (.gitignore)
│   ├── GIS/AGGLO_PERIMETRE_AVEC_LAC.gpkg
│   └── GPS/
│       ├── declic/            # waypoints-sample/*.csv, cleaned-base/parquet/legs.parquet
│       └── pl23/              # waypoints/*.csv, 20241120_dump_situee/*.csv
├── docs/01_prepare_inputs.md
├── Notebooks/01_prepare_inputs.ipynb
├── outputs/                   # non versionné ; un sous-dossier par jeu de données
├── requirements.txt
└── src/parking_search_prediction/
    ├── config.py              # chargement du YAML, chemins
    ├── data_prep.py           # lecture, nettoyage, appariement, segmentation
    ├── spatial_study_area.py  # filtre de périmètre sur la destination des legs
    ├── loop_segments.py       # filtre des traces en boucle
    ├── validation.py          # validation des segments pour ReLUT
    ├── waypoint_parquet.py    # conversion CSV → Parquet partitionné
    ├── partitioned.py         # traitement par tranches, synthèse, test d'équivalence
    ├── pipeline.py            # prepare() et enchaînement des étapes
    └── utils.py               # haversine, écriture JSON
```

---

## 2. Démarrage rapide

1. Ouvrir le notebook depuis `Notebooks/` (ou la racine du projet).
2. Cellule **Paramètres** : `DATA_SOURCE = "pl23"` et vérifier les chemins de `DATASETS["pl23"]` (en particulier `waypoint_glob`).
3. Exécuter de haut en bas jusqu'à **Vérification des entrées** : toutes les lignes doivent être ✅.
4. Premier passage :
   - `RUN_CONVERSION = True` → conversion CSV → Parquet (une fois) ;
   - `RUN_EQUIVALENCE_TEST = True` → vérifie que le traitement par tranches donne exactement le même résultat que la lecture CSV directe, sur quelques fichiers.
5. Exécuter **Exécution** (`prepare()`), puis les cellules de contrôle.
6. Passages suivants : remettre `RUN_CONVERSION` et `RUN_EQUIVALENCE_TEST` à `False`.

Pour un essai rapide : `RUN_FULL = False` (2 partitions).

Pour refaire un traitement Déclic d'avant : `DATA_SOURCE = "declic"` ; la conversion et le test d'équivalence sont alors ignorés, et `RUN_FULL = False` limite à `MAX_FILES` / `MAX_USERS` / `MAX_SEGMENTS`.

---

## 3. Données d'entrée

### Waypoints (CSV, un ou plusieurs fichiers)

| Colonne | Rôle | Clé YAML |
|---|---|---|
| `user_id` | identifiant utilisateur (UUID Motiontag) | `raw_waypoints.user_col` |
| `tracked_at` | horodatage, ex. `2023-05-01 05:06:28 +0000` | `time_col` |
| `latitude`, `longitude` | position WGS84 | `lat_col`, `lon_col` |
| `accuracy` | précision GPS (m) | `accuracy_col` |
| `speed` | vitesse (`-1` = inconnue) | `speed_col`, `speed_unit` |
| `altitude`, `course` | conservées, non utilisées | — |

Le délimiteur est détecté automatiquement (`raw_waypoints.delimiter: auto`) : virgule (Déclic), point-virgule (PL23), tabulation…

### Legs

| Source (`leg_source`) | Fichier | Colonnes |
|---|---|---|
| `pl23_legs_parquet` (pl23) | `241120_legs_nogeom.csv` (dump Situee PL23) | préréglées : `user_id_fors`, `leg_id`, `started_at`, `finished_at`, `mode`, `detected_mode` (facultatif) |
| `declic_legs_parquet` (declic) | `legs.parquet` de la base nettoyée Déclic | `leg_matching.*_col` du YAML |
| `declic_legs_export` (declic) | export Déclic CSV/Parquet avec géométrie | préréglées : `user_id`, `leg_id`, `started_at`, `finished_at`, `mode`, `detected_mode`, `storyline_id`, `trip_id`, `mode_niv1/2`, `mode_mrmt` |

Le format (CSV ou Parquet) est déduit de l'extension. Seules les lignes `type = Track` sont gardées. Les dates sans fuseau sont lues comme UTC (vérifié pour le PL23 : `started_at` = `started_at_in_timezone` converti en UTC). Les préréglages se surchargent avec `leg_matching.pl23_columns` / `leg_matching.declic_export_columns`.

### Correspondance d'identifiants (pl23, obligatoire)

Les legs PL23 identifient la personne par l'identifiant panel (`user_id_fors`, ex. `CH3181`), les waypoints par l'UUID Motiontag. La table `241120_user_statistics_nogeom.csv` fait le lien (`user_id_fors` → `user_id_motiontag`). À la lecture des legs :

- `source_user_id` prend la valeur Motiontag, l'identifiant panel reste dans `legs_user_id_original` ;
- les legs d'utilisateurs absents de la table sont retirées et comptées (`user_id_map_*` dans le rapport d'appariement) ;
- une correspondance ambiguë (un identifiant associé à plusieurs autres) **arrête le run**.

Les colonnes utilisateur des waypoints et des legs se règlent **séparément** : il est normal qu'elles n'aient pas le même nom.

### Enrichissement des legs (pl23)

Fait une seule fois sur la table des legs (`leg_enrichment.py`, section `leg_enrichment` du YAML), avant l'appariement :

| Source | Apport |
|---|---|
| `241120_map_track_trip_journey.csv` | `trip_id`, `journey_id` de chaque leg ; legs sans trip retirées (`DROP_LEGS_WITHOUT_TRIP`) |
| `241120_trips.csv` | `trip_purpose`, `trip_modes`, `trip_duration_seconds`, `trip_distance_m`, `trip_started_at` / `trip_finished_at`, `trip_user_id_fors` / `trip_user_id_motiontag`, `leading_activity_id` (activité de destination) |
| calculé | `purpose_agg` (table de l'étape 23, `Autres` à défaut), `time_of_day_end_trip` (bins de l'étape 23, heure de fin de la leg), `weekday` — en `LOCAL_TIMEZONE` (Europe/Zurich ; l'étape 23 utilisait l'heure UTC du fichier, `UTC` pour la reproduire) |
| `EPFL_vague1_v4.csv` | `space_home`, `space_work`, `zone_domicile`, `zone_travail`, `KLASSE_ARE_dom`, `Genre_actuel`, `age`, `Weight` — joints par `user_id_fors` = `IDNO`, même dérivation que l'étape 23 |

**Marche après la voiture.** Pour chaque leg voiture, la leg suivante du même trip est retenue si c'est une marche (`walk_modes`) qui commence entre 0 et `POST_CAR_WALK_MAX_GAP_MINUTES` après la fin de la voiture et mesure au plus `POST_CAR_WALK_MAX_DISTANCE_M`. Colonnes : `next_leg_id`, `next_leg_mode`, `next_leg_gap_seconds`, `post_car_walk_matched`, `post_car_walk_leg_id`, `post_car_walk_gap_seconds`, `post_car_walk_length_m`, `post_car_walk_duration_seconds` et `post_car_walk_reason` (`matched`, `no_next_leg_in_trip`, `next_leg_not_walk`, `negative_gap`, `gap_too_long`, `walk_length_unknown`, `walk_too_long`). Les waypoints de ces marches sont extraits pendant l'appariement dans `waypoints_post_car_walk.parquet`, avec `car_leg_id`. Sont aussi ajoutés `leg_order_in_trip`, `legs_in_trip` et `is_last_car_leg_in_trip` (la leg retenue par l'étape 23).

Ordre de grandeur sur le dump complet (669 277 legs) : 98,5 % des legs ont un trip ; sur 193 684 legs voiture, 59 678 (31 %) sont suivies d'une marche retenue (médiane 181 m, 41 s d'écart) ; 51 % terminent leur trip.

### Corpus commun de legs

Dernière étape (« 11. corpus commun ») : à partir des waypoints voiture des legs du périmètre, une ligne de statistiques par leg (points, durée, distance parcourue, distance origine–destination, trou maximal dans le cercle de destination, durée du trip, cellule H3 de la destination), puis les filtres de `leg_filters` du YAML. Sortie : `legs/corpus_by_leg.parquet` (une ligne par leg, `corpus_status`, `corpus_exclusion_reason(s)`, statistiques `corpus_*`) et `reports/leg_corpus_summary.json`. **Les étapes 32 et 22 ne traitent que les legs `in_corpus`** : les deux méthodes partent du même corpus. Le filtre de précision (`leg_filters.max_accuracy_m`) est appliqué au nettoyage, avant l'appariement. L'effet de chaque filtre est visualisé dans le notebook 21. Voir [`notebooks.md`](notebooks.md).

### Périmètre d'étude

GeoPackage (`STUDY_AREA_BOUNDARY_FILE`, couche `STUDY_AREA_BOUNDARY_LAYER`). Les polygones sont fusionnés.

---

## 4. Paramètres

### Cellule « Paramètres » du notebook

| Paramètre | Effet |
|---|---|
| `DATA_SOURCE` | `pl23` (source de travail) ou `declic` (rétrocompatibilité) |
| `DATASETS[...]["waypoint_source"]` | `parquet` pour pl23, `csv` pour declic ; fixe le mode de lecture |
| `DATASETS[...]["user_id_map_file"]` | table de correspondance des identifiants ; obligatoire pour pl23, `None` pour declic |
| `DATASETS[...]["output_dir"]` | dossier de sortie, **distinct par jeu** pour ne rien écraser |
| `DATASETS[...]["waypoint_parquet_dir"]` | dataset Parquet (pl23) |
| `RUN_FULL` | `False` = run réduit (`MAX_PARTITIONS` pour pl23, `MAX_FILES/USERS/SEGMENTS` pour declic) |
| `N_PARTITIONS` | nombre de partitions à la conversion (le changer impose de reconvertir) |
| `GROUPS_PER_RUN` | partitions par tranche ; moins = moins de RAM ; le changer recalcule toutes les tranches |
| `FORCE_RERUN` | recalculer même les tranches déjà terminées (après une modification du code) |
| `RUN_CONVERSION` | lancer la conversion CSV → Parquet |
| `RUN_EQUIVALENCE_TEST`, `EQUIVALENCE_TEST_FILES` | test CSV ↔ Parquet sur les N premiers CSV |
| `CAR_MODES` | modes considérés comme voiture (`effective_mode`) |
| `APPLY_STUDY_AREA_FILTER` | restreindre aux legs dont la destination est dans le périmètre |
| `LOOP_RADIUS_M`, `APPLY_LOOP_FILTER` | boucles au niveau des segments : laisser `APPLY_LOOP_FILTER = False`, le corpus commun filtre les boucles au niveau des legs (`leg_filters.min_origin_destination_distance_m`) |

### `config/full_waypoints.yaml`

Contient les seuils de nettoyage (`cleaning`), de segmentation (`segmentation`) et de validation (`relut_validation`), ainsi que les colonnes. Les seuils sont ceux de l'ancienne configuration du projet.

---

## 5. Traitement, étape par étape

Chaque ligne affichée par `prepare()` donne l'heure, la durée de l'étape qui vient de finir et les volumes restants.

| # | Étape | Règle | Rapport |
|---|---|---|---|
| 0 | Contrôle utilisateurs (pl23) | part des utilisateurs des waypoints qui ont des legs ; arrêt immédiat si 0 | `prepare_partitioned_summary.json` → `user_overlap` |
| 1 | Lecture | CSV ou partitions ; `waypoint_id = <fichier>:<ligne>` | — |
| 2 | Nettoyage GPS | retire : utilisateur/horodatage manquant, coordonnées hors bornes, précision > `max_accuracy_m` (NaN gardés), doublons ; recalcule distance et vitesse, vitesse inconnue → vitesse calculée, plafond `max_speed_kmh` | `cleaning_report.json` |
| 3 | Lecture des legs | `type = Track`, dates valides, correspondance d'identifiants | `waypoint_leg_matching_summary.json` |
| 3b | Enrichissement (pl23) | trip, motif, moment, questionnaire, marche après voiture | `leg_enrichment_summary.json`, `legs_enriched.parquet` |
| 4 | Appariement | chaque waypoint est rattaché à la leg du même utilisateur dont l'intervalle contient son horodatage ; statuts `matched`, `outside_track`, `overlapping_tracks`, `user_without_tracks` | idem |
| 4b | Traces de marche (pl23) | waypoints appariés aux marches retenues | `waypoints_post_car_walk.parquet` |
| 5 | Filtre voiture | waypoints `matched` dont `effective_mode` ∈ `CAR_MODES` (`effective_mode` = mode confirmé, sinon détecté) | idem |
| 6 | Périmètre | destination de la leg = dernier waypoint apparié ; doit être dans le périmètre | `spatial_study_area_filter_summary.json` |
| 7 | Segmentation | coupe aux trous > `gap_minutes`, retire la queue immobile, garde les segments dans les bornes points/durée/distance | `segment_candidates.csv` (`kept`) |
| 8 | Boucles (segments) | distance début–fin de chaque segment, marquée dans `segment_candidates.csv` ; filtrage désactivé par défaut (`APPLY_LOOP_FILTER = False`) | `loop_filter_summary.json`, `loop_start_end_distance_m`, `is_loop_segment` |
| 9 | ReLUT | format `trip_candidate_id, lon, lat, timestamp, speed_kmh` + validation | `relut_validation_*.{csv,json}` |
| 10 | Écriture | sorties ci-dessous | — |

Les flags de périmètre et de boucle sont **toujours calculés** ; les filtres ne retirent les données que si `APPLY_*` est vrai. Les dénominateurs avant/après sont dans les rapports.

---

## 6. Lecture directe (declic) et traitement par tranches (pl23)

**Lecture CSV directe (declic)** : tous les waypoints sont chargés en mémoire, comme dans le traitement Déclic d'avant. Adapté aux échantillons (quelques GB au plus).

**Parquet partitionné (pl23)** : pour les gros volumes (dizaines de GB).

1. *Conversion* (`waypoint_parquet.py`) : copie fidèle des CSV, partition = `crc32(user_id) % N_PARTITIONS`. Un utilisateur est entièrement dans une partition. Reprenable fichier par fichier ; manifeste `_conversion_manifest.json` (lignes, utilisateurs, lignes sans utilisateur, tailles).
2. *Traitement par tranches* (`partitioned.py`) : `GROUPS_PER_RUN` partitions à la fois passent par exactement le même code que le mode CSV. Aucune étape ne compare deux utilisateurs : le résultat est identique au mode CSV (vérifié par le test d'équivalence).
3. *Reprise* : une tranche terminée écrit `_DONE.json` avec une empreinte (manifeste, partitions, sections de config). Au relancement, elle est sautée si l'empreinte est identique. Une modification **du code** n'est pas détectée → `FORCE_RERUN = True`.
4. *Synthèse* : les compteurs des tranches sont additionnés, les paramètres vérifiés identiques, les ratios et médianes recalculés sur les totaux. Cinq contrôles de cohérence (lignes = manifeste, utilisateurs disjoints, identifiants uniques, totaux cohérents) arrêtent le run en cas d'échec.

---

## 7. Sorties

Dans `outputs/<jeu>/` :

| Fichier | Contenu | Utilisé par |
|---|---|---|
| `data/waypoint_leg_assignments.parquet` | tous les waypoints nettoyés + leg, modes, statut d'appariement | audit |
| `data/waypoints_caronly.parquet` | waypoints voiture après filtre de périmètre | étapes 21–22 |
| `data/legs_enriched.parquet` | pl23 : une ligne par leg — trip, motif, moment, questionnaire, marche suivante | temporel, spatial, analyses (jointure `leg_id`) |
| `data/waypoints_post_car_walk.parquet` | pl23 : traces des marches qui suivent une leg voiture (`car_leg_id`) | spatial, analyses |
| `data/prepared_relut_points.parquet` / `.csv` | points des segments retenus, format ReLUT | étape 32 |
| `data/segment_candidates.csv` | un segment candidat par ligne : stats, `kept`, `is_loop_segment` | audit, 31 |
| `reports/cleaning_report.json` | compteurs de nettoyage | — |
| `reports/leg_enrichment_summary.json` | legs avec / sans trip, questionnaire, marche après voiture (motifs de rejet) | — |
| `reports/waypoint_leg_matching_summary.json` | appariement, modes, périmètre, correspondance d'identifiants | — |
| `reports/waypoint_leg_matching_by_mode.csv` | waypoints appariés par mode | — |
| `reports/spatial_study_area_filter_summary.json` | legs dans / hors périmètre | — |
| `reports/loop_filter_summary.json` | boucles détectées, avant/après | — |
| `reports/relut_validation_by_segment.csv` / `_summary.json` | validation par segment | 32 |
| `reports/prepare_partitioned_summary.json` | pl23 : plan, contrôles, détail par tranche | — |

Pour pl23, `waypoint_leg_assignments.parquet` et `waypoints_caronly.parquet` sont des **dossiers** (un fichier par tranche) : `pd.read_parquet(chemin)` les lit comme un seul fichier. Les autres tables sont concaténées en un seul fichier.

---

## 8. Performance et mémoire

- Le pic de RAM dépend de la taille d'une tranche, pas du volume total. Ordre de grandeur mesuré : ~20 GB pour 4 partitions sur ~350 millions de waypoints (≈ 22 millions de waypoints par tranche, avant appariement). Si la machine swappe (CPU bas, disque très actif), réduire `GROUPS_PER_RUN`.
- Les étapes 7 (segmentation) et 9 (validation) bouclent en Python segment par segment : ce sont les plus lentes sur les gros volumes. Les durées affichées par étape permettent de le vérifier.
- La conversion est rapide (lecture CSV ~150–200 MB/s) et ne se fait qu'une fois par jeu.

---

## 9. Dépannage

| Symptôme | Cause probable | Solution |
|---|---|---|
| ❌ à la vérification des entrées | chemin incorrect | corriger `DATASETS` ; `**/*.csv` si sous-dossiers ; chemins relatifs à la racine |
| `0. contrôle utilisateurs` arrête le run | identifiants différents entre waypoints et legs | le message montre 3 identifiants de chaque côté ; PL23 → `user_id_map_file` ; vérifier la casse et la colonne utilisateur des legs |
| `legs=0` ou `matched_share` proche de 0 | mauvais fichier de legs, autre vague, périodes disjointes | comparer les périodes (legs PL23 : 24.04 – 06.06.2023) |
| `missing required columns` sur les legs | colonnes différentes du préréglage | le message liste les colonnes du fichier ; ajuster `leg_matching.*_col` ou les surcharges |
| `colonnes texte absentes de l'en-tête` (conversion) | délimiteur ou nom de colonne | `raw_waypoints.delimiter`, `user_col`, `time_col` |
| `valeur non numérique` (conversion) | texte dans une colonne float | ajouter la colonne à `parquet_conversion.text_columns` |
| Test d'équivalence en échec | différence réelle entre les modes | `_equivalence_test/equivalence_result.json` liste les différences |
| Tranche sautée avec un résultat faux | code modifié, empreinte inchangée | `FORCE_RERUN = True` une fois |
| Run très lent | swap, ou segmentation/validation | voir § 8 |

---

## 10. Limites connues

- Le filtre de précision retire ~35 % des points sur Déclic, majoritairement hors trajets ; à confirmer sur les legs voiture avant de durcir le seuil.
- Les indicateurs de qualité des legs PL23 (`misdetected_completely`, `merged`, `low_quality_legs_*`) ne sont pas utilisés.
- Les 2 665 legs PL23 qui chevauchent la précédente du même utilisateur produisent des waypoints `overlapping_tracks`, exclus du filtre voiture.
