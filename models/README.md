# Modèle ReLUT ParkingSearchPrediction

`ParkingSearchPrediction.h5` : réseau de détection de la recherche de stationnement du projet start2park
(ReLUT, Frankfurt University of Applied Sciences), licence BSD-3 (`LICENSE_ReLUT.txt`).
Source : https://github.com/ReLUT/parking-search-prediction (dossier `model/`).

Le pipeline lit les poids avec `h5py` et évalue le réseau en numpy (`relut_detection.py`) :
TensorFlow n'est pas nécessaire. Résultats vérifiés identiques au `predict.py` amont
(écart de probabilité < 1e-6, labels identiques).
