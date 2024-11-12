import numpy as np
import pandas as pd

import time, os

from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from utils.utils_data import *

import optuna


def objective(trial, X_train, X_val):
    """
    Funzione obiettivo che viene usata per fare l'ottimizzazione di iperpatametri di Isolation Forest
    :param trial: trial passato dall'ottimizzatore di optuna
    :param X_train: train set
    :param X_val: validation set
    :return: silhouette score
    """

    ## Definisci la griglia di iperparametri
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_samples': ['auto', 0.5, 0.75, 1.0],
        'contamination': ['auto', 0.001, 0.01, 0.1],
        'max_features': [0.5, 1.0],
        'bootstrap': [True, False]
    }

    # Suggerisci iperparametri dalla griglia
    n_estimators = trial.suggest_categorical("n_estimators", param_grid['n_estimators'])
    max_samples = trial.suggest_categorical("max_samples", param_grid['max_samples'])
    contamination = trial.suggest_categorical("contamination", param_grid['contamination'])
    max_features = trial.suggest_categorical("max_features", param_grid['max_features'])
    bootstrap = trial.suggest_categorical("bootstrap", param_grid['bootstrap'])

    # Definisci il modello
    iso_forest = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        max_features=max_features,
        bootstrap=bootstrap,
        random_state=42
    )

    # Fit del modello su X_train
    iso_forest.fit(X_train)

    # Predizioni su X_val
    labels = iso_forest.predict(X_val)
    labels = np.where(labels == -1, 1, 0)

    # Calcolo del Silhouette Score su X_val
    score = silhouette_score(X_val, labels)

    return score


def main():
    """
    Funzione che importa il dataset, esegue l'ottimizzazione di iperparametri con optuna,
     testa il miglior modello sul test set e infine salva i risultati su file
    :return: None
    """
    ## Importa il dataset
    df = import_dataset()
    red_df = reduce_features(df, expl_var_threshold=0.8)

    ## Estrai features e targets
    X, y = red_df.drop('Class', axis=1), red_df['Class']
    X = X.values

    # Suddividi il dataset in training, validation e test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Crea uno studio Optuna e ottimizza la funzione obiettivo
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, X_val), n_trials=100)

    # Ottieni i migliori iperparametri
    best_params = study.best_params
    best_value = study.best_value

    # Stampa i migliori iperparametri
    print("Migliori Iperparametri trovati:")
    print(best_params)

    # Stampa il miglior score su X_val
    print("Miglior Score su X_val:")
    print(best_value)

    # Valuta il miglior modello su X_test
    best_model = IsolationForest(
        n_estimators=best_params["n_estimators"],
        max_samples=best_params["max_samples"],
        contamination=best_params["contamination"],
        max_features=best_params["max_features"],
        bootstrap=best_params["bootstrap"],
        random_state=42
    )
    best_model.fit(X_train)
    test_labels = best_model.predict(X_test)
    test_labels = np.where(test_labels == -1, 1, 0)
    test_score = silhouette_score(X_test, test_labels)

    print("Silhouette Score su X_test:")
    print(test_score)

    # Salva la configurazione del miglior modello e i risultati in un dataframe
    results_df = pd.DataFrame([best_params])
    results_df["val_silhouette_score"] = best_value
    results_df["test_silhouette_score"] = test_score

    # Salva il dataframe in un file CSV
    save_path = os.path.join("/home/ec/Credit_Card_AD/results", "best_isolation_forest_results.csv")
    results_df.to_csv(save_path, index=False)
    pass



if __name__ == "__main__":

    start_time = time.time()
    main()
    print("--- {} seconds ---".format(time.time() - start_time))
    print("--- {} min ---".format(float(time.time() - start_time)/60))