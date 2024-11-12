import sys, os, time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import OneClassSVM
from sklearn.pipeline import Pipeline
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDOneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from utils.utils_data import *
from utils.utils_models import *

def main():
    ## importo il dataset
    df = import_dataset()
    red_df = reduce_features(df, expl_var_treshold=0.8)

    ## estraggo features e targets
    X, y = red_df.drop('Class', axis=1), red_df['Class']

    # Identifica gli outliers con il range interquartile
    score_IQR = compute_silhouette_score(X)

    ## addestro alcuni modelli e metto i classification report in un dataframe che salverò dopo
    df_res = train_unsupervised_models(X, y)

    # Aggiungi i risultati degli outliers al DataFrame finale
    outliers_results = pd.DataFrame({ 'Classifier': ['Outliers (IQR)'], 'Silhouette Score': [score_IQR] })
    final_results_unsupervised = pd.concat([df_res, outliers_results], ignore_index=True)
    print("\nDataFrame delle performance finale:")
    print(final_results_unsupervised)

    ## salvo il dataframe con i risultati
    output_name = os.path.join('/home/ec/Credit_Card_AD/results', 'res_unsupervised_dataframe.pkl')
    df_res.to_pickle(output_name)
    pass


def unsupervised_learning_scores(X, preds, name):
    """
    Calcola le metriche di valutazione non supervisionate e visualizza i dati
    """
    # Calcola il silhouette score
    silhouette = silhouette_score(X, preds)
    print(f'Silhouette Score for {name}: {silhouette}')

    # Visualizza i dati
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=preds, cmap='coolwarm', edgecolor='k', s=20)
    plt.title(f'PCA - Anomaly Detection with {name}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

    # Aggiungi i risultati al DataFrame
    df = pd.DataFrame({
        'Classifier': [name],
        'Silhouette Score': [silhouette]
    })

    return df


# def supervised_learning_scores(y_true, y_pred, name):
#     """
#     Calcola le metriche di valutazione supervisionate e visualizza i dati
#     """
#     # Calcola il classification report
#     report = classification_report(y_true, y_pred, zero_division=1, output_dict=True)
#
#     # Inizializza una lista per salvare i risultati
#     results = []
#
#     # Aggiungi i risultati al DataFrame
#     for label, metrics in report.items():
#         if isinstance(metrics, dict):
#             results.append({
#                 'Classifier': name,
#                 'Class': label,
#                 'Precision': metrics.get('precision', np.nan),
#                 'Recall': metrics.get('recall', np.nan),
#                 'F1-Score': metrics.get('f1-score', np.nan),
#                 'Support': metrics.get('support', np.nan)
#             })
#
#     # Crea un DataFrame con i risultati
#     results_df = pd.DataFrame(results)
#
#     return results_df


def train_unsupervised_models(X, y, supervised=False):
    """
    Addestra e valuta modelli di classificazione supervisionata e non supervisionata
    """

    # Dizionario dei classificatori non supervisionati con la pipeline per SGDClassifier e Nystroem
    unsupervised_classifiers = {
        'SGDOneClassSVM_rbf': Pipeline(
        [('nystroem', Nystroem(kernel='rbf', gamma=0.1, n_components=300)),
         ('sgd', SGDOneClassSVM())
         ]),
        'SGDOneClassSVM_linear': SGDOneClassSVM(),
        'IsolationForest': IsolationForest(contamination=0.05, random_state=42),
        'LocalOutlierFactor': LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True)
        # novelty=True to use LOF in fit/predict mode
        }

    # Crea DataFrames per salvare le performance
    results_unsupervised = []

    X = X.values

    # Evaluation dei classificatori non supervisionati
    for name, clf in unsupervised_classifiers.items():
        print('\n')
        print(f'Modello attualmente in uso: {name}')

        clf.fit(X)
        y_pred = clf.predict(X)

        # Modifica le etichette di previsione per allinearle a 0 (normale) e 1 (anomalo)
        y_pred = np.where(y_pred == -1, 1, 0)

        # Calcola le metriche non supervisionate
        df_unsupervised = unsupervised_learning_scores(X, y_pred, name)
        results_unsupervised.append(df_unsupervised)

    final_results_unsupervised = pd.concat(results_unsupervised, ignore_index=True)


    print("\nDataFrame delle performance dei classificatori non supervisionati:")
    print(final_results_unsupervised)

    return final_results_unsupervised


def identify_outliers(df):
    """
    Funzione per identificare outliers utilizzando l'IQR
    :param df:  input dataframe con le features
    :return: dataframe con gli outliers predetti
    """
    outliers = pd.DataFrame()

    for column in df.columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        column_outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        # outliers = outliers.append(column_outliers)
        outliers = pd.concat([outliers, column_outliers])

    outliers = outliers.drop_duplicates()
    return outliers


def compute_silhouette_score(df):
    """
    Calcola il silhouette score per gli outliers identificati dalla funzione identify_outliers
    :param df: input dataframe con le features
    :return: silhouette score
    """

    outliers = identify_outliers(df)

    # Creare etichette per silhouette score: 1 per outliers e 0 per normali
    labels = np.zeros(df.shape[0])
    labels[df.index.isin(outliers.index)] = 1

    # Standardizzare il dataset
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    # Calcolare il silhouette score
    score = silhouette_score(df_scaled, labels)
    print(f"Silhouette Score degli outliers identificati: {score}")
    return score



if __name__ == "__main__":

    start_time = time.time()
    main()
    print("--- {} seconds ---".format(time.time() - start_time))
    print("--- {} min ---".format(float(time.time() - start_time)/60))

 # usare "export PYTHONPATH=$PYTHONPATH:/home/ec/Credit_Card_AD"




