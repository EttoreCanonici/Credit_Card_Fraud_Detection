import numpy as np
import pandas as pd
import sys, os, time

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer, classification_report

# sklearn models
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

from utils.utils_data import *
from utils.utils_models import *

# # Aggiungi il percorso della directory principale del progetto al sys.path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def train_RF(rebalance=False, fine_tune=True):
    """
    Funzione che esegue l' addestramento/ottimizzazione di iperpatametri di un classificatore Random Forest su un set di dati
    :param rebalance: specificare se si richiede il ribilanciamento del dataset con l'algoritmo SMOTE (default=False)
    :param fine_tune: specificare se si desidera eseguire l'ottimizzazione di iperparametri o il semplice addestramento (default=True)
    :return: un DataFrame contenente il classification report del modello di Random Forest usato
    """
    ## Load del dataset
    df = import_dataset()
    df_red = reduce_features(df, expl_var_threshold=0.8)
    X, y = df_red.drop('Class', axis=1), df_red['Class']

    ## Dividi il dataset in training e test set, da cui ricaveremo un validation set
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        random_state=42,
                                                        stratify=y)

   ## Applica SMOTE per bilanciare il dataset
    if rebalance:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    ## Istanzio un classificatore Random Forest
    rf_clf = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1, max_features='sqrt')

    if fine_tune:
        ## definizione spazio degli iperparametri (griglia in questo caso)
        param_grid = {
            'n_estimators': [10, 100, 500, 1000],
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        }

        ## Crea un oggetto scorer che usa l'F1 come metrica
        scorer = make_scorer(f1_score)

        ## Crea l'oggetto GridSearchCV
        grid_search = GridSearchCV(estimator=rf_clf, param_grid=param_grid, scoring=scorer,cv=5)

        ## Fitta GridSearchCV
        grid_search.fit(X_train, y_train)

        ## Stampo i migliori iperparametri trovati
        print("Best parameters:", grid_search.best_params_)
        print("Best score:", grid_search.best_score_)

        ## miglior modello addesrato tramite GridSearchCV
        model = grid_search.best_estimator_


        ## converto i risultati in un DataFrame
        results_df = pd.DataFrame(grid_search.cv_results_)

        ## MOstra le colonne di interesse
        columns_of_interest = ['mean_test_score', 'std_test_score', 'params']
        df = results_df[columns_of_interest]
        print(df)

        ## Inizializza un nuovo modello con i migliori parametri
        # model = RandomForestClassifier(**grid_search.best_params_, class_weight='balanced', random_state=42, n_jobs=-1, max_features='sqrt')

    else:
        model = rf_clf
        model.fit(X_train, y_train)


    preds = model.predict(X_test)
    ## probabilità, utili in caso si volesse ottimizzare una soglia per decidere quali sono le frodi
    probs = model.predict_proba(X_test)
    report = classification_report(y_test, preds, output_dict=True)

    ## converto il report in DataFrame
    report_df = pd.DataFrame(report).transpose()
    print(report_df)

    return report_df




if __name__ == "__main__":

    start_time = time.time()
    print('Random Forest')
    df = train_RF(fine_tune=False, rebalance=False)
    print("--- {} seconds ---".format(time.time() - start_time))
    print("--- {} min ---".format(float(time.time() - start_time)/60))