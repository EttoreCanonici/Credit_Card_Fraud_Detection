import pandas as pd
import numpy as np

# sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report

# imblearn
from imblearn.over_sampling import SMOTE, ADASYN

# models
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

def train_models(X, y, oversample=True):
    """
    Funzione che prende in input features e target di un dataset e poi esegue l'addestramento dei modelli.
    Infine, calcola il classification report e mette i risultati in un dataframe.
    :param X: features del dataset
    :param y: target del dataset
    :return: dataframe contenente il classification report per i vari modelli
    """

    ## Dividi il dataset in training e test set
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        random_state=42,
                                                        stratify=y)

    ## Applica SMOTE al training set per provare a ribilanciare il dataset
    if oversample:
        adasyn = SMOTE(random_state=42)
        X_train, y_train = adasyn.fit_resample(X_train, y_train)

    ## Inizializza i classificatori
    classifiers = {
        'SVM': SVC(class_weight='balanced'),
        'SGD Classifier': SGDClassifier(class_weight='balanced'),
        'Random Forest': RandomForestClassifier(class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(),
        'Hist Gradient Boosting': HistGradientBoostingClassifier(class_weight='balanced'),
        'Logistic Regression': LogisticRegression(class_weight='balanced'),
        'k-NN': KNeighborsClassifier(n_neighbors=7),
        'Naive Bayes': GaussianNB(),
        'Stack classifier': stack_model(),
    }


    ## Crea un DataFrame per salvare le performance
    results = []

    ## Evaluation dei classificatori
    for name, clf in classifiers.items():

        print('Modello attualmente in fase di addestramento: {}'.format(name))

        ## Addestra il classificatore e fai le predizioni sul test set
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        ## Calcola il classification report
        report = classification_report(y_test, y_pred, output_dict=True)

        ## Aggiungi i risultati al DataFrame
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                results.append({
                    'Classifier': name,
                    'Class': label,
                    'Precision': metrics.get('precision', np.nan),
                    'Recall': metrics.get('recall', np.nan),
                    'F1-Score': metrics.get('f1-score', np.nan),
                    'Support': metrics.get('support', np.nan)
                })

    ## Crea un DataFrame con i risultati
    results_df = pd.DataFrame(results)

    ## Mostra il DataFrame dei risultati
    print("\nDataFrame delle performance della Cross Validation:")
    print(results_df)

    return results_df


def train_models_CV(X, y):
    """
    Funzione che prende in input features e target di un dataset e poi esegue l'addestramento dei modelli.
    Infine, calcola il classification report e mette i risultati in un dataframe
    :param X: features del dataset
    :param y: target del dataset
    :return: dataframe contenente il classification report per i vari modelli
    """

    ## Dividi il dataset in training e test set, da cui ricaveremo un validation set
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        random_state=42,
                                                        stratify=y)

    ## Inizializza i classificatori
    classifiers = {
        'SVM': SVC(kernel='rbf'),
        'SGD Classifier': SGDClassifier(),
        'Random Forest': RandomForestClassifier(),
        # 'Gradient Boosting': GradientBoostingClassifier(),
        'Logistic Regression': LogisticRegression(),
        'k-NN': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB()
    }

    ## Crea un DataFrame per salvare le performance
    results = []

    ## Evaluation dei classificatori
    for name, clf in classifiers.items():
        ## predizioni della cross validation
        y_pred = cross_val_predict(clf, X_train, y_train, cv=5)

        ## Calcola il classification report
        report = classification_report(y_train, y_pred, output_dict=True)

        ## Aggiungi i risultati al DataFrame
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                results.append({
                    'Classifier': name,
                    'Class': label,
                    'Precision': metrics.get('precision', np.nan),
                    'Recall': metrics.get('recall', np.nan),
                    'F1-Score': metrics.get('f1-score', np.nan),
                    'Support': metrics.get('support', np.nan)
                })

    ## Crea un DataFrame con i risultati
    results_df = pd.DataFrame(results)

    ## Mostra il DataFrame dei risultati
    print("\nDataFrame delle performance della Cross Validation:")
    print(results_df)

    return results_df

def stack_model():
    """
    crea un classificatore ensemble di tipo stacked, il cui meta-modello è la regressione logistica
    :return: l'oggetto classificatore, che potrà essere addestrato e testato
    """

    ## Inizializza i modelli base
    base_learners = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        # ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ('hgb', HistGradientBoostingClassifier(random_state=42)),
        # ('svc', SVC(probability=True, random_state=42)),
        ('svcg', SGDClassifier(random_state=42)),
        ('k-NN', KNeighborsClassifier()),
        ('Naive Bayes', GaussianNB()),
    ]

    ## Istanzia il meta-modello
    meta_model = LogisticRegression(random_state=42, class_weight='balanced')

    ## Create the StackingClassifier
    stacking_clf = StackingClassifier(estimators=base_learners, final_estimator=meta_model)
    return stacking_clf



if __name__ == "__main__":

    from utils.utils_data import *

    df = import_dataset()
    df_red = reduce_features(df, expl_var_treshold=0.3)
    X, y = df_red.drop('Class', axis=1), df_red['Class']
    _ = train_models(X, y)