import pandas as pd
import numpy as np

# sklearn
from sklearn.preprocessing import StandardScaler


def import_dataset():
    """
    Importa il dataset da un csv file e lo divide in features e
    :return:
    """
    ## importa dataset
    data_path = '/home/ec/Credit_Card_AD/datasets/creditcard'
    # data_path = os.path.join(os.getcwd(), 'datasets', 'creditcard')
    data = pd.read_csv(data_path)

    ## Crea una copia del DataFrame originale
    df = data.copy()

    ## Inizializza StandardScaler
    scaler = StandardScaler()

    ## Seleziona le feature da riscalare
    features_to_scale = ['Time', 'Amount']

    ## Riscalare le feature selezionate
    df[features_to_scale] = scaler.fit_transform(data[features_to_scale])
    return df


def reduce_features(df, expl_var_threshold=0.8):
    """
    Funzione che riduce le feature di un dataset in base alla varianza spiegata cumulativa.
    :param df: DataFrame con le feature e il target
    :param expl_var_treshold: soglia della varianza spiegata cumulativa (default 0.8)
    :return: DataFrame ridotto con le feature selezionate e la colonna 'Class'
    """
    from sklearn.decomposition import PCA

    ## crea una copia del dataset senza i target
    df_feat = df.copy().drop('Class', axis=1)

    ## Inizializza la PCA
    pca = PCA()

    ## Adatta il modello PCA ai dati
    pca.fit(df_feat)

    ## Calcola la varianza spiegata
    explained_variance = pca.explained_variance_ratio_

    ## Calcola la varianza spiegata cumulativa
    explained_variance_cumulative = np.cumsum(pca.explained_variance_ratio_)

    ## Trova il numero di componenti che spiegano l'80% della varianza
    num_components = np.argmax(explained_variance_cumulative >= expl_var_threshold) + 1

    ## Calcola i loadings per le componenti selezionate
    loadings = pca.components_.T[:, :num_components] * np.sqrt(pca.explained_variance_[:num_components])

    ## Crea un DataFrame per i loadings e somma i valori assoluti dei loadings per ogni feature
    loadings_df = pd.DataFrame(loadings,
                               columns=[f'PC{i + 1}' for i in range(num_components)],
                               index=df_feat.columns,
                               )

    feature_importance = loadings_df.abs().sum(axis=1)

    ## Ordina le feature in base all'importanza
    sorted_features = feature_importance.sort_values(ascending=False)

    ## Seleziona le feature che contribuiscono all'80% della varianza
    cumulative_importance = sorted_features.cumsum() / sorted_features.sum()
    selected_features = cumulative_importance[cumulative_importance <= expl_var_threshold].index

    ## dataset con le features selezionate
    reduced_df = df_feat[selected_features].copy()
    reduced_df.loc[:, 'Class'] = df['Class']

    return reduced_df


if __name__ == "__main__":

    df = import_dataset()
    df_red = reduce_features(df)