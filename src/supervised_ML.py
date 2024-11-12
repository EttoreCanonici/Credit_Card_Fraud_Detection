import pandas as pd
import numpy as np
import os, time, sys

from utils.utils_data import *
from utils.utils_models import *



def main():
    """
    funzione che importa il dataset, esegue l'analisi e stampa su file i risultati
    :return: None
    """


    ## importo il dataset
    df = import_dataset()
    red_df = reduce_features(df, expl_var_treshold=0.8)

    ## estraggo features e targets
    X, y = red_df.drop('Class', axis=1), red_df['Class']

    ## addestro alcuni modelli e metto i classification report in un dataframe che salverò dopo
    df_res = train_models(X, y, oversample=False)

    ## salvo il dataframe con i risultati
    output_name = os.path.join('/home/ec/Credit_Card_AD/results', 'res_dataframe.pkl')
    df_res.to_pickle(output_name)
    pass



if __name__ == "__main__":

    start_time = time.time()
    main()
    print("--- {} seconds ---".format(time.time() - start_time))
    print("--- {} min ---".format(float(time.time() - start_time)/60))
