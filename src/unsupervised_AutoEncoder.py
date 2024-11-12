import numpy as np
import pandas as pd
import time, sys, os, pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score

from utils.utils_data import *
from utils.utils_models import *
from utils.utils_deep_learning_AD import *
from utils.utils_callbacks import *
from utils.Autoencoder import *
from config_files.conf_file_Autoencoder import conf




def main():
    """
    funzione che importa il dataset, addestra il modello, ottimizza la soglia di rilevamento degli outliers
    :return: None
    """

    useConf = sys.argv[1]
    myConf = conf[useConf]

    ## importo il dataset
    df = import_dataset()
    red_df = reduce_features(df, expl_var_threshold=1)

    ## estraggo features e targets
    X, y = red_df.drop('Class', axis=1), red_df['Class']
    X = np.array(X.values)

    ## Suddividi il dataset in training, validation e test
    X_train, X_temp = train_test_split(X, test_size=0.4, random_state=42)
    X_val, X_test = train_test_split(X_temp, test_size=0.5, random_state=42)

    ## Crea i DataLoader
    batch_size = myConf['bacth_size']
    train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(X_val, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(X_test, batch_size=batch_size, shuffle=False)

    ## Definisci il dispositivo (CPU o GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## Crea l'autoencoder
    input_dim = X_train.shape[1]
    model = Autoencoder(input_dim, myConf['hidden_dim_AE'], myConf['dropout']).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=myConf['weight_decay'])

    ## Addestra l'autoencoder
    epochs = myConf['epochs']
    res_dict = train_autoencoder(model,
                      train_loader,
                      val_loader,
                      optimizer,
                      criterion,
                      device,
                      epochs,
                      patience_es=myConf['patience_es'],
                      patience_lr=myConf['patience_lr'],
                      save_path=myConf['save_path'],
                      )

    ## Rileva anomalie sui dati di validazione per trovare la soglia ottimale
    reconstruction_errors = detect_anomalies(model, val_loader, device)
    best_threshold = find_optimal_threshold(reconstruction_errors, X_val, myConf['num_trials'])
    print(f"Miglior soglia trovata: {best_threshold}")

    ## Rileva anomalie sui dati di test con la soglia ottimale
    reconstruction_errors_test = detect_anomalies(model, test_loader, device)
    ## se la threshold è superiore al max(reconstruction_errors_test) esegui un rescaling
    if best_threshold > reconstruction_errors.max():
        rescaled_best_threshold = best_threshold * (reconstruction_errors_test.max()/reconstruction_errors.max())
    else:
        rescaled_best_threshold = best_threshold
    anomalies_test = reconstruction_errors_test > rescaled_best_threshold
    print(f"Number of anomalies detected: {np.sum(anomalies_test)}")

    ## Calcola il silhouette score
    silhouette_score_value = calculate_silhouette_score(X_test, anomalies_test)
    print(f"Silhouette Score: {silhouette_score_value}")

    ## Salva la configurazione del miglior modello e i risultati in un dict
    res_dict['best_threshold'] = best_threshold
    res_dict['best_silhouette_score'] = silhouette_score_value


    ## Salva il dict con pickle library
    with open(myConf['res_path'], 'wb') as f:
        pickle.dump(res_dict, f)

    pass


if __name__ == "__main__":
    """
    per eseguire: python + unsupervised_AutoEncoder.py + "conf da conf_file_Autoencoder"  
    """
    start_time = time.time()
    main()
    print("--- {} seconds ---".format(time.time() - start_time))
    print("--- {} min ---".format(float(time.time() - start_time)/60))