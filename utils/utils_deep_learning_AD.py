import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from utils.utils_data import *
from utils.utils_models import *
from utils.utils_callbacks import *

import time





def evaluate(model, data_loader, criterion, device):
    """
    Funzione per valutare il modello sui dati di validazione
    :param model: modello da valutare
    :param data_loader: data_loader per la validazione
    :param criterion: loss function
    :param device: cpu o gpu
    :return: average validation loss
    """
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device, dtype=torch.float)
            output = model(data)
            loss = criterion(output, data)
            val_loss += loss.item()
    return val_loss / len(data_loader)


def train_autoencoder(model, train_loader, val_loader, optimizer, criterion, device, epochs, patience_es, patience_lr, save_path):
    """
    Funzione per addestrare l'autoencoder per più epoche con valutazione, early stopping e checkpoint
    :param model: modello da addestrare
    :param train_loader: data_loader per il training
    :param val_loader: data_loader per la validazione
    :param optimizer: ottimizzatore
    :param criterion: loss function
    :param device: cpu o gpu
    :param epochs: numero di epoche
    :param patience_es: numero di epoche senza miglioramenti per fermare l'addestramento
    :param patience_lr: numero di epoche senza miglioramenti per fare scattare la diminuzione del learning rate
    :param save_path: percorso per salvare il miglior modello
    :return: dizionario con le losses di trainig e di validazione
    """
    early_stopping = EarlyStopping(patience=patience_es)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience_lr, verbose=True)
    save_best_model = SaveBestModel(save_path)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

        scheduler.step(val_loss)
        save_best_model(model, val_loss)

        if early_stopping.step(val_loss):
            print('Early stopping')
            break


    return {"train_losses": train_losses, "val_losses": val_losses}


def train_one_epoch(model, data_loader, optimizer, criterion, device):
    """
    funzione che esegue l'addestramento del modello per una singola epoca
    :param model: modello da addestrare
    :param data_loader: data_loader per fare il trainin
    :param optimizer: ottimizzatore
    :param criterion: loss function
    :param device: cpu o gpu
    :return: average training loss
    """
    model.train()
    train_loss = 0

    for data in data_loader:
        data = data.to(device, dtype=torch.float)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(data_loader)



def detect_anomalies(model, data_loader, device):
    """
    Funzione per rilevare anomalie
    :param model: modello addestrato
    :param data_loader: data_loader da analizzare
    :param device: cpu o gpu
    :return:
    """
    model.eval()
    reconstruction_errors = []
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device, dtype=torch.float)
            output = model(data)
            loss = nn.functional.mse_loss(output, data, reduction='none')
            loss = loss.mean(dim=1)
            reconstruction_errors.extend(loss.cpu().numpy())
    return np.array(reconstruction_errors)


def calculate_silhouette_score(X, anomalies):
    """
    Funzione che calcola il silhouette score di un dataset X
    :param X: dati da analizzare
    :param anomalies: anomalie precedentemente rilevate
    :return: silhouette score
    """
    ## Creare le etichette per il silhouette score: 1 per anomalia, 0 per normale
    labels = np.zeros(len(X), dtype=int)
    labels[anomalies] = 1

    ## Verificare se ci sono almeno due etichette distinte
    if len(set(labels)) < 2:
        raise ValueError("Number of labels is less than 2. Silhouette score requires at least two distinct labels.")

    score = silhouette_score(X, labels)
    return score


def find_optimal_threshold(reconstruction_errors, X_val, num_trials=10):
    """
    Funzione che esegue l'ottimizzazione della soglia per fare detection di outliers
    :param reconstruction_errors: errori di ricostruzione
    :param X_val: validation set
    :return: miglior threshold trovata
    """
    # Genera l'array delle varie thresholds
    array = np.linspace(min(reconstruction_errors), max(reconstruction_errors), num_trials)
    # Elimina il primo e l'ultimo elemento, tanto sono thresholds banali
    modified_array = array[1:-1]

    best_score = -1
    best_threshold = 0
    for threshold in modified_array:
        anomalies = reconstruction_errors > threshold
        try:
            score = calculate_silhouette_score(X_val, anomalies)
            if score > best_score:
                best_score = score
                best_threshold = threshold
        except ValueError as e:
            # Gestione del caso in cui ci sia una sola etichetta
            print(e)
    return best_threshold
