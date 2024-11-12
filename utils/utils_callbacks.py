import torch


class EarlyStopping:
    #     """
    #     Classe per l'early stopping
    #     """
    def __init__(self, *, min_delta=0.0, patience=0):
        self.eps = 1e-10
        self.min_delta = min_delta + self.eps
        self.patience = patience
        self.best = float("inf")
        self.wait = 0
        self.done = False

    def step(self, current):
        self.wait += 1

        if current < self.best - self.min_delta:
            self.best = current
            self.wait = 0
        elif self.wait >= self.patience:
            self.done = True

        return self.done

class SaveBestModel:
    """
    Classe per fare checkpoints del modello, ovvero salva i pesi se il modello migliora
    """
    def __init__(self, save_path):
        self.best_val_loss = float('inf')
        self.save_path = save_path
        self.best_model_state = None

    def __call__(self, model, val_loss):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_model_state = model.state_dict()
            torch.save(self.best_model_state, self.save_path)
            print(f'Model saved with validation loss: {val_loss:.4f}')

    def get_best_model(self, model):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            return model
        else:
            print("No best model saved yet.")
            return None
