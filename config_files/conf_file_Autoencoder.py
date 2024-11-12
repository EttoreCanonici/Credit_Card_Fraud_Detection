conf = {
    'quick_test_AE': {
        # addestramento
        'epochs': 15,
        'bacth_size': 256,
        'lr': 1e-3,
        'patience_es': 15,
        'patience_lr': 5,
        'weight_decay': 1e-5,
        # modello
        'hidden_dim_AE': 10,
        'dropout': 0.2,
        'num_trials': 7,
        # feature selection
        'expl_var_threshold': 1,
        # filenames
        'save_path': "/home/ec/Credit_Card_AD/models/logs/Autoencoder_quick",  # dove verranno salvati i pesi del modello
        'res_path': "/home/ec/Credit_Card_AD/results/AE_results_quick.pkl"
        # dove verranno salvate le loss, la threshold ottimale e lo score finale
    },

    'test_AE': {
        # addestramento
        'epochs': 150,
        'bacth_size': 256,
        'lr': 1e-3,
        'patience_es': 15,
        'patience_lr': 5,
        'weight_decay': 1e-5,
        # modello
        'hidden_dim_AE': 6,
        'dropout': 0.2,
        'num_trials': 20,
        # feature selection
        'expl_var_threshold': 1,
        # filenames
        'save_path': "/home/ec/Credit_Card_AD/models/logs/Autoencoder", # dove verranno salvati i pesi del modello
        'res_path': "/home/ec/Credit_Card_AD/results/AE_results.pkl" # dove verranno salvate le loss, la threshold ottimale e lo score finale
    },

}

