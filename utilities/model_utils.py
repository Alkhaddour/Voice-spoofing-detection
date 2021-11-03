import os
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, confusion_matrix, roc_auc_score

from config import PROGRESS_STEP_TRAIN, PROGRESS_STEP_VAL, PATIENCE
from utilities.disply_utils import info


class ModelManager:
    def __init__(self, model_name, models_dir):
        self.batch_losses_train = []
        self.batch_losses_val = []
        self.epochs_losses_train = []
        self.epochs_losses_val = []
        self.last_updated = 0
        self.best_batch_val_loss = float('Inf')
        self.best_epoch_val_loss = float('Inf')

        self.models_dir = models_dir
        self.model_name = model_name

    def save_checkpoint(self, chkpt_name, model, valid_loss):
        """
        Saves the model weights with current validation loss
        """
        torch.save({'model_state_dict': model.state_dict(), 'valid_loss': valid_loss},
                   os.path.join(self.models_dir, chkpt_name))

    def load_checkpoint(self, chkpt_path, model):
        """
        Load model weights from file
        """
        state_dict = torch.load(os.path.join(self.models_dir, chkpt_path))
        model.load_state_dict(state_dict['model_state_dict'])
        return model, state_dict['valid_loss']

    def save_metrics(self, metrics_file):
        state_dict = {'train_losses': self.batch_losses_train,
                      'val_losses': self.batch_losses_val}

        torch.save(state_dict, os.path.join(self.models_dir, metrics_file))

    def load_metrics(self, metrics_file):
        state_dict = torch.load(os.path.join(self.models_dir, metrics_file))
        return state_dict['train_losses'], state_dict['val_losses']

    def update_train_loss(self, train_loss, step, total_steps, epoch_id, n_epochs, print_freq=PROGRESS_STEP_TRAIN):
        self.batch_losses_train.append(train_loss)
        if (step + 1) % print_freq == 0 or (step + 1) == total_steps:
            info(f"Training:   batch {step + 1:03d}/{total_steps:03d} "
                 f"from epoch {epoch_id:02d}/{n_epochs:02d} -- Loss = {train_loss}")

    def update_val_loss(self, val_loss, step, total_steps, epoch_id, n_epochs, print_freq=PROGRESS_STEP_VAL):
        self.batch_losses_val.append(val_loss)
        if (step + 1) % print_freq == 0 or (step + 1) == total_steps:
            info(f"Validating: batch {step + 1:03d}/{total_steps:03d} "
                 f"from epoch {epoch_id:02d}/{n_epochs:02d} -- Loss = {val_loss}")

    def update_model(self, model, epoch_val_loss):
        model_file_name = self.model_name + '.pkl'
        metrics_file_name = self.model_name + '_metrics.pkl'
        if epoch_val_loss < self.best_epoch_val_loss:
            self.best_epoch_val_loss = epoch_val_loss
            self.save_checkpoint(model_file_name, model, self.best_epoch_val_loss)
            self.save_metrics(metrics_file_name)
            self.last_updated = 0
            info(f"Model updated, new loss {epoch_val_loss}")
        else:
            self.last_updated += 1
            info(f"Best loss {self.best_epoch_val_loss}")

        return self.last_updated <= PATIENCE

    def update_epochs_loss(self, train_loss, val_loss, epoch_id, n_epochs):
        self.epochs_losses_train.append(train_loss)
        self.epochs_losses_val.append(val_loss)
        info(f"Epoch {epoch_id:02d}/{n_epochs:02d} -- Train loss = {train_loss}, Validation loss = {val_loss} ")


class Metrics:
    def __init__(self, user_defined_metrics=None):
        self.user_defined_metric = user_defined_metrics

    @staticmethod
    def get_best_prec_recall_threshold(y_true, y_prob):
        # find precision recall graph
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        # find F-score at each of its points
        f_score = [0 if (prec + rec == 0) else (2 * prec * rec) / (prec + rec) for (prec, rec) in
                   zip(precision, recall)]
        # locate the index of the largest f score
        idx = np.argmax(f_score)
        return thresholds[idx], f_score[idx]

    @staticmethod
    def calculate_metrics(y_true, y_pred, y_prob, str_formatted=True, sep='\t'):
        metrics = {"Accuracy": None, "F1": None, "AUC": None, "NPV": None, "Precision (PPR)": None,
                   "Sensitivity (Recall/TPR)": None, "Specificity (1-FPR)": None}

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        prec = 0 if tp + fp == 0 else tp / (tp + fp)
        rec = 0 if tp + fn == 0 else tp / (tp + fn)
        f1 = 0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
        sps = 0 if tn + fp == 0 else tn / (tn + fp)
        npv = 0 if tn + fn == 0 else tn / (tn + fn)

        metrics["Accuracy"] = (tp + tn) / (tp + tn + fp + fn) * 100
        metrics["F1"] = f1 * 100
        metrics["AUC"] = roc_auc_score(y_true, y_prob) * 100
        metrics["NPV"] = npv * 100
        metrics["Precision (PPR)"] = prec * 100
        metrics["Sensitivity (Recall/TPR)"] = rec * 100
        metrics["Specificity (1-FPR)"] = sps * 100

        if str_formatted:
            metrics_str = ''
            for metric_name, metric_val in metrics.items():
                metrics_str = metrics_str + f'{metric_name} = {metric_val:.4f}' + f'{sep}'
            return metrics, metrics_str
        return metrics, None
