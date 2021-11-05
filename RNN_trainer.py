# This file defines model train function

import torch
import numpy as np
from config import LSTM_NUM_LAYERS, HIDDEN_SIZE, PATIENCE
from utilities.disply_utils import info


def train(model, model_manager, train_loader, val_loader, n_epochs, loss_fn, optimizer, scheduler, device):
    """
    Train model
    :param model: Model to be trained
    :param model_manager: Model manager to keep tracking the model performance
    :param train_loader: Train Loader (pytorch dataloader)
    :param val_loader: Validation Loader (pytorch dataloader)
    :param n_epochs: Number of epochs
    :param loss_fn: Loss function
    :param optimizer: Optimizer object
    :param scheduler: Scheduler object
    :param device: Accelerator used for training (GPU or CPU)
    :return:
    """
    info(f"Training")
    model = model.to(device)

    for epoch_id in range(1, n_epochs + 1):
        info(f"Current lr = {scheduler.get_last_lr()}")
        epoch_train_losses = []
        epoch_val_losses = []
        for batch_id, (_, batch, labels) in enumerate(train_loader):
            labels = labels.reshape(-1).float().to(device)
            outputs = predict_batch(model, batch, device)
            loss = loss_fn(outputs, labels)
            epoch_train_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            model_manager.update_train_loss(loss.item(), batch_id, len(train_loader), epoch_id, n_epochs)
        for batch_id, (_, batch, labels) in enumerate(val_loader):
            labels = labels.reshape(-1).float().to(device)
            outputs = predict_batch(model, batch, device)
            loss = loss_fn(outputs, labels)
            epoch_val_losses.append(loss.item())
            model_manager.update_val_loss(loss.item(), batch_id, len(val_loader), epoch_id, n_epochs)

        # Update learning rate
        scheduler.step()
        # Find average loss over epoch
        train_epoch_loss = np.mean(epoch_train_losses)
        val_epoch_loss = np.mean(epoch_val_losses)
        model_manager.update_epochs_loss(train_epoch_loss, val_epoch_loss, epoch_id, n_epochs)

        # check if model still learning
        is_learning = model_manager.update_model(model, val_epoch_loss)
        if is_learning is False:
            info(f"Training stopped because model did not improve for {PATIENCE} epochs")
            return model, model_manager

    return model, model_manager


def predict_batch(model, batch, device):
    model = model.to(device)
    batch = batch.to(device)

    current_batch_size = batch.shape[0]
    h0 = torch.randn(LSTM_NUM_LAYERS, current_batch_size, HIDDEN_SIZE).to(device)  # D * num_layers, N, H_out
    c0 = torch.randn(LSTM_NUM_LAYERS, current_batch_size, HIDDEN_SIZE).to(device)
    outputs = model(batch, h0, c0)
    outputs = outputs.reshape(-1)
    return outputs

