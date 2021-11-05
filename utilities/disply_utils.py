# This file contains different display utilities used in the project
from datetime import datetime

from matplotlib import pyplot as plt
from config import SHOW_WARNINGS, SHOW_INFO_MESSAGES
import numpy as np


def warn(msg, with_time=True):
    """
    Show warning message
    """
    if SHOW_WARNINGS is True:
        if with_time is True:
            print(f'[{datetime.now()}] -- {msg}')
        else:
            print(msg)


def info(msg, with_time=True):
    """
    Show info message
    """
    if SHOW_INFO_MESSAGES is True:
        if with_time is True:
            print(f'[{datetime.now()}] -- {msg}')
        else:
            print(msg)


def plot_losses(plot_name, train_loss, val_loss):
    x = [i for i in range(1, len(train_loss) + 1)]
    plt.plot(x, train_loss, color='red', linestyle='dashed', linewidth=1, markersize=12, label='Train')
    plt.plot(x, val_loss, color='green', linestyle='dashed', linewidth=1, markersize=12, label='Val')
    plt.xlabel("Epoch number")
    plt.ylabel("Loss")
    plt.legend(fontsize=14)
    plt.title(plot_name)
    plt.show()
