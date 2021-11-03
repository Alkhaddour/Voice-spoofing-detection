from datetime import datetime

from matplotlib import pyplot as plt
from config import SHOW_WARNINGS, SHOW_INFO_MESSAGES
import numpy as np
import torch


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


def plot_histogram(data, bin_width, plot_title, xlabel, ylabel):
    # fixed bin size
    bin_min = int(min(data))
    bin_max = int(max(data)) + 1
    bins = np.arange(bin_min, bin_max, bin_width)  # fixed bin size

    plt.xlim([0, max(data) + 1])

    plt.hist(data, bins=bins, alpha=0.5)
    plt.title(plot_title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.show()


def plot_losses(plot_name, train_loss, val_loss):
    x = [i for i in range(1, len(train_loss) + 1)]
    plt.plot(x, train_loss, color='red', linestyle='dashed', linewidth=1, markersize=12, label='Train')
    plt.plot(x, val_loss, color='green', linestyle='dashed', linewidth=1, markersize=12, label='Val')
    plt.xlabel("Epoch number")
    plt.ylabel("Loss")
    plt.legend(fontsize=14)
    plt.title(plot_name)
    plt.show()
