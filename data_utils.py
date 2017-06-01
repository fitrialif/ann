# coding: utf-8

import numpy as np
from matplotlib import pyplot as plt
import os


def int_to_onehot(y_raw, vals):
    """
    Input:
    y_raw: array of integer labels (numpy array)
    vals: number of classes

    Compute:
    matrix of one-hot vectors corresponding to labels

    Output:
    y: one-hot vector form of y_raw
    """
    n = y_raw.shape[0]
    y = np.zeros((n, vals))
    y[np.arange(n), y_raw.astype(int)] = 1
    return y

def plot_val(directory, LR, val_errors, test_errors, rho=None):
    """
    Input:
    directory: name of directory where plots will be saved (string)
    LR: learning rate used (float)
    val_errors: list of validation errors

    Compute:
    plot of validation error over number of epochs

    Output:
    fig: plot of val. errors over number of epochs (.png file)
    LRval_errors: list of the validation errors (.csv file)
    """
    n = len(val_errors)
    indices = [i for i in range(1, n+1)]

    fig = plt.figure()
    plt.xlabel("Epochs")
    plt.ylabel("Validation Error %")
    plt.plot(indices, val_errors, marker='o')

    if rho is not None:
        r = rho * 100
        plt.title("Error over Epochs " + "(rho = " + str(r) + ")" )
        for xy in zip(indices, val_errors):
            plt.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')
        plt.savefig(directory + "/" + str(r) + "RHOplot.png")
        plt.clf()

        np.savetxt(directory + "/" + str(r) + "RHOval_errors.csv", val_errors, delimiter=",")
        np.savetxt(directory + "/" + str(r) + "RHOtest_errors.csv", test_errors, delimiter=",")
    else:
        plt.title("Error over Epochs " + "(LR = " + str(LR) + ")" )
        for xy in zip(indices, val_errors):
            plt.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')
        plt.savefig(directory + "/" + str(LR) + "LRplot.png")
        plt.clf()

        np.savetxt(directory + "/" + str(LR) + "LRval_errors.csv", val_errors, delimiter=",")
        np.savetxt(directory + "/" + str(LR) + "LRtest_errors.csv", test_errors, delimiter=",")

def record_model(prefix, N_hidden, LR, val_errors, W1, b1, W2, b2):     
    new_dir = prefix + "N" + str(N_hidden) + "LR" + str(LR)  # name of directory for plots, weights, losses
    os.mkdir(new_dir)

    np.savetxt(new_dir + "/W1.csv", W1, delimiter=",")
    np.savetxt(new_dir + "/b1.csv", b1, delimiter=",")
    np.savetxt(new_dir + "/W2.csv", W2, delimiter=",")
    np.savetxt(new_dir + "/b2.csv", b2, delimiter=",")
    np.savetxt(new_dir + "/val_losses.csv", val_losses, delimiter=",")

    plt.xlabel("Epochs")
    plt.ylabel("Cross-entropy Error (Validation Set)")
    plt.title("Error over Epochs " + "(N = " + str(N_hidden) + "; LR = " + str(LR) + ")" )
    plt.plot([x for x in range(1, len(val_losses) + 1)], val_losses, marker='o')
    plt.savefig(new_dir + "/plot.png")
    plt.clf()

