# coding: utf-8

import numpy as np


def xavier_init(dims, bias=False):
    """
    Input:
    dims: dimension of the matrix to be initialized; dims = (f_i, f_o)

    Compute:
    Xavier initialization for Tanh layers

    Output:
    W: weight matrix with Xavier initialization
    b: bias vector with Xavier initialization (if bias=True)
    """
    r = np.sqrt(6 / sum(dims))  # compute the radius
    if bias:
        # normalize distribution to Unif(-1, 1), then multiply by radius
        b = (2 * np.random.random((1,dims[1])) - 1) * r              
        return b
    else:
        # normalize distribution to Unif(-1, 1), then multiply by radius
        W = (2 * np.random.random(dims) - 1) * r
        return W


def SGD(theta, g_hat, epsilon, wd):
    """
    Input:
    theta: weights to be updated (numpy array or list of numpy arrays)
    g_hat: gradient of loss wrt weights (np array or list of np arrays)
    epsilon: learning rate (float)
    wd: weight decay coefficient (float)

    Compute:
    stochastic gradient descent (batch size determined elsewhere)

    Output:
    t: updated weights if theta was list - not returned
    theta: updated weights if theta was numpy array (numpy array) - not returned
    """
    # theta and g_hat could either be numpy arrays or lists of numpy arrays
    if type(theta) is list: # if they're lists (of numpy arrays)
        for t, g in zip(theta, g_hat):  # perform SGD on each element
            SGD(t, g, epsilon, wd)
    else:
        theta -= epsilon * (g_hat + 2 * wd * theta) # weight update


def RMSProp(theta, g_hat, rho, epsilon, r, delta=1e-6):
    """
    Input:
    theta: weights to be updated (numpy array or list of numpy arrays)
    g_hat: gradient of loss wrt weights (np array or list of np arrays)
    rho: decay parameter #1 (float)
    epsilon: learning rate (float)
    delta: decay parameter #1 (float)

    Comptue:
    RMSProp based on Lecture 6, slide #30 (CMSC 35246)

    Output:
    t: updated weights if theta was list - not returned
    theta: updated weights if theta was numpy array (numpy array) - not returned
    """
    # theta and g_hat could either be numpy arrays or lists of numpy arrays
    if type(theta) is list: # if they're lists (of numpy arrays)
        for t, g in zip(theta, g_hat):  # perform RMSProp on each element
            RMSProp(t, g, rho, epsilon, delta, r)
    else:
        # update exponentially decaying average of gradients
        r *= rho
        r += (1 - rho) * np.multiply(g_hat, g_hat)
        
        # alter the gradient using r
        dtheta = np.multiply(- epsilon / (delta + np.sqrt(r)), g_hat)

        theta += dtheta # weight update

