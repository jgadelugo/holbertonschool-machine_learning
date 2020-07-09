#!/usr/bin/env python3
"""performs back propagation over a pooling layer of a neural network"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """performs back propagation over a pooling layer of a neural network
    @dz: np.ndarray shape(m, h_new, w_new, c_new) containing the partial
    derivatives with respect to the unactivated output
    @A_prev: np.ndarray shape(m, h_prev, w_prev, c_prev) ouput of the previous
    layer
        @h_prev: height of the previous layer
        @w_prev: width of previous layer
        c_prev: number of channels in previous layer
    @W: np.ndarray shape(kh, kw, c_prev, c_new) kernels for the conv.
        @kh: filter height
        @kw: filter width
    b: np.ndarray shape(1, 1, 1, c_new)
    @padding: same or valid, deppending on padding
    @stride: tuple of (sh, sw)
        @sh: stride height
        @sw: stride width
    Return partial derivatives with respect to the prev layer, kernel,
    and biases
    """
    kh, kw, c_prev, c_new = W.shape

    m, h_new, w_new, c_new = dZ.shape

    # variables of previous layer
    m, h_prev, w_prev, c_prev = A_prev.shape

    sh, sw = stride

    if padding == 'same':
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:
        pw = 0
        ph = 0
    
    A_prev = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')
    # initializing the output
    dA = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)

    for i in range(m):
        for j in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    dZ_cust = dZ[i, j, w, c]
                    W_cust = W[:, :, :, c]
                    dA[i, j*sh: j*sh+kh, w*sw:w*sw+kw] += W_cust * dZ_cust
                    dW[:, :, :, c] += A_prev[i, j*sh: j*sh+kh,
                                            w*sw: w*sw+kw] * dZ_cust

    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    dA_prev = dA[:, ph:dA.shape[1]-ph, pw:dA.shape[2]-pw]
    return dA_prev, dW, db
