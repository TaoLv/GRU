import theano
from theano import tensor as T
import numpy as np

import mkl_gru
from mkl_gru import GRU

np.random.seed(12345)


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


x = np.random.rand(1, 80, 620).astype(np.float64)
w_x = np.random.rand(3*620, 1000).astype(np.float64) - np.random.rand(3* 620, 1000).astype(np.float64)
w_xh = w_x[0:620, :]
w_xz = w_x[620: 2*620, :]
w_xr = w_x[2*620:, :]

w_h = np.random.rand(3*1000, 1000).astype(np.float64) - np.random.rand(3*1000, 1000).astype(np.float64)
w_hh = w_h[0:1000, :]
w_hz = w_h[1000:2000, :]
w_hr = w_h[2000:, :]

hid = np.zeros((80, 1000), np.float64)


def GRU_NP():
    global x, w_xr, w_xz, w_xh, w_hr, w_hz, w_hh, hid
    for i in range(x.shape[0]):
        x_r = np.dot(x[i], w_xr)
        x_z = np.dot(x[i], w_xz)
        x_h = np.dot(x[i], w_xh)
        
        t = x_z + np.dot(hid, w_hz)
        z_t = sigmoid(t)

        t = x_r + np.dot(hid, w_hr)
        r_t = sigmoid(t)

        t0 = np.dot(hid, w_hh)
        t = x_h + r_t * t0
        can_h_t = np.tanh(t)

        hid = (1. - z_t) * hid + z_t * can_h_t
    return hid


def GRU_MKL():
    global x, w_x, w_h, hid
    # avoid hid is modified by numpy function
    hid = np.zeros((80, 1000), np.float64)
    
    X = T.dtensor3('X')
    W_x = T.dmatrix('W_x')
    W_h = T.dmatrix('W_h')
    Hid = T.dmatrix('Hid')

    Z = GRU(hid=1000, return_sequences=True, max_len=100)(X, W_x, W_h, Hid)
    f = theano.function([X, W_x, W_h, Hid], Z)

    o, zt, rt, hcan, hht = f(x, w_x, w_h, hid)
    return o


if __name__ == '__main__':
    a = GRU_NP()
    b = GRU_MKL()

    assert np.allclose (a, b[-1])
