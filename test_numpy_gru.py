import theano
from theano import tensor as T
import numpy as np

import mkl_gru
from mkl_gru import GRU


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


x = np.random.rand(32, 80, 620).astype(np.float64)
w_x = np.random.rand(3*620, 1000).astype(np.float64) - np.random.rand(3* 620, 1000).astype(np.float64)
w_xh = w_x[0:620, :]
w_xz = w_x[620: 2*620, :]
w_xr = w_x[2*620:, :]

w_h = np.random.rand(3*1000, 1000).astype(np.float64) - np.random.rand(3*1000, 1000).astype(np.float64)
w_hh = w_h[0:1000, :]
w_hz = w_h[1000:2000, :]
w_hr = w_h[2000:, :]

b = np.random.rand(3*1000).astype(np.float64) - np.random.rand(3*1000).astype(np.float64)
b_h = b[0:1000]
b_z = b[1000:2000]
b_r = b[2000:]

hid = np.zeros((80, 1000), np.float64)


def GRU_NP():
    global x, w_xr, w_xz, w_xh, w_hr, w_hz, w_hh, b_r, b_z, b_h, hid
    for i in range(32):
        x_r = np.dot(x[i], w_xr) + b_r
        x_z = np.dot(x[i], w_xz) + b_z
        x_h = np.dot(x[i], w_xh) + b_h
        
        t = x_z + np.dot(hid, w_hz)
        z_t = sigmoid(t)

        t = x_r + np.dot(hid, w_hr)
        r_t = sigmoid(t)

        t0 = np.dot(hid, w_hh)
        t = x_h + r_t * t0
        can_h_t = np.tanh(t)

        hid = (1. - z_t) * hid + z_t * can_h_t
        # print("%.9f, %.9f, %.9f" %(hid[0,0], hid[1, 0], hid[-1,-1]))
    return hid


def GRU_MKL():
    global x, w_x, w_h, b
    
    X = T.dtensor3('X')
    W_x = T.dmatrix('W_x')
    W_h = T.dmatrix('W_h')
    B = T.dvector('b')

    Z = GRU(hid=1000, return_sequences=True)(X, W_x, W_h, B)
    f = theano.function([X, W_x, W_h, B], Z)
    #theano.printing.pydotprint(f, outfile='gru.png', var_with_name_simple=True)

    for i in range(100):
        o = f(x, w_x, w_h, b)
    
    return o[31]


if __name__ == '__main__':
    #a = GRU_NP()
    b = GRU_MKL()

