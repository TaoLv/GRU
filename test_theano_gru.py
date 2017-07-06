

import time
import theano
import numpy as np

from theano import tensor as T
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh

X = T.ftensor3('X')
W_xh = T.fmatrix('W_xh')
W_xz = T.fmatrix('W_xz')
W_xr = T.fmatrix('W_xr')

W_hh = T.fmatrix('W_hh')
W_hz = T.fmatrix('W_hz')
W_hr = T.fmatrix('W_hr')

B_h = T.fvector('B_h')
B_z = T.fvector('B_z')
B_r = T.fvector('B_r')


hid = T.fmatrix('W_hh')


def step(x_h, x_z, x_r, h_tm1):
    global W_hh, W_hz, W_hr, B_h, B_z, B_r
    z_t = sigmoid(x_z + T.dot(h_tm1, W_hz) + B_z)
    r_t = sigmoid(x_r + T.dot(h_tm1, W_hr) + B_r)
    can_h_t = tanh(x_h + r_t * T.dot(h_tm1, W_hh) + B_h)

    h_t = (1. - z_t) * h_tm1 + z_t * can_h_t
    return h_t

def GRU_theano():
    global X, W_xr, W_xz, W_xh, hid
    X_r = T.dot(X, W_xr)
    X_z = T.dot(X, W_xz)
    X_h = T.dot(X, W_xh)

    fn = lambda x_h, x_z, x_r, h_tm1: step(x_h, x_z, x_r, h_tm1)

    result, updates = theano.scan(fn, sequences=[X_h, X_z, X_r], outputs_info=hid, name='test_theano_gru_scan')

    return result


if __name__ == '__main__':
    o = GRU_theano()

    f = theano.function([X, W_xh, W_xz, W_xr, W_hh, W_hz, W_hr, B_h, B_z, B_r, hid], o)

    # theano.printing.pydotprint(f, outfile='gru_theano.png', var_with_name_simple=True)

    v_x = np.random.rand(32, 80, 620).astype(np.float32)
    v_w_xh = np.random.rand(620, 1000).astype(np.float32) - np.random.rand(620, 1000).astype(np.float32)
    v_w_xz = np.random.rand(620, 1000).astype(np.float32) - np.random.rand(620, 1000).astype(np.float32)
    v_w_xr = np.random.rand(620, 1000).astype(np.float32) - np.random.rand(620, 1000).astype(np.float32)
    v_w_hh = np.random.rand(1000, 1000).astype(np.float32) - np.random.rand(1000, 1000).astype(np.float32)
    v_w_hz = np.random.rand(1000, 1000).astype(np.float32) - np.random.rand(1000, 1000).astype(np.float32)
    v_w_hr = np.random.rand(1000, 1000).astype(np.float32) - np.random.rand(1000, 1000).astype(np.float32)

    v_b_h = np.random.rand(1000).astype(np.float32) - np.random.rand(1000).astype(np.float32)
    v_b_z = np.random.rand(1000).astype(np.float32) - np.random.rand(1000).astype(np.float32)
    v_b_r = np.random.rand(1000).astype(np.float32) - np.random.rand(1000).astype(np.float32)

    v_hid = np.zeros((80, 1000), np.float32)
    
    tic = time.time()
    out = f(v_x, v_w_xh, v_w_xz, v_w_xr, v_w_hh, v_w_hz, v_w_hr, v_b_h, v_b_z, v_b_r, v_hid)

    toc = time.time()
    print('time: %.6f' %(toc - tic))
