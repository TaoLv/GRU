
import time
import theano
import numpy as np

from theano import tensor as T
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh

from mkl_gru import GRU

X = T.dtensor3('X')
W_xh = T.dmatrix('W_xh')
W_xz = T.dmatrix('W_xz')
W_xr = T.dmatrix('W_xr')

W_hh = T.dmatrix('W_hh')
W_hz = T.dmatrix('W_hz')
W_hr = T.dmatrix('W_hr')

B_h = T.dvector('B_h')
B_z = T.dvector('B_z')
B_r = T.dvector('B_r')

hid = T.dmatrix('hid')

np.random.seed(123)


def step(x_h, x_z, x_r, h_tm1):
    global W_hh, W_hz, W_hr, B_h, B_z, B_r
    z_t = sigmoid(x_z + T.dot(h_tm1, W_hz) + B_z)
    r_t = sigmoid(x_r + T.dot(h_tm1, W_hr) + B_r)
    can_h_t = tanh(x_h + r_t * T.dot(h_tm1, W_hh) + B_h)

    h_t = (1. - z_t) * h_tm1 + z_t * can_h_t
    return h_t

def GRU_theano():
    global X, W_xr, W_xz, W_xh, hid, W_hh, Whz, Whr
    X_r = T.dot(X, W_xr)
    X_z = T.dot(X, W_xz)
    X_h = T.dot(X, W_xh)

    fn = lambda x_h, x_z, x_r, h_tm1: step(x_h, x_z, x_r, h_tm1)

    result, updates = theano.scan(fn, sequences=[X_h, X_z, X_r], outputs_info=hid, name='test_theano_gru_scan')

    loss = result.sum()
    gx, gwxh, gwxz, gwxr, gwhh, gwhz, gwhr, ghinit, gbh, gbz, gbr = theano.grad(loss,
                                    [X, W_xh, W_xz, W_xr, W_hh, W_hz, W_hr, hid, B_h, B_z, B_r])

    f = theano.function([X, W_xh, W_xz, W_xr, W_hh, W_hz, W_hr, hid, B_h, B_z, B_r],
                        [gx, gwxh, gwxz, gwxr, gwhh, gwhz, gwhr, ghinit, gbh, gbz, gbr], updates=updates)

    return f


def GRU_MKL():

    X = T.dtensor3('X')
    W_x = T.dmatrix('W_x')
    W_h = T.dmatrix('W_h')
    B = T.dvector('b')
    Hid_init = T.dmatrix('hid_init')

    Z = GRU(hid=1000, return_sequences=True)(X, W_x, W_h, Hid_init, B)

    loss = Z[0].sum()
    gi, gwx, gwh, ghinit, gbias = theano.grad(loss, [X, W_x, W_h, Hid_init, B])
    f = theano.function([X, W_x, W_h, Hid_init, B], [gi, gwx, gwh, ghinit, gbias])
    theano.printing.pydotprint(f, outfile='gru_backward_bias.png', var_with_name_simple=True)

    return f


if __name__ == '__main__':
    f_theano = GRU_theano()
    f_mkl = GRU_MKL()

    v_x = np.random.rand(2, 80, 620).astype(np.float64)
    v_w = np.random.rand(3*620, 1000).astype(np.float64) - np.random.rand(3*620, 1000).astype(np.float64) 
    v_w_xh = v_w[0:620, :]
    v_w_xz = v_w[620:1240, :]
    v_w_xr = v_w[1240:, :]

    v_wh = np.random.rand(3*1000, 1000).astype(np.float64) - np.random.rand(3*1000, 1000).astype(np.float64) 

    v_w_hh = v_wh[0:1000, :]
    v_w_hz = v_wh[1000:2000, :]
    v_w_hr = v_wh[2000:, :]

    v_hid = np.zeros((80, 1000), np.float64)

    v_bias = np.zeros((3000,), np.float64)
    v_b_h = v_bias[0:1000]
    v_b_z = v_bias[1000:2000]
    v_b_r = v_bias[2000:]
    
    out = f_theano(v_x, v_w_xh, v_w_xz, v_w_xr, v_w_hh, v_w_hz, v_w_hr, v_hid, v_b_h, v_b_z, v_b_r)
    #print(out[-1])

    foo = f_mkl(v_x, v_w, v_wh, v_hid, v_bias)
    b1 = foo[-1][0:1000]
    b2 = foo[-1][1000:2000]
    b3 = foo[-1][2000:]

    assert np.allclose(out[-3], b1, atol=1e-4, rtol=1e-3)
    assert np.allclose(out[-2], b2, atol=1e-4, rtol=1e-3)
    assert np.allclose(out[-1], b3, atol=1e-4, rtol=1e-3)

    assert np.allclose(out[-4], foo[-2], atol=1e-4, rtol=1e-3)

