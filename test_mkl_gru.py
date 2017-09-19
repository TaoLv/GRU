import time
import theano
from theano import tensor as T
from mkl_gru import GRU
import numpy as np

np.random.seed(12345)

def GRU_MKL():
    X = T.dtensor3('X')
    W_x = T.dmatrix('W_x')
    W_h = T.dmatrix('W_h')
    B = T.dvector('b')
    Hid = T.dmatrix('Hid_init')

    Z = GRU(hid=1000, return_sequences=True, max_len=100)(X, W_x, W_h, Hid)
    f = theano.function([X, W_x, W_h, Hid], Z)

    return f

def GRU_MKL_NO_MAX_LEN():
    X = T.dtensor3('X')
    W_x = T.dmatrix('W_x')
    W_h = T.dmatrix('W_h')
    B = T.dvector('b')
    Hid = T.dmatrix('Hid_init')

    Z = GRU(hid=1000, return_sequences=True)(X, W_x, W_h, Hid)
    f = theano.function([X, W_x, W_h, Hid], Z)

    return f

def test_GRU():
    f_mkl = GRU_MKL()
    # f_mkl = GRU_MKL_NO_MAX_LEN()

    v_x = np.random.rand(10, 80, 620).astype(np.float64)
    v_w = np.random.rand(3*620, 1000).astype(np.float64) - np.random.rand(3*620, 1000).astype(np.float64) 
    v_wh = np.random.rand(3*1000, 1000).astype(np.float64) - np.random.rand(3*1000, 1000).astype(np.float64) 
    v_hid = np.zeros((80, 1000), np.float64)

    foo, zt, rt, hcan, hht = f_mkl(v_x, v_w, v_wh, v_hid)

    v_x = np.random.rand(20, 80, 620).astype(np.float64)
    foo, zt, rt, hcan, hht = f_mkl(v_x, v_w, v_wh, v_hid)

    v_x = np.random.rand(30, 80, 620).astype(np.float64)
    foo, zt, rt, hcan, hht = f_mkl(v_x, v_w, v_wh, v_hid)

    v_x = np.random.rand(10, 80, 620).astype(np.float64)
    foo, zt, rt, hcan, hht = f_mkl(v_x, v_w, v_wh, v_hid)


if __name__ == '__main__':
    test_GRU()
