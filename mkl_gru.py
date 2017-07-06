
import theano
from theano import tensor, gof
from theano.tensor.blas import ldflags


class GRU(gof.Op):
    __props__ = ('hid', 'step', 'dim', 'return_sequences')

    def __init__(self, hid, step=None, dim=None, return_sequences=False):
        self.hid = hid
        self.step = step
        self.dim = dim
        self.return_sequences = return_sequences
        super(GRU, self).__init__()

    def make_node(self, *inputs):

        if len(inputs) is 3:
            inp = list(map(tensor.as_tensor_variable, inputs))
        elif len(inputs) is 4:
            inp = list(map(tensor.as_tensor_variable, inputs))
        else:
            raise ValueError('GRU: number of parameter is wrong.')

        assert inp[0].ndim is 3
        assert inp[1].ndim is 2
        assert inp[2].ndim is 2

        if self.return_sequences:
            bcast = [inp[0].type.broadcastable[1], inp[0].type.broadcastable[2]]
            out = [tensor.tensor(dtype=inp[0].type.dtype, broadcastable=bcast)]
        else:
            out = [inp[0].type()]

        return gof.Apply(self, inp, out)

    def c_headers(self):
        headers = ['<mkl.h>', '<omp.h>']
        return headers

    def c_libraries(self):
        return ldflags()

    def c_support_code_struct(self, node, name):
        if node.inputs[0].type.dtype == 'float32':
            dtype = 'float'
        elif node.inputs[0].type.dtype == 'float64':
            dtype = 'double'
        else:
            raise TypeError('GRU: dtype %s is not supported.'
                            % (node.inputs[0].type.dtype))

        ccode = """
        %(dtype)s** A;
        %(dtype)s** B;
        %(dtype)s** C;

        MKL_INT lda[1];
        MKL_INT ldb[1];
        MKL_INT ldc[1];

        MKL_INT m[1];
        MKL_INT n[1];
        MKL_INT k[1];

        CBLAS_TRANSPOSE transA[1];
        CBLAS_TRANSPOSE transB[1];

        %(dtype)s alpha[1];
        %(dtype)s beta[1];
        MKL_INT size_per_grp[1];

        size_t time_step;
        size_t batch_size;
        size_t embed_dims;

        %(dtype)s* temp;
        %(dtype)s* x_hzr;

        """ % locals()
        return ccode

    def c_init_code_struct(self, node, name, sub):

        ccode = """
        A = NULL;
        B = NULL;
        C = NULL;

        lda[0] = 0;
        ldb[0] = 0;
        ldc[0] = 0;

        m[0] = 0;
        n[0] = 0;
        k[0] = 0;

        alpha[0] = 1.0;
        beta[0] = 1.0;

        transA[0] = CblasNoTrans;
        transB[0] = CblasNoTrans;
        size_per_grp[0] = 1;

        time_step = 0;
        batch_size = 0;
        embed_dims = 0;

        temp = NULL;
        x_hzr = NULL;

        """ % locals()
        return ccode

    def c_cleanup_code_struct(self, node, name):
        ccode = """
        if (A) {
            mkl_free (A);
            A = NULL;
        }

        if (B) {
            mkl_free (B);
            B = NULL;
        }

        if (C) {
            mkl_free (C);
            C = NULL;
        }

        if (temp) {
            mkl_free(temp);
        }

        if (x_hzr) {
            mkl_free(x_hzr);
        }

        """
        return ccode

    def c_code(self, node, name, inputs, outputs, sub):
        if len(inputs) is 3:
            with_bais = 0
            X, W_x, W_h = inputs
        elif len(inputs) is 4:
            with_bias = 1
            X, W_x, W_h, b = inputs
        else:
            raise TypeError('GRU: too much arguments')

        z, = outputs
        hid = self.hid
        if self.return_sequences:
            return_sequences = 1
        else:
            return_sequences = 0

        if node.inputs[0].type.dtype == 'float32':
            dtype = 's'
            d = 'float'
        elif node.inputs[0].type.dtype == 'float64':
            dtype = 'd'
            d = 'double'
        else:
            raise TypeError('GRU: dtype %s is not supported.'
                            % (node.inputs[0].type.dtype))

        ccode = """
            time_step = PyArray_DIMS(%(X)s)[0];
            batch_size = PyArray_DIMS(%(X)s)[1];
            embed_dims = PyArray_DIMS(%(X)s)[2];

            npy_intp dims[3] = {0, 0, 0};
            %(d)s* x_ptr = NULL;
            %(d)s* w_x_ptr = NULL;
            %(d)s* w_h_ptr = NULL;

            vmlSetMode(vmlGetMode() & 0xFFFFFFF0 | VML_HA);
           
            if (A == NULL) {
                A = (%(d)s**)mkl_malloc(3 * time_step * sizeof (%(d)s*), 64);
            }

            if (B == NULL) {
                B = (%(d)s**)mkl_malloc(3 * time_step * sizeof (%(d)s*), 64);
            }

            if (C == NULL) {
                C = (%(d)s**)mkl_malloc(3 * time_step * sizeof (%(d)s*), 64);
            }

            PyArrayObject* x_src = NULL;
            if (!PyArray_IS_C_CONTIGUOUS(%(X)s)) {
                printf(\"Warning: Need convert X to C-Contiguous\\n\");
                x_src = (PyArrayObject*)PyArray_ContiguousFromAny((PyObject*)%(X)s,
                                            PyArray_TYPE(%(X)s),
                                            PyArray_NDIM(%(X)s),
                                            PyArray_NDIM(%(X)s));
                if (!x_src) {
                    PyErr_SetString(PyExc_RuntimeError, \"GRU: fail to cast X to contiguous array\");
                    goto gru_fail;
                }
                x_ptr = (%(d)s*) PyArray_DATA(x_src);
            } else {
                x_ptr = (%(d)s*) PyArray_DATA(%(X)s);
            }

            //// step 1. Dot(X, (W_x))
            if (embed_dims * 3 != PyArray_DIMS(%(W_x)s)[0]) {
                PyErr_SetString(PyExc_RuntimeError, \"GRU: X * W_x size error\");
                goto gru_fail;
            }

            PyArrayObject* w_x_src = NULL;
            if (!PyArray_IS_C_CONTIGUOUS(%(W_x)s)) {
                printf(\"Warning: Need convert W_x to C-Contiguous\\n\");
                w_x_src = (PyArrayObject*)PyArray_ContiguousFromAny((PyObject*)%(W_x)s,
                                            PyArray_TYPE(%(W_x)s),
                                            PyArray_NDIM(%(W_x)s),
                                            PyArray_NDIM(%(W_x)s));
                if (!w_x_src) {
                    PyErr_SetString(PyExc_RuntimeError, \"GRU: fail to cast W_x to contiguous array\");
                    goto gru_fail;
                }
                w_x_ptr = (%(d)s*) PyArray_DATA(w_x_src);
            } else {
                w_x_ptr = (%(d)s*) PyArray_DATA(%(W_x)s);
            }

            PyArrayObject* w_h_src = NULL;
            if (!PyArray_IS_C_CONTIGUOUS(%(W_h)s)) {
                printf(\"Warning: Need convert W_h to C-Contiguous\\n\");
                w_h_src = (PyArrayObject*)PyArray_ContiguousFromAny((PyObject*)%(W_h)s,
                                            PyArray_TYPE(%(W_h)s),
                                            PyArray_NDIM(%(W_h)s),
                                            PyArray_NDIM(%(W_h)s));
                if (!w_h_src) {
                    PyErr_SetString(PyExc_RuntimeError, \"GRU: fail to cast W_h to contiguous array\");
                    goto gru_fail;
                }
                w_h_ptr = (%(d)s*) PyArray_DATA(w_h_src);
            } else {
                w_h_ptr = (%(d)s*) PyArray_DATA(%(W_h)s);
            }

            if (x_hzr == NULL) {
                x_hzr = (%(d)s*)mkl_malloc(time_step * 3 * batch_size * %(hid)s * sizeof (%(d)s), 64);
            }

            if (%(with_bias)s) {
                #pragma omp parallel for
                for (int i = 0; i < time_step; i++) {
                    for (int j = 0; j < batch_size; j++) {
                        size_t offset0 = %(hid)s * j + %(hid)s * batch_size * i;
                        size_t offset1 = %(hid)s * j + %(hid)s * batch_size * (i + time_step);
                        size_t offset2 = %(hid)s * j + %(hid)s * batch_size * (i + 2 * time_step);

                        memcpy((void*)(x_hzr + offset0), (void*)PyArray_DATA(%(b)s), %(hid)s * sizeof (%(d)s));
                        memcpy((void*)(x_hzr + offset1), (void*)PyArray_DATA(%(b)s) + %(hid)s *  sizeof (%(d)s), %(hid)s * sizeof (%(d)s));
                        memcpy((void*)(x_hzr + offset2), (void*)PyArray_DATA(%(b)s) + 2 * %(hid)s * sizeof (%(d)s), %(hid)s * sizeof (%(d)s));
                    }
                }
            } else {
                memset((char*)x_hzr, 0, time_step * 3 * batch_size * %(hid)s * sizeof (%(d)s));
            }

            m[0] = batch_size;
            k[0] = embed_dims;
            n[0] = %(hid)s;
            #pragma omp parallel for
            for (int i = 0; i < time_step; i++) {
                A[i] = x_ptr + i * m[0] * k[0];
                A[i + time_step] = A[i];
                A[i + 2 * time_step] = A[i];

                B[i] = w_x_ptr;
                B[i + time_step] = w_x_ptr + embed_dims * %(hid)s;
                B[i + 2 * time_step] = w_x_ptr + 2 * embed_dims * %(hid)s;

                C[i] = x_hzr + i * m[0] * n[0];
                C[i + time_step] = x_hzr + (i + time_step) * m[0] * n[0];
                C[i + 2 * time_step] = x_hzr + (i + 2 * time_step) * m[0] * n[0];
            }

            size_per_grp[0] = 3 * time_step;
            cblas_%(dtype)sgemm_batch(CblasRowMajor, transA, transB, m, n, k,
                                      alpha, A, k, B, n, beta, C, n, 1, size_per_grp);

            //// step 2. construct output
            if (%(z)s == NULL) {
                if (%(return_sequences)s) {
                    dims[0] = time_step;
                    dims[1] = batch_size;
                    dims[2] = %(hid)s;
                    %(z)s = (PyArrayObject*) PyArray_ZEROS(3, dims, PyArray_TYPE(%(X)s), 0);
                } else {
                    dims[0] = batch_size;
                    dims[1] = %(hid)s;
                    %(z)s = (PyArrayObject*) PyArray_ZEROS(2, dims, PyArray_TYPE(%(X)s), 0);
                }
            }

            if (%(z)s == NULL) {
                PyErr_SetString(PyExc_RuntimeError, \"GRU: create output array failed\");
                goto gru_fail;
            }

            //// step 3: step on time_step
            // loop on step
            A[0] = (%(d)s*)PyArray_DATA(%(z)s);
            A[1] = A[0];
            A[2] = A[0];
            B[0] = (%(d)s*)PyArray_DATA(%(W_h)s) + %(hid)s * %(hid)s;     // w_hz
            B[1] = (%(d)s*)PyArray_DATA(%(W_h)s) + 2 * %(hid)s * %(hid)s; // w_hr
            B[2] = (%(d)s*)PyArray_DATA(%(W_h)s);                         // w_hh

            if (NULL == temp) {
                temp = (%(d)s*)mkl_malloc(batch_size * %(hid)s * sizeof (%(d)s), 64);
            }
            memset((void*)temp, 0, batch_size * %(hid)s * sizeof (%(d)s));

            m[0] = batch_size;
            k[0] = %(hid)s;
            n[0] = %(hid)s;
            size_per_grp[0] = 2;
            for (int i = 0; i < time_step; i++) {
                //z_t, r_t
                C[0] = x_hzr + (i + time_step) * m[0] * n[0];
                C[1] = x_hzr + (i + 2 * time_step) * m[0] * n[0];
                C[2] = temp;
                
                // do below two function with batch-gemm first, then sigmoid respectively
                // z_t = K.sigmoid(x_z + K.dot(h_tm1, self.W_hz) + self.b_z)
                // r_t = K.sigmoid(x_r + K.dot(h_tm1, self.W_hr) + self.b_r)

                cblas_%(dtype)sgemm_batch(CblasRowMajor, transA, transB, m, n, k,
                                          alpha, A, k, B, n, beta, C, n, 1, size_per_grp);

                // sigmoid(C[0]), sigmoid(C[1])
                int t = 0;
                /*
                #pragma omp parallel for
                for (t = 0; t < m[0] * n[0]; t++) {
                    C[0][t] = -C[0][t];
                    C[1][t] = -C[1][t];
                }
                */

                v%(dtype)sExp(m[0] * n[0], C[0], C[0]);
                v%(dtype)sExp(m[0] * n[0], C[1], C[1]);
                #pragma omp parallel for
                for (t = 0; t < m[0] * n[0]; t++) {
                    // C[0][t] = (%(d)s)((double)1.0 / ((double)1.0 + (double)(C[0][t])));
                    // C[1][t] = (%(d)s)((double)1.0 / ((double)1.0 + (double)(C[1][t])));
                    C[0][t] = (%(d)s)((double)(C[0][t]) / ((double)1.0 + (double)(C[0][t])));
                    C[1][t] = (%(d)s)((double)(C[1][t]) / ((double)1.0 + (double)(C[1][t])));
                }

                // GEMM -> Mul -> Add -> tanh, can_h_t is stored in temp
                // can_h_t = K.tanh(x_h + r_t * K.dot(h_tm1, self.W_hh) + self.b_h)
                cblas_%(dtype)sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m[0], n[0], k[0],
                                    1.0, A[2], k[0], B[2], n[0], 0.0, C[2], n[0]);

                v%(dtype)sMul(batch_size * %(hid)s, C[1], temp, temp);
                v%(dtype)sAdd(batch_size * %(hid)s, x_hzr + i * m[0] * n[0], temp, temp);

                // tanh(temp)
                v%(dtype)sTanh(m[0] * n[0], temp, temp);

                // h_t = (1. - z_t) * h_tm1 + z_t * can_h_t

                #pragma omp parallel for num_threads(16)
                for (int j = 0; j < batch_size * %(hid)s; j++) {
                    A[0][j] = (%(d)s)( ((double)1.0 - (double)(C[0][j])) * (double)(A[0][j]) + (double)(C[0][j]) * (double)(temp[j]));
                }

                // set output data
                if (%(return_sequences)s && (i < time_step - 1)) {
                    %(d)s* ptr = (%(d)s*)PyArray_DATA(%(z)s) + (i + 1) * batch_size * %(hid)s;
                    memcpy((char*)ptr,
                           (char*)(A[0]),
                           batch_size * %(hid)s * sizeof (%(d)s));

                    A[0] = ptr;
                    A[1] = A[0];
                    A[2] = A[0];
                }

                printf(\"%%.9f, %%.9f, %%.9f\\n\", A[0][0], A[0][1000], A[0][batch_size * %(hid)s -1]);
            }

            gru_fail:
            Py_XDECREF(x_src);
            Py_XDECREF(w_x_src);
            Py_XDECREF(w_h_src);
        """ % locals()
        return ccode

    def grad(self, inp, grads):
        raise NotImplemented('GRU: grad()')

    def c_code_cache_version(self):
        return (1, 0, 0)
