import numpy as np


def sigmoid(x):
    result = 1 / (1+ np.exp(-x))
    return result




class LSTM:
    def __init__(self, weight_x, weight_h, b):

        self.params=[weight_x, weight_h, b]
        self.grads=[np.zeros_like(weight_x), np.zeros_like(weight_h), np.zeros_like(b)]
        self.cache=None

    def forward(self, x, h_prev, c_prev ):
        weight_x, weight_h, b =self.params
        input_layer_, hiddel_layer_=h_prev.shape
        middle__mat=np.dot(x,weight_x) + np.dot(h_prev,weight_h) + b

        # slice
        forget_mat= sigmoid(middle__mat[:,0:hiddel_layer_])
        input_mat = sigmoid(middle__mat[:, hiddel_layer_:2*hiddel_layer_])
        g_mat = np.tanh(middle__mat[:, 2*hiddel_layer_, 3*hiddel_layer_])
        output_mat = sigmoid(middle__mat[:, 3*hiddel_layer_, 4 * hiddel_layer_])

        c_next = ( c_prev * forget_mat ) + ( g_mat * input_mat )
        h_next = np.tanh( c_next  ) * output_mat

        self.cache = [ forget_mat, input_mat, g_mat, output_mat, h_next, c_next, h_prev, c_prev, x ]
        return h_next , c_next

    def backward(self, d_h_next, d_c_next):
        weight_x, weight_h, b = self.params
        forget_mat, input_mat, g_mat, output_mat, h_next, c_next, h_prev, c_prev, x = self.cache
        
        tanh_c_next = np.tanh(c_next)
        d_c_next_calc_by_hidden = d_c_next + (d_h_next * output_mat) * (1 - tanh_c_next**2)

        d_g_mat = d_c_next_calc_by_hidden * input_mat
        d_input_mat = d_c_next_calc_by_hidden * g_mat
        d_forget_mat = d_c_next_calc_by_hidden * c_prev
        d_output_mat = d_h_next * np.tanh(c_next)

        d_input_mat = d_input_mat * (1-input_mat) * input_mat
        d_g_mat = d_g_mat * (1 - g_mat ** 2)
        d_forget_mat = d_forget_mat * (1 - forget_mat) * forget_mat
        d_output_mat = d_output_mat * ( 1 - output_mat ) * output_mat

        d_middle_mat = np.hstack(d_forget_mat, d_input_mat, d_g_mat, d_output_mat)

        d_weight_h = np.dot(h_prev.T, d_middle_mat)
        d_weight_x = np.dot(x.T, d_middle_mat)
        d_b = d_middle_mat.sum(axis=0)

        self.grads[0][...] = d_weight_x
        self.grads[1][...] = d_weight_h
        self.grads[2][...] = d_b

        dx = np.dot(d_middle_mat, weight_x.T)
        d_h_prev = np.dot( d_middle_mat, weight_h.T)

        return  dx, d_h_prev, d_c_prev


class Time_LSTM:
    def __init__(self, weight_xs, weight_hs, bs, stateful = False):
        self.params=[weight_hs, weight_xs, bs]
        self.grads=[np.zeros_like(weight_hs), np.zeros_like(weight_xs), np.zeros_like(bs)]
        self.layers = None
        self.h_prev, self.c_prev = None, None
        self.stateful = stateful

    def forward(self, xs ):

        weight_hs, weight_xs, bs = self.params

        N, T, D = xs.shape
        H = weight_hs.shape[1]
        hs=np.empty((N, T, H), dtype='f')

        self.layers=[]

        if self.stateful == False or self.h_prev == None:
            self.h_prev = np.zeros((N,H))

        if self.stateful == False or self.c_prev ==None:
            self.c_prev = np.zeros((N,H))

        for i in range(T):
            layer = LSTM(self.params)
            self.h_prev , self.c_prev = layer.forward(xs[:, i, :], h_prev, c_prev )
            self.layers.append(layer)
            hs[:, i, :] = self.h_prev
        return hs

    def backward(self, dhs ):

        weight_hs, weight_xs, bs = self.params 
        N, T, H = dhs.shape
        D = weight_xs.shape[0]
        dxs = np.empty((N, T, D), dtype='f')
        dh, dc = 0, 0
        grads = [0, 0, 0]

        for i in reversed(range(T)):
            layer = self.layers[i]
            dx, dh, dc = layer.backward(dhs[:, i, :]+dh, dc)
            dxs[:, i, :] = dx
            for g, grad in enumerate(layer.grads):
                grads[g] += grad
        
        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh

        return dxs


    def resetstate(self):
        self.h_prev = None
        self.c_prev = None

    def setstate(self, cell, hidden):
        self.h_prev = hidden
        self.c_prev = cell

class Embeding:
    def __init__(self, weight, grads):
        self.weight = weight                    #
        self.grads = grads   # 导数
        self.index= None                        # 索引

    def forward(self, index):
        # if index >= self.weight.shape[0]:
        #     print(" index > row count")
        #     return None
        self.index = index
        return self.weight[index]

    def backward(self, ds):
        np.add.at(self.grads, self.index, ds)
        return None


class TimeEmbedding:
    def __init__(self, weight):
        self.weight=weight
        self.grads=np.zeros_like(self.weight)
        self.model = Embeding(self.weight, self.grads)

    def forward(self, xs):
        N, T = xs.shape
        D = self.weight.shape[1]

        xsData = np.empty((N,T,D), dtype='f')

        for i in range(T):
            self.model.forward(xs)

        return None

    def backward(self, dout):
        return None



