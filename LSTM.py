
import numpy as np
import time
from dataset import ptb


def sigmoid(x):
    result = 1 / (1+ np.exp(-x))
    return result


def softmax(x):
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))

    return x



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
        g_mat = np.tanh(middle__mat[:, 2*hiddel_layer_:3*hiddel_layer_])
        output_mat = sigmoid(middle__mat[:, 3*hiddel_layer_: 4 * hiddel_layer_])

        c_next = ( c_prev * forget_mat ) + ( g_mat * input_mat )
        h_next = np.tanh( c_next  ) * output_mat

        self.cache = [ forget_mat, input_mat, g_mat, output_mat, h_next, c_next, h_prev, c_prev, x ]
        return h_next , c_next

    def backward(self, d_h_next, d_c_next):
        weight_x, weight_h, b = self.params
        forget_mat, input_mat, g_mat, output_mat, h_next, c_next, h_prev, c_prev, x = self.cache
        
        tanh_c_next = np.tanh(c_next)
        d_c_next_calc_by_hidden = d_c_next + (d_h_next * output_mat) * (1 - tanh_c_next**2)

        # TODO: d_c_prev は dct * forget かな？
        d_c_prev = d_c_next_calc_by_hidden * forget_mat
        d_g_mat = d_c_next_calc_by_hidden * input_mat
        d_input_mat = d_c_next_calc_by_hidden * g_mat
        d_forget_mat = d_c_next_calc_by_hidden * c_prev
        d_output_mat = d_h_next * np.tanh(c_next)

        d_input_mat = d_input_mat * (1-input_mat) * input_mat
        d_g_mat = d_g_mat * (1 - g_mat ** 2)
        d_forget_mat = d_forget_mat * (1 - forget_mat) * forget_mat
        d_output_mat = d_output_mat * ( 1 - output_mat ) * output_mat

        d_middle_mat = np.hstack((d_forget_mat, d_input_mat, d_g_mat, d_output_mat))

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
        self.params=[weight_xs, weight_hs, bs]
        self.grads=[np.zeros_like(weight_hs), np.zeros_like(weight_xs), np.zeros_like(bs)]
        self.layers = None
        self.h_prev, self.c_prev = None, None
        self.stateful = stateful

    def forward(self, xs ):

        weight_hs, weight_xs, bs = self.params

        N, T, D = xs.shape
        H = weight_hs.shape[0]
        hs=np.empty((N, T, H), dtype='f')

        self.layers=[]


        if self.stateful == False or self.h_prev == None:
            self.h_prev = np.zeros((N,H))

        if self.stateful == False or self.c_prev ==None:
            self.c_prev = np.zeros((N,H))

        for i in range(T):
            layer = LSTM(weight_hs, weight_xs, bs)
            self.h_prev, self.c_prev = layer.forward(xs[:, i, :], self.h_prev, self.c_prev)
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
    def __init__(self, weight):
        self.weight = weight                    #
        self.index= None                        # 索引
        self.params = [weight]
        self.grads = [np.zeros_like(weight)]

    def forward(self, index):
        # if index >= self.weight.shape[0]:
        #     print(" index > row count")
        #     return None
        self.index = index
        return self.weight[index]

    def backward(self, ds):
        dout = np.zeros_like(self.weight)
        dout[...] = 0
        np.add.at(dout, self.index, ds)
        self.grads[0] = dout
        return None


class TimeEmbedding:
    def __init__(self, weight):
        self.weight=weight
        self.grads=[np.zeros_like(self.weight)]
        self.layers = None
        self.params=[weight]


    def forward(self, xs):
        N, T = xs.shape
        D = self.weight.shape[1]
        self.layers = []
        xsData = np.empty((N,T,D), dtype='f')

        for i in range(T):
            model = Embeding(self.weight)
            xsData[:, i, :] = model.forward(xs[:, i])
            self.layers.append(model)
        return xsData

    def backward(self, dout):

        N, T, D = dout.shape
        grad = 0
        for i in reversed(range(T)):
            layer = self.layers[i]
            layer.backward(dout[:, i, :])
            grad += layer.grads[0]

        self.grads[0][...] = grad

        return self.grads


class Affine:

    def __init__(self, weight, b):
        self.params = [weight, b]
        self.grads = [np.zeros_like(weight), np.zeros_like(b)]
        self.cache = None

    def forwad(self, x):
        weight, b = self.params
        out = np.dot(x, weight) + b
        self.x = x
        return out

    def backward(self, ds):
        weight, b = self.params

        dx = np.dot(ds, weight.T)
        dweight = np.dot(self.x.T, ds)

        db = np.sum(ds, axis=0)

        self.grads[0][...] = dweight
        self.grads[1][...] = dx

        return dx


class TimeAffine:

    def __init__(self, weights, bs):
        self.params = [weights, bs]
        self.grads = [np.zeros_like(weights), np.zeros_like(bs)]
        self.cache = None
        self.layers = None

    def forward(self, xs):
        weights, bs = self.params
        N, T, D = xs.shape

        rxs = xs.reshape(N * T, -1)
        self.xs = xs
        out = np.dot(rxs, weights) + bs
        return out.reshape(N, T, -1)

    def backward(self, douts):
        weights, bs = self.params

        N, T, H = douts.shape

        douts = douts.reshape(N * T, -1)
        xs = self.xs
        xs = xs.reshape(N * T, -1)
        dweight = np.dot(xs.T, douts)
        dx = np.dot(douts, weights.T)
        db = np.sum(douts, axis=0)

        self.grads[0][...] = dweight
        self.grads[1][...] = db

        return dx.reshape(*self.xs.shape)


class TimeSoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None
        self.ignore_label = -1

    def forward(self, xs, ts):
        N, T, V = xs.shape

        if ts.ndim == 3:  # 教師ラベルがone-hotベクトルの場合
            ts = ts.argmax(axis=2)

        mask = (ts != self.ignore_label)

        # バッチ分と時系列分をまとめる（reshape）
        xs = xs.reshape(N * T, V)
        ts = ts.reshape(N * T)
        mask = mask.reshape(N * T)

        ys = softmax(xs)
        ls = np.log(ys[np.arange(N * T), ts])
        ls *= mask  # ignore_labelに該当するデータは損失を0にする
        loss = -np.sum(ls)
        loss /= mask.sum()

        self.cache = (ts, ys, mask, (N, T, V))
        return loss

    def backward(self, dout=1):
        ts, ys, mask, (N, T, V) = self.cache

        dx = ys
        dx[np.arange(N * T), ts] -= 1
        dx *= dout
        dx /= mask.sum()
        dx *= mask[:, np.newaxis]  # ignore_labelに該当するデータは勾配を0にする

        dx = dx.reshape((N, T, V))

        return dx


class simpleLSTM:
    def __init__(self, VOCAB_SIZE=1000, WORDVEC_SIZE=50, HIDDED_SIZE=50):
        rn = np.random.randn
        weight_embedding = (rn(VOCAB_SIZE, WORDVEC_SIZE) / 100).astype('f')
        weight_affien = (rn(HIDDED_SIZE, VOCAB_SIZE) / np.sqrt(HIDDED_SIZE)).astype('f')
        weight_LSTM_x = (rn(WORDVEC_SIZE, 4 * HIDDED_SIZE) / np.sqrt(WORDVEC_SIZE)).astype('f')
        weight_LSTM_hidden = (rn(HIDDED_SIZE, 4 * HIDDED_SIZE) / np.sqrt(HIDDED_SIZE)).astype('f')
        weight_LSTM_bias = (rn(4 * HIDDED_SIZE) / np.sqrt(HIDDED_SIZE)).astype('f')

        weight_bias = (rn(VOCAB_SIZE) / np.sqrt(HIDDED_SIZE)).astype('f')

        self.layers = [TimeEmbedding(weight_embedding),
                       Time_LSTM(weight_LSTM_x, weight_LSTM_hidden, weight_LSTM_bias, stateful=True),
                       TimeAffine(weight_affien, weight_bias)]
        self.softWithLoss = TimeSoftmaxWithLoss()
        # すべての重みと勾配をリストにまとめる
        self.params = []
        self.grads =  []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs, ts):

        for layer in self.layers:
            xs = layer.forward(xs)

        
        
        loss = self.softWithLoss.forward(xs, ts)
        return loss

    def backward(self, dout=1):
        dout = self.softWithLoss.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

        return dout

    def reset_state(self):
        self.layers[1].resetstate()


class SGD:
    '''
    確率的勾配降下法（Stochastic Gradient Descent）
    '''

    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]


class trainer:
    def __init__(self, batch_sise=20, time_size=5, VOCAB_SIZE=1000, WORDVEC_SIZE=50, HIDDED_SIZE=50,
                 learning_r=0.01, max_eporch=100):
        self.batch_size = batch_sise
        self.time_size = time_size
        self.vocab_size = VOCAB_SIZE
        self.wordvec_size = WORDVEC_SIZE
        self.hidden_size = HIDDED_SIZE
        self.learning_r = learning_r
        self.max_eporch = max_eporch
        

    def read_data(self, path):
        corpus, word_to_id, id_to_word = ptb.load_data('train')
        corpus_test, _, _ = ptb.load_data('test')
        self.vocab_size = len(word_to_id)
        self.corpus = corpus
        self.word_to_id = word_to_id
        self.id_to_word = id_to_word
        self.xs =self.corpus[:-1] 
        self.ts = self.corpus[1:]
    # def get_xs(self):
    #     return self.corpus[:-1]

    # def get_tx(self):
    #     return self.corpus[1:]

    def get_step_sise(self):
        return self.xs.size // self.time_size

    def get_batch_count(self):
        batch_count = self.xs.size // ( self.batch_size * self.time_size )
        if (self.xs.size %  ( self.batch_size * self.time_size)) != 0:
            batch_count += 1

        return batch_count

    def get_batch_xs_ts(self, index):

        batch_xs = np.zeros((self.time_size, self.batch_size)).astype('i')
        batch_ts = np.zeros((self.time_size, self.batch_size)).astype('i')
        step_size = self.get_step_sise()
        data_size = self.xs.size

        for i in range(self.batch_size):
            for t in range(self.time_size):
                batch_xs[t, i] = self.xs[(i + step_size * t + index * self.batch_size) % data_size]
                batch_ts[t, i] = self.ts[(i + step_size * t + index * self.batch_size) % data_size]

        return batch_xs, batch_ts

    def train(self):
        batch_count=self.get_batch_count()
        optimizer=SGD(self.learning_r)
        total_lose=0
        lose_count = 0
        start_time = time.time()
        for i in range(self.max_eporch):
            for batch_index in range(batch_count):
                lose_count +=1
                batch_xs, batch_ts = self.get_batch_xs_ts(batch_index)
                model=simpleLSTM(VOCAB_SIZE=self.vocab_size)
                total_lose += model.forward(batch_xs, batch_ts)
                model.backward()
                params = model.params
                grads = model.grads
                optimizer.update(params, grads)
               

                if batch_index % 20 == 0 :
                    ppl = np.exp(total_lose / lose_count)
                    print( "ppl %f count %d time %d"  %(ppl, lose_count, time.time()-start_time) )
                    total_lose, lose_count = 0, 0

        

trainer = trainer(batch_sise=40, time_size=10, VOCAB_SIZE=1000, WORDVEC_SIZE=100, HIDDED_SIZE=100,
                    learning_r=10, max_eporch= 100)
trainer.read_data("")
trainer.train()
