import theano
from theano import tensor as T
import theano.tensor.signal.downsample
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np
import scipy.io
import time

srng = RandomStreams(seed=10001)

def identityFunc(x):
    return x

def sigmoidFunc(x):
#    return 1./(1+T.exp(-x))
    return T.nnet.sigmoid(x)

def tanhFunc(x):
    return T.tanh(x)

def reluFunc(x):
    return T.maximum(x, 0.)

def floatX(x):
    return np.asarray(x, dtype=theano.config.floatX)


class LayerBase(object):
    learning_rate = None
    weight_decay = None
    dropout_prob = None

    def __init__(self, **kwargs):
        super(LayerBase, self).__init__(**kwargs)
        self.learning_rate = 0.0005
        self.weight_decay = 0.
        self.dropout_prob = None
        if 'learning_rate' in kwargs:
            self.learning_rate = kwargs['learning_rate']
        if 'weight_decay' in kwargs:
            self.weight_decay = kwargs['weight_decay']
        if 'dropout_prob' in kwargs:
            self.dropout_prob = kwargs['dropout_prob']

    def init_weights(self, shape, init_std=0.01):
        return theano.shared(floatX(np.random.randn(*shape) * init_std))

    def _fprop(self, y_in):
        pass

    def fprop(self, y_in):
        if self.dropout_prob:
            y_in = self._dropout_input(y_in)
        return self._fprop(y_in)

    def get_params(self):
        pass

    def _dropout_input(self, x, p=0.):
        if p > 0:
            retain_prob = 1 - p
            x *= srng.binomial(x.shape, p=retain_prob, 
                                dtype=theano.config.floatX())
            x /= retain_prob
        return x


class FullyConnected(LayerBase):
    "fcLayer: b c h w"

    def __init__(self, input_dim, output_dim, nonlinearity=identityFunc, **kwargs):
        super(FullyConnected, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nonlinearity = nonlinearity

        # init weights
        self.W = self.init_weights((input_dim, output_dim))
        self.b = self.init_weights((output_dim,))

        super(FCLayer, self).__init__(**kwargs)

    def _fprop(self, y_in):
        x = T.flatten(y_in, outdim = 2)
        res = T.dot(x, self.W) + self.b.dimshuffle('x', 0)
        res = self.nonlinearity(res)
        return res

    def get_params(self):
        return [self.W, self.b]

class Conv2D(LayerBase):
    "conv: b c0 h w -> b c1 h w"
    def __init__(self, input_dim, output_dim, kernel_shape=(1,1),
            kernel_stride=(1,1), border_mode='full', nonlinearity=identityFunc,
            **kwargs):
        super(Conv2D, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_shape = kernel_shape
        self.kernel_stride = kernel_stride
        self.border_mode = border_mode
        self.nonlinearity = nonlinearity

        # init weights        
        self.W = self.init_weights((output_dim, input_dim, kernel_shape[0], kernel_shape[1]), init_std=0.1)
        self.b = self.init_weights((output_dim,), init_std=0.1)

    def _fprop(self, y_in):
        x = y_in
        res = T.nnet.conv2d(x, self.W, border_mode = self.border_mode, subsample = self.kernel_stride) 
        res = res + self.b.dimshuffle('x', 0, 'x', 'x')
        res = self.nonlinearity(res)
        return res

    def get_params(self):
        return [self.W, self.b]

class Maxpooling(LayerBase):
    "maxpooling: b c h0 w0 -> b c h1 w1"
    def __init__(self, kernel_shape=(1,1), kernel_stride=None, padding=(0, 0), **kwargs):
        super(Maxpooling, self).__init__(**kwargs)
        self.kernel_shape = kernel_shape
        self.kernel_stride = kernel_shape
        self.padding = padding

    def _fprop(self, y_in):
        x = y_in
        res = T.signal.downsample.max_pool_2d(
                    input = x,
                    ds = self.kernel_shape,
                    ignore_border = False,
                    st = self.kernel_stride,
                    padding = self.padding)
        return res

    def get_params(self):
        return []
        
class Softmax(LayerBase):
    "softmax: b c -> b c"
    def __init__(self, **kwargs):
        super(Softmax, self).__init__(**kwargs)

    def fprop(self, y_in):
        x = y_in - y_in.max(axis=1, keepdims=True)
        res = x/T.exp(x).sum(axis=1, keepdims=True)
        return res

    def cost(self, y_in, y_truth):
        x = y_in - y_in.max(axis=1, keepdims=True)
        log_prob = x - T.log(T.exp(x).sum(axis=1, keepdims=True))
        res = -(y_truth * log_prob).sum(axis=1).mean()
        return res

    def get_params(self):
        return []

class EuclideanLoss(LayerBase):
    "euclidean loss: b c  -> b c"
    def __init__(self, **kwargs):
        super(EuclideanLoss, self).__init__(**kwargs)

    def _fprop(self, y_in):
        return y_in

    def cost(self, y_in, y_truth):
        res = T.sqrt(((y_in-y_truth)**2+1e-8).sum(axis=1)).mean()
        return res

    def get_params(self):
        return []

class LSTM(LayerBase):
    """
    i_t = sigmoid(W_i * X + H_i * h_{t-1} + V_i * c_{t-1} + b_i)
    f_t = sigmoid(W_f * X + H_f * h_{t-1} + V_f * c_{t-1} + b_f)
    o_t = sigmoid(W_o * X + H_o * h_{t-1} + V_o * c_t + b_o)

    c_t = f_t .* c_{t-1} + i_t .* tanh(W_c * X + H_c * h_{t-1} + b_c)
    h_t = o_t .* tanh(c_t)
    """

    W_i = None
    W_f = None
    W_o = None
    W_c = None
    H_i = None
    H_f = None
    H_o = None
    H_c = None
    b_i = None
    b_f = None
    b_o = None
    b_c = None

    def __init__(self, input_dim, output_dim, **kwargs):
        super(LSTM, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.W_i = self.init_weights((input_dim, output_dim))
        self.W_f = self.init_weights((input_dim, output_dim))
        self.W_o = self.init_weights((input_dim, output_dim))
        self.W_c = self.init_weights((input_dim, output_dim))
        self.H_i = self.init_weights((output_dim, output_dim))
        self.H_f = self.init_weights((output_dim, output_dim))
        self.H_o = self.init_weights((output_dim, output_dim))
        self.H_c = self.init_weights((output_dim, output_dim))
        self.b_i = self.init_weights((output_dim,))
        self.b_f = self.init_weights((output_dim,))
        self.b_o = self.init_weights((output_dim,))
        self.b_c = self.init_weights((output_dim,))

    def _fprop(self, y_in):
        outputs, updates = theano.scan(fn = self._one_step,
                                       sequences = T.arange(y_in.shape[-1]),
                                       outputs_info = self._get_init_state(x),
                                       non_sequences = [y_in],
                                       truncate_gradient = -1)
        return outputs[0]

    def _get_input(self, x, t):
        return x[:,:,:,t:t+1]

    def _one_step(self, t, h_t1, c_t1, x):
        tmp_x = self._get_input(x, t)
        def _dot(w, x):
            return T.tensordot(w, x, axes=[[2], [0]])
        i_t = sigmoidFunc(_dot(self.W_i, tmp_x)+_dot(self.H_i, h_t1)+self.b_i.dimshuffle('x','x','x',0))
        f_t = sigmoidFunc(_dot(self.W_f, tmp_x)+_dot(self.H_f, h_t1)+self.b_f.dimshuffle('x','x','x',0))
        o_t = sigmoidFunc(_dot(self.W_o, tmp_x)+_dot(self.H_o, h_t1)+self.b_o.dimshuffle('x','x','x',0))

        c_t = f_t * c_t1 + i_t * tanhFunc(_dot(self.W_c, tmp_x)+_dot(self.H_c, h_t1)+self.b_c.dimshuffle('x','x','x',0))
        h_t = o_t * tanhFunc(c_t)

        c_t = c_t.dimshuffle(0,1,3,2)
        h_t = h_t.dimshuffle(0,1,3,2)
        return h_t, c_t

    def _get_init_state(self, x):
        h_0 = T.zeros((x.shape[0], x.shape[1], self.output_dim, 1), dtype=theano.config.floatX())
        c_0 = T.zeros((x.shape[0], x.shape[1], self.output_dim, 1), dtype=theano.config.floatX())
        return h_0, c_0

    def get_params(self):
        return [self.W_i, self.W_f, self.W_o, self.W_c,
                self.H_i, self.H_f, self.H_o, self.H_c,
                self.b_i, self.b_f, self.b_o, self.b_c]

class RNN(LayerBase):
    """
    h_t = tanh(W * X + H * h_{t-1} + b)
    """
    W = None
    H = None
    b = None

    def __init__(self, input_dim, output_dim, **kwargs):
        super(RNN, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.W = self.init_weights((input_dim, output_dim))
        self.H = self.init_weights((output_dim, output_dim))
        self.b = self.init_weights((output_dim,))

    def _fprop(self, y_in):
        outputs, updates = theano.scan(fn = self._one_step,
                                       sequences = T.arange(y_in.shape[-1]),
                                       outputs_info = self._get_init_state(x),
                                       non_sequences = [y_in],
                                       truncate_gradient = -1)
        return outputs

    def _get_input(self, x, t):
        return x[:,:,:,t:t+1]

    def _one_step(self, t, h_t1, x):
        tmp_x = self._get_input(x, t)
        def _dot(w, x):
            return T.tensordot(w, x, axes=[[2], [0]])
        h_t = tanhFunc(_dot(self.W, tmp_x)+_dot(self.H, h_t1)+self.b.dimshuffle('x','x','x',0))
        h_t = h_t.dimshuffle(0,1,3,2)
        return h_t

    def _get_init_state(self, x):
        h_0 = T.zeros((x.shape[0], x.shape[1], self.output_dim, 1), dtype=theano.config.floatX())
        return h_0

    def get_params(self):
        return [self.W, self.H, self.b]

class ColumnWiseFullyConnected(LayerBase):
    """
    (b,c,h1,w) => (b,c,h2,w)
    """

    W = None
    b = None
    def __init__(self, input_dim, output_dim, nonlinearity=identityFunc, **kwargs):
        super(ColumnWiseFullyConnected, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nonlinearity = nonlinearity

        self.W = self.init_weights((input_dim, output_dim))
        self.b = self.init_weights((output_dim,))

    def _fprop(self, y_in):
        x = y_in.dimshuffle(0,1,3,2)
        z = T.tensordot(x, self.W)
        z = self.nonlinearity(z.dimshuffle(0,1,3,2) + b.dimshuffle('x', 'x', 0, 'x'))
        return z

    def get_params(self):
        return [self.W, self.b]

class ColumnWiseSoftmax(LayerBase):
    """
    (b,c,h,w) => (b,c,h,w)
    constant : temperature = 1.
    """

    def __init__(self):
        super(ColumnWiseSoftmax, self).__init__(**kwargs)

    def fprop(self, y_in):
        x = y_in - y_in.max(axis=2, keepdims=True)
        res = x/T.exp(x).sum(axis=2, keepdims=True)
        return res

    def cost(self, y_in, y_truth):
        x = y_in - y_in.max(axis=2, keepdims=True)
        log_prob = x - T.log(T.exp(x).sum(axis=2, keepdims=True))
        res = -(y_truth * log_prob).sum(axis=[2,3]).mean()
        return res

    def get_params(self):
        return []
