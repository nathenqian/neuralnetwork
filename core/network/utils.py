import theano
import numpy as np
from theano import tensor as T
from random import randint

def relu(x):
    return T.maximum(x, 0.0)


def randFloat(l, r):
    return 1.0 * randint(0, 10000000) / 10000000.0 * (r - l) + l


def floatX(X):
    return np.asarray(X, dtype = theano.config.floatX)


def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.1))


def init_weights_zero(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0))


class Fclayer():
    def __init__(self, x, y, input, **kargs):
        self.w = init_weights((x, y))
        self.b = theano.shared(floatX(np.random.randn(y) * 0.1))
        self.input = input
        self.output = T.dot(input, self.w) + self.b
        self.params = [self.w, self.b]

        if "weight_decay" in kargs:
            self.weight_decay = kargs["weight_decay"]
        else:
            self.weight_decay = 0.001

        if "learning_rate" in kargs:
            self.learning_rate = kargs["learning_rate"]
        else:
            self.learning_rate = 0.01

        if "momentum" in kargs:
            self.momentum = kargs["momentum"]
        else:
            self.momentum = 0.9

    # def generateParamater(self):
    #     self.weight_decay = randFloat(0.001, 0.00001)
    #     self.learning_rate = randFloat(0.06, 0.001)
    #     self.momentum = randFloat(0.8, 1)

    def calcGred(self, cost):
        params = [self.w, self.b]
        update = []
        gradients = T.grad(cost = cost, wrt = params)
        for p, g in zip(params, gradients):
            param_update = theano.shared(p.get_value()*0., broadcastable = p.broadcastable)
            update.append((p, (1) * p + param_update))
            update.append((param_update, self.momentum * param_update - (self.learning_rate) * (g + p * self.weight_decay)))
        return update

class SoftmaxLayer():
    def __init__(self, input, answer):
        self.input = input
        self.output = T.nnet.softmax(self.input)
        self.answer = answer
        self.cost = T.mean(T.nnet.categorical_crossentropy(self.output, answer))
        self.pred = T.argmax(self.output, axis = 1)
        
class SigmoidLayer():
    def __init__(self, input):
        self.input = input
        self.output = T.nnet.sigmoid(self.input)