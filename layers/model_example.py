from layers import *
import theano
import theano.tensor as T
import numpy as np

def local_fprop(x, y_truth):

    fc0 = ColumnWiseFullyConnected(input_dim = 1000,
            output_dim = 128,
            nonlinearity = tanhFunc)
    fc1 = ColumnWiseFullyConnected(input_dim = 128,
            output_dim = 256,
            nonlinearity = tanhFunc)
    fc2 = ColumnWiseFullyConnected(input_dim = 512,
            output_dim = 256,
            nonlinearity = tanhFunc)
    rnn0 = RNN(input_dim = 256,
            output_dim = 256)
    softmax = ColumnWiseSoftmax()

    t1 = fc0.fprop(x)
    t2 = fc1.fprop(t1)
    t2_2 = rnn0.fprop(t2)
    t2_final = T.concatenate([t2, t2_2], axis=2)

    t3 = fc2.fprop(t2_final)
    my_out = softmax.fprop(t3)
    #my_cost = softmax.cost(t3, y_truth)

    def _local_cost(y, y_truth):
        tmp = y * y_truth
        tmp = T.log(tmp.sum(axis=2)).mean(axis=2).mean()
        tmp = -tmp
        return tmp

    my_cost = _local_cost(my_out, y_truth)

    my_update = []
    layers = [fc0, fc1, fc2, rnn0, softmax]
    momemtum = 0.9
    for layer in layers:
        for p in layer.get_params():
            lr = layer.learing_rate
            wd = layer.weight_decay
            g = T.grad(cost=my_cost, wrt=p)
            p_delta = theano.,shared(floatX(np.zeros(p.get_value().shape)))
            updates.append(p, p+p_delta*momemtum-lr*(g+wd*p))
            updates.append(p_delta, p_delta*momemtum-lr*(g+wd*p))
    
    return my_out, my_cost, my_update


if __name__ == '__main__':
    data = T.tensor4d()
    label = T.tensor4d()

    output, cost, update = local_fprop(data, label)
    train_func = theano.function(inputs=[data, label], outputs=[output, cost], updates=update)
    test_func = theano.function(inputs=[data, label], outputs=[output, cost])

    #then start train with function
