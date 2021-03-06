import sys
sys.path.append("/Users/nathenqian/Documents/code/nueral/layers")
from layers import *
import theano
import theano.tensor as T
import numpy as np
from json import loads, dumps

import os
sys.path.append("/Users/nathenqian/Documents/code/nueral/core")
from data.data import PoemDataGenerator
from time import localtime

def printTime():
    t = localtime()
    print "time is %d:%d:%d"%(t[3], t[4], t[5])

word_dictionary, data_generator = (None, None)

def local_fprop(x, y_truth): # x = sentences  y_truth is [[][]]
    global word_dictionary

    # fc0 = ColumnWiseFullyConnected(
    #         input_dim=len(word_dictionary),
    #         output_dim=256,
    #         nonlinearity=sigmoidFunc,
    #         dropout_prob=0.2,
    #         name = "fc0"
    #     )

    rnn0 = RNN(
            # input_dim=256,
            # output_dim=256,
            input_dim=len(word_dictionary),
            output_dim=256,
            # nonlinearity=sigmoidFunc,
            name = "rnn0"
            )

    fc1 = ColumnWiseFullyConnected(
        input_dim=256,
        output_dim=len(word_dictionary),
        nonlinearity=identityFunc,
        dropout_prob=0.1,
        name = "fc1"
        )

    softmax = ColumnWiseSoftmax(
        name = "softmax"
    )

    # embeded1_out = fc0.fprop(x)
    # rnn0_out = rnn0.fprop(embeded1_out)
    rnn0_out = rnn0.fprop(x)
    softmax_in = fc1.fprop(rnn0_out)
    my_out = softmax.fprop(softmax_in)
    # my_out = softmax.fprop(rnn0_out)
    # my_cost = softmax.cost(t3, y_truth)

    def _local_cost(y, y_truth):
        tmp = y * y_truth
        # tmp = T.log(tmp.sum(axis=2)).mean(axis=2).mean()
        tmp = T.log(tmp.max(axis=2)).sum(axis=2).mean()
        tmp = -tmp
        # tmp = T.sqrt(((y - y_truth)**2+1e-8).sum(axis=2)).sum().sum()
        return tmp

    my_cost = _local_cost(my_out, y_truth)

    my_update = []
    # layers = [fc0, fc1, rnn0, softmax]
    layers = [rnn0, softmax, fc1]

    momemtum = 0.9
    for layer in layers:
        for p in layer.get_params():
            lr = layer.learning_rate
            wd = layer.weight_decay
            g = T.grad(cost=my_cost, wrt=p)
            p_delta = theano.shared(floatX(np.zeros(p.get_value().shape)))
            my_update.append((p, p-lr*(g+wd*p)))
            # my_update.append((p, p+p_delta*momemtum-lr*(g+wd*p)))
            # my_update.append((p_delta, p_delta*momemtum-lr*(g+wd*p)))

    return my_out, my_cost, my_update, layers


def readDataFile(base_dir):
    dict_path = "dictionary.data2"
    train_data_path = "train.data2"

    with open(os.path.join(base_dir, dict_path), "r") as f:
        word_dictionary = loads(f.read())
    data_generator = PoemDataGenerator(os.path.join(base_dir, train_data_path), word_dictionary)
    print "data_size is " + str(len(data_generator.data))
    return word_dictionary, data_generator


if __name__ == '__main__':
    base_dir = "/Users/nathenqian/Documents/code/nueral/poem"
    word_dictionary, data_generator = readDataFile(base_dir)
    print "the size of dictionary is  " + str(len(word_dictionary))
    data = T.tensor4()
    label = T.tensor4()

    output, cost, update, layers = local_fprop(data, label)
    train_func = theano.function(inputs=[data, label], outputs=[output, cost], updates=update, on_unused_input='warn', allow_input_downcast=True)
    test_func = theano.function(inputs=[data, label], outputs=[output, cost], on_unused_input='warn', allow_input_downcast=True)

    # then start train with function
    while True:
        print ">> enter command"
        print ">> "
        command = raw_input()
        if command == "t":
            print ">> enter time"
            t = int(raw_input())
            print ">> enter words number"
            cnt = int(raw_input())
            while t > 0:
                t -= 1
                data_generator.resetData()
                print "total data size is " + str(len(data_generator.data))
                index = 0
                while data_generator.hasNext() == True:
                    index += 1
                    print "train %s data" % (str(index))
                    train_data_in, train_data_label = data_generator.calcData()
                    output, cost = train_func(train_data_in, train_data_label)
                    print cost
                    # print data_generator.translate(output)
                    print data_generator.showProb(output, train_data_label)
                    # print data_generator.translate(train_data_sentence)
                    # from IPython import embed;embed()                    
                    data_generator.next()
                    if index == cnt:
                        break
        elif command == "shuffle":
            data_generator.shuffle()
        elif command == "save":
            print ">> enter name of this data"
            temp_name = raw_input()
            for layer in layers:
                layer.save_data(temp_name)
        elif command == "load":
            print ">> enter name of this data"
            temp_name = raw_input()
            for layer in layers:
                layer.load_data(temp_name)
        elif command == "ls":
            print ">> enter size"
            temp_name = int(raw_input())
            print data_generator.list(temp_name)
        elif command == "sort":
            data_generator.sortBySentenceLength()
        else:
            print ">> error"