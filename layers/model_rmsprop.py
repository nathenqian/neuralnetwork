from layers import *
import theano
import theano.tensor as T
import numpy as np
from json import loads, dumps
import sys
import os
sys.path.append("../core")
from data.data import DataGenerator
from data.data import DatasetSimple
from time import localtime
import argparse
from monitor import Monitor

def printTime():
    t = localtime()
    print "time is %d:%d:%d"%(t[3], t[4], t[5])

word_dictionary, data_generator = (None, None)

def local_fprop(x, y_truth, image_feature): # x = sentences  y_truth is [[][]]
    global word_dictionary

    fc0 = ColumnWiseFullyConnected(
            input_dim=len(word_dictionary),
            output_dim=128,
            nonlinearity=reluFunc,
            name = "fc0"
    )

    fc1 = ColumnWiseFullyConnected(
            input_dim=128,
            output_dim=256,
            nonlinearity=reluFunc,
            name = "fc1",
            dropout_prob = 0.5
    )

    rnn0 = LSTM(
            input_dim=256,
            output_dim=256,
            # ,nonlinearity=reluFunc
            name = "lstm0"
    )

    """
    multimodal_rnn = ColumnWiseFullyConnected(
            input_dim=256,
            output_dim=256,
            nonlinearity=reluFunc,
            name = "multimodal_rnn",
    )
    """

    multimodal_image = ColumnWiseFullyConnected(
            input_dim=4096,
            output_dim=256,
            nonlinearity=reluFunc,
            name = "multimodal_image"
    )

    multimodal_input = ColumnWiseFullyConnected(
        input_dim=256,
        output_dim=256,
        nonlinearity=reluFunc,
        name = "multimodal_input"
    )

    fc2 = ColumnWiseFullyConnected(
        input_dim=256,
        output_dim=len(word_dictionary),
        nonlinearity=specialTanhFunc,
        name = "fc2",
        dropout_prob=0.5
        )

    softmax = ColumnWiseSoftmax(
        name = "softmax"
    )

    embeded1_out = fc0.fprop(x)
    embeded2_out = fc1.fprop(embeded1_out)
    rnn0_out = rnn0.fprop(embeded2_out)
    # return rnn0_out ,rnn0_out , []
#    multimodal_rnn_out = multimodal_rnn.fprop(rnn0_out)
    multimodal_rnn_out = rnn0_out
    multimodal_image_out = multimodal_image.fprop(image_feature)
    multimodal_input_out = multimodal_input.fprop(embeded2_out)
    multimodal_out = multimodal_rnn_out + multimodal_input_out + multimodal_image_out
    softmax_in = fc2.fprop(multimodal_out)
    my_out = softmax.fprop(softmax_in)
    # my_cost = softmax.cost(t3, y_truth)

    def _local_cost(y, y_truth):
        tmp = y * y_truth
        # tmp = T.log(tmp.sum(axis=2)).mean(axis=2).mean()
        tmp = T.log(tmp.max(axis=2)).sum(axis=2).mean()
        tmp = -tmp
        return tmp

    my_cost = _local_cost(my_out, y_truth)

    my_update = []
    layers = [fc0, fc1, fc2, 
#multimodal_rnn, 
        multimodal_input, rnn0, softmax,
        multimodal_image
    ]

    # L2 regularization
    for layer in layers:
        for p in layer.get_params():
            my_cost += (p**2).mean()

    # RMSprop
#    rho = 0.9
#    epision = 1e-6
    for layer in layers:
        for p in layer.get_params():
            lr = layer.learning_rate
            wd = layer.weight_decay
            g = T.grad(cost=my_cost, wrt=p)
#            p_delta = theano.shared(p.get_value() * 0.)
#            p_delta_new = rho * p_delta + (1 - rho) * g**2;
#            gradient_scaling = T.sqrt(p_delta_new + epision)
#            g = g / gradient_scaling
#            my_update.append((p_delta, p_delta_new))
            my_update.append((p, p - lr * (p * wd + g)))

    return my_out, my_cost, my_update, layers


def readDataFile(base_dir):
    with open(os.path.join(base_dir, "word_dictionary.txt"), "r") as f:
        word_dictionary = loads(f.read())
    data_generator = DatasetSimple(os.path.join(base_dir, "config_all.conf"), word_dictionary, base_dir+'/image_npy')
    return word_dictionary, data_generator


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--show',
                help="show the every point's situation", action='store_true')
    parser.add_argument('-l', '--load_path',
                help="load from model file")
    args = parser.parse_args()

    base_dir = "../local_data"
    word_dictionary, data_generator = readDataFile(base_dir)
#    from IPython import embed;embed()
    print "the size of dictionary is  " + str(len(word_dictionary))
    data = T.tensor4()
    label = T.tensor4()
    image_feature = T.tensor4()

    output, cost, update, layers = local_fprop(data, label, image_feature)
    train_func = theano.function(inputs=[data, label, image_feature], outputs=[output, cost], updates=update,
            on_unused_input='warn', allow_input_downcast=True)
#    test_func = theano.function(inputs=[data, label, image_feature], outputs=[output, cost],
#            on_unused_input='warn', allow_input_downcast=True)
    
    mtr = Monitor(layers, load_path = args.load_path, save_path = './train_log_rmsprop')
    if args.load_path:
        mtr.load()

    # then start train with function
    while True:
        epoch_no = 0
        while epoch_no < 100:
            epoch_no += 1
            print "training {} epoch, total data size is {}".format(epoch_no, str(len(data_generator.data)))
            data_generator.shuffle()
            cost_set = []
            index = 0
            for datapoint in data_generator.get_data_stream():
                index += 1
                train_data_sentence = datapoint[0]
                train_data_result = datapoint[1]
                train_data_image_feature = datapoint[2]
                output, cost = train_func(train_data_sentence, train_data_result, train_data_image_feature)
                cost_set.append(cost)

                sys.stderr.write("train {0} data, cost = {1} \r".format(index, cost))
                sys.stderr.flush()
                if args.show:
                    print cost
                    print data_generator.translate(output)
                    print data_generator.translate(train_data_sentence)
            batch_cost = sum(cost_set)/len(cost_set)
            print 'current batch cost = {}'.format(batch_cost)
            # save after trained
            mtr.save(batch_cost)
            print 'min batch cost = {}'.format(mtr.min_cost)
