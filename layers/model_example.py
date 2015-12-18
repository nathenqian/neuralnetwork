from layers import *
import theano
import theano.tensor as T
import numpy as np
from json import loads, dumps
import sys
import os
sys.path.append("/Users/nathenqian/Documents/code/nueral/core")
from data.data import DataGenerator
from time import localtime
def printTime():
    t = localtime()
    print "time is %d:%d:%d"%(t[3], t[4], t[5])

word_dictionary, data_generator = (None, None)

def local_fprop(x, y_truth, image_feature): # x = sentences  y_truth is [[][]]
    global word_dictionary

    fc0 = ColumnWiseFullyConnected(
            input_dim=len(word_dictionary),
            output_dim=128,
            nonlinearity=tanhFunc,
            dropout_prob=0.2,
            name = "fc0"
        )

    fc1 = ColumnWiseFullyConnected(
        input_dim=128,
        output_dim=256,
        nonlinearity=tanhFunc,
        dropout_prob=0.2,
        name = "fc1"
        )

    rnn0 = LSTM(
            input_dim=256,
            output_dim=256,
            # ,nonlinearity=tanhFunc
            name = "lstm0"
            )

    multimodal_rnn = ColumnWiseFullyConnected(
            input_dim=256,
            output_dim=256,
            nonlinearity=tanhFunc,
            dropout_prob=0.2,
            name = "multimodal_rnn"
        )

    multimodal_image = ColumnWiseFullyConnected(
            input_dim=4096,
            output_dim=256,
            nonlinearity=tanhFunc,
            dropout_prob=0.2,
            name = "multimodal_image"
    )

    multimodal_input = ColumnWiseFullyConnected(
        input_dim=256,
        output_dim=256,
        nonlinearity=tanhFunc,
        dropout_prob=0.2,
        name = "multimodal_input"
    )

    fc2 = ColumnWiseFullyConnected(
        input_dim=256,
        output_dim=len(word_dictionary),
        nonlinearity=specialTanhFunc,
        dropout_prob=0.2,
        name = "fc2"
        )

    softmax = ColumnWiseSoftmax(
        name = "softmax"
    )

    embeded1_out = fc0.fprop(x)
    embeded2_out = fc1.fprop(embeded1_out)
    rnn0_out = rnn0.fprop(embeded2_out)
    # return rnn0_out ,rnn0_out , []
    multimodal_rnn_out = multimodal_rnn.fprop(rnn0_out)
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
    layers = [fc0, fc1, fc2, multimodal_rnn, multimodal_input, rnn0, softmax,
        multimodal_image
    ]
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
    with open(os.path.join(base_dir, "word_dictionary.txt"), "r") as f:
        word_dictionary = loads(f.read())
    data_generator = DataGenerator(os.path.join(base_dir, "data.conf"), word_dictionary, "/Users/nathenqian/Documents/code/nueral/iaprtc12/image_npy")
    print "data_size is " + str(len(data_generator.data))
    return word_dictionary, data_generator


if __name__ == '__main__':
    base_dir = "/Users/nathenqian/Documents/code/nueral/iaprtc12"
    word_dictionary, data_generator = readDataFile(base_dir)
    print "the size of dictionary is  " + str(len(word_dictionary))
    data = T.tensor4()
    label = T.tensor4()
    image_feature = T.tensor4()

    output, cost, update, layers = local_fprop(data, label, image_feature)
    train_func = theano.function(inputs=[data, label, image_feature], outputs=[output, cost], updates=update, on_unused_input='warn', allow_input_downcast=True)
    test_func = theano.function(inputs=[data, label, image_feature], outputs=[output, cost], on_unused_input='warn', allow_input_downcast=True)

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
                    train_data_sentence, train_data_result, train_data_image_feature = data_generator.calcData()
                    output, cost = train_func(train_data_sentence, train_data_result, train_data_image_feature)
                    print cost
                    # print data_generator.translate(output)
                    print data_generator.showProb(output, train_data_sentence)
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