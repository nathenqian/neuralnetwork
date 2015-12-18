from layers import *
import theano
import theano.tensor as T
import numpy as np
from json import loads, dumps
import sys
import os
sys.path.append("/Users/nathenqian/Documents/code/nueral/core")
from data.data import DataGenerator, DataGeneratorFeature
from time import localtime
def printTime():
    t = localtime()
    print "time is %d:%d:%d"%(t[3], t[4], t[5])

word_dictionary, data_generator = (None, None)

def local_fprop(image_feature, y_truth): # x = sentences  y_truth is [[][]]
    global word_dictionary

    # fc0 = ColumnWiseFullyConnected(
    #         input_dim=len(word_dictionary),
    #         output_dim=512,
    #         nonlinearity=tanhFunc,
    #         # dropout_prob=0.2,
    #         name = "fc0"
    #     )

    # fc1 = ColumnWiseFullyConnected(
    #     input_dim=512,
    #     output_dim=256,
    #     nonlinearity=tanhFunc,
    #     # dropout_prob=0.2,
    #     name = "fc1"
    #     )


    multimodal_image = ColumnWiseFullyConnected(
            input_dim=4096,
            output_dim=1024,
            nonlinearity=tanhFunc,
            dropout_prob=0.2,
            name = "multimodal_image"
    )

    multimodal_image2 = ColumnWiseFullyConnected(
            input_dim=1024,
            output_dim=256,
            nonlinearity=tanhFunc,
            dropout_prob=0.2,
            name = "multimodal_image2"
    )


    fc2 = ColumnWiseFullyConnected(
        input_dim=256,
        output_dim=len(word_dictionary),
        nonlinearity=specialTanhFunc,
        dropout_prob=0.2,
        name = "fc2"
        )

    euc0 = EuclideanLoss(
        name = "euc"
    )


    # embeded1_out = fc0.fprop(x)
    # embeded2_out = fc1.fprop(embeded1_out)
    multimodal_image_out = multimodal_image.fprop(image_feature)
    multimodal_image_out2 = multimodal_image2.fprop(multimodal_image_out)
    # multimodal_out = embeded2_out + multimodal_image_out
    my_out = fc2.fprop(multimodal_image_out2)

    # my_cost = softmax.cost(t3, y_truth)

    def _cost(y_in, y_truth):
        res = T.sqrt(((y_in-y_truth)**2+1e-8).sum(axis=2)).sum()
        return res

    my_cost = _cost(my_out, y_truth)

    my_update = []
    layers = [fc2, multimodal_image, multimodal_image2]
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
    data_generator = DataGeneratorFeature(os.path.join(base_dir, "data.conf"), word_dictionary, "/Users/nathenqian/Documents/code/nueral/iaprtc12/image_npy")
    print "data_size is " + str(len(data_generator.data))
    return word_dictionary, data_generator


if __name__ == '__main__':
    base_dir = "/Users/nathenqian/Documents/code/nueral/iaprtc12"
    word_dictionary, data_generator = readDataFile(base_dir)
    print "the size of dictionary is  " + str(len(word_dictionary))

    label = T.tensor4()
    image_feature = T.tensor4()

    output, cost, update, layers = local_fprop(label, image_feature)
    train_func = theano.function(inputs=[label, image_feature], outputs=[output, cost], updates=update, on_unused_input='warn', allow_input_downcast=True)
    test_func = theano.function(inputs=[label, image_feature], outputs=[output, cost], on_unused_input='warn', allow_input_downcast=True)

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
                batch_size = 100
                while data_generator.hasNext() == True:
                    index += batch_size
                    print "train %s data" % (str(index))
                    train_data_result, train_data_image_feature = data_generator.calcBatchData(batch_size)
                    output, cost = train_func(train_data_image_feature, train_data_result)
                    print cost
                    # print data_generator.translate(output)
                    # print data_generator.showProb(output, train_data_result)
                    # print data_generator.translate(train_data_sentence)
                    # from IPython import embed;embed()                    
                    data_generator.nextBatch(batch_size)
                    if index >= cnt:
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
        elif command == "test":
            data_generator.resetData()
            data_generator.shuffle()
            test_case = 100
            total_feature = 0
            error_feature = 0
            print "start test"
            while test_case > 0:
                test_case -= 1
                data_result, data_image_feature = data_generator.calcData()
                a, b = test_func(data_image_feature, data_result)
                feature_index = []
                output_list = []
                case_total = 0
                for i in range(0, data_result.shape[2]):
                    if (data_result[0, 0, i, 0] > 0.00001):
                        total_feature += 1
                        case_total += 1
                        feature_index.append(i)
                    output_list.append((a[0, 0, i, 0], i))
                output_list = sorted(output_list, key = lambda a : -a[0])
                # print output_list[0 : 10], feature_index
                for i in range(case_total):
                    if output_list[i][1] not in feature_index:
                        error_feature += 1

                data_generator.next()
            print "finish test total_feature is %s   error_feature is %s" % (str(total_feature), str(error_feature))

        else:
            print ">> error"
