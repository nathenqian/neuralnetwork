import os
import sys
from json import dumps, loads
sys.path.append("/Users/nathenqian/Documents/code/nueral/caffe")
from caffe_net import CaffeNet
import numpy as np

base_dir = "/Users/nathenqian/Documents/code/nueral/iaprtc12/annotations_complete_eng"
npy_dir = "/Users/nathenqian/Documents/code/nueral/iaprtc12/image_npy"

def main(config_dir):
    global caffe_net
    with open(config_dir) as f:
        config = f.read()
    config = loads(config)
    print "total config size is " + str(len(config))
    index = 0
    list_size = 0
    image_dir_list = []
    image_feature_list = []
    for i in config:
        index += 1
        list_size += 1
        if index % 100 == 0:
            print "processing index is %s" % (str(index))
        image_dir = i["image_dir"]
        print image_dir
        image_feature = i["image_feature"]
        image_dir_list.append(image_dir)
        image_feature_list.append(image_feature)
        if list_size == 10:
            npy_eles = caffe_net.processDir(image_dir_list)
            for ind in range(0, 10):
                np.save(image_feature_list[ind], npy_eles[ind, :])
            image_dir_list = []
            image_feature_list = []
            list_size = 0
    if list_size > 0:
        npy_eles = caffe_net.processDir(image_dir_list)
        for ind in range(0, list_size):
            np.save(image_feature_list[ind], npy_eles[ind, :])



    # print "total words number is " + str(word_index) # 6378
# calcWordNumber()
caffe_net = CaffeNet()
for i in range(37, 41):
    if i < 10:
        ind = "0" + str(i)
    else:
        ind = str(i)
    print "process index is " + str(i)
    main("/Users/nathenqian/Documents/code/nueral/iaprtc12/data_config_" + ind + ".conf")
