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
    for i in config:
        index += 1
        if index % 100 == 0:
            print "processing index is %s" % (str(index)) 
        image_dir = i["image_dir"]
        image_feature = i["image_feature"]
        npy_ele = caffe_net.processDir(image_dir)
        np.save(image_feature, npy_ele)



    # print "total words number is " + str(word_index) # 6378
# calcWordNumber()
caffe_net = CaffeNet()
main("/Users/nathenqian/Documents/code/nueral/iaprtc12/data_config.conf")