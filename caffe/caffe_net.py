import sys
sys.path.append("/Users/nathenqian/Downloads/caffe/python")
import caffe
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


class CaffeNet:
    def __init__(self):
        caffe.set_mode_cpu()

        self.net = caffe.Net('/Users/nathenqian/Documents/code/nueral/caffe/VGG_ILSVRC_16_layers.prototxt', 
            "/Users/nathenqian/Documents/code/nueral/caffe/VGG_ILSVRC_16_layers.caffemodel", caffe.TEST)
        print "init net finish --------------------------------------------------------------------------"

        # print net.blobs["data"].data.shape
    def processDir(self, dir_):
        # im = Image.open("/Users/nathenqian/Documents/code/nueral/iaprtc12/images/01/1037.jpg")
        im = Image.open(dir_)
        im_input = np.random.randn(10, 3, 224, 224) * 0
        for i in range(0, 10):
            im_input[i, :, :, :] = np.asarray(im.resize((224, 224))).swapaxes(1, 2).swapaxes(0, 1) / 256.0
        self.net.blobs['data'].data[...] = im_input
        self.net.forward()
        return self.net.blobs["fc7"].data[0, :]


# caffe_net = CaffeNet()
# print caffe_net.processDir("/Users/nathenqian/Documents/code/nueral/iaprtc12/images/01/1037.jpg")

        # print net.blobs["prob"].data
        # print np.argmax(net.blobs["prob"].data, axis = 1)
        # print net.blobs["prob"].data[0, np.argmax(net.blobs["prob"].data, axis = 1)]

# im = Image.open("/Users/nathenqian/Documents/code/nueral/iaprtc12/images/01/1137.jpg")
# im_input = np.random.randn(10, 3, 224, 224) * 0
# for i in range(0, 10):
#     im_input[i, :, :, :] = np.asarray(im.resize((224, 224))).swapaxes(1, 2).swapaxes(0, 1) / 256.0

# net.blobs['data'].data[...] = im_input
# net.forward()
# print net.blobs["prob"].data
# print np.argmax(net.blobs["prob"].data, axis = 1)
# print net.blobs["prob"].data[0, np.argmax(net.blobs["prob"].data, axis = 1)]

# # transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
# # transformer.set_mean('data', np.load('python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
# # transformer.set_transpose('data', (2,0,1))
# # transformer.set_channel_swap('data', (2,1,0))
# # transformer.set_raw_scale('data', 255.0)



# # im_input = im[np.newaxis, np.newaxis, :, :]
# # net.blobs['data'].reshape(*im_input.shape)
# # net.blobs['data'].data[...] = im_input

# # print net.blobs["conv"].data
# # print "---------------------------------------------------"
# # net.forward()
# # print net.blobs["conv"].data
# # net.save("mymodel.caffemodel")
