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
            "/Users/nathenqian/Documents/code/nueral/caffe/VGG_ILSVRC_16_layers.caffemodel", 

            caffe.TEST)
        print "init net finish --------------------------------------------------------------------------"

        # print net.blobs["data"].data.shape
    def processDir(self, dir_):
        # im = Image.open("/Users/nathenqian/Documents/code/nueral/iaprtc12/images/01/1037.jpg")
        im_input = np.random.randn(10, 3, 224, 224) * 0
        i = 0
        for single in dir_:
            im = Image.open(single)
            # for i in range(0, 10):
            # print single, i
            im_input[i, :, :, :] = np.asarray(im.resize((224, 224))).swapaxes(1, 2).swapaxes(0, 1) # / 256.0

            temp = np.copy(im_input[i, 0, :, :])
            # im_input[i, 0, :, :] -= 103.939
            # im_input[i, 1, :, :] -= 116.779
            # im_input[i, 2, :, :] -= 123.68
            # im_input[i, 0, :, :] -= 123.68
            # im_input[i, 1, :, :] -= 116.779
            # im_input[i, 2, :, :] -= 103.939
            im_input[i, 0, :, :] = np.copy(im_input[i, 2, :, :]) - 103.939
            im_input[i, 1, :, :] -= 116.779
            im_input[i, 2, :, :] = temp - 123.68

            i += 1
        # print im_input
        # print im_input[0, 0, 222, 2]
        # print im_input[0, 1, 222, 2]
        # print im_input[0, 2, 222, 2]
        self.net.blobs['data'].data[...] = im_input
        self.net.forward()
        return self.net.blobs["fc7"].data
        # return self.net.blobs["prob"].data


# caffe_net = CaffeNet()
# l = [
#     "/Users/nathenqian/Documents/code/nueral/iaprtc12/images/10/10008.jpg",
#     "/Users/nathenqian/Documents/code/nueral/iaprtc12/images/37/37089.jpg", 
#     "/Users/nathenqian/Documents/code/nueral/iaprtc12/images/37/37090.jpg",
#     "/Users/nathenqian/Documents/code/nueral/iaprtc12/images/10/10005.jpg",
#     "/Users/nathenqian/Documents/code/nueral/iaprtc12/images/10/10006.jpg",
#     "/Users/nathenqian/Documents/code/nueral/iaprtc12/images/37/37066.jpg",
#     "/Users/nathenqian/Documents/code/nueral/iaprtc12/images/37/37066.jpg",
#     "/Users/nathenqian/Documents/code/nueral/iaprtc12/images/37/37069.jpg",
#     "/Users/nathenqian/Documents/code/nueral/iaprtc12/images/37/37070.jpg"
# ]
# a = caffe_net.processDir(l)
# print a
# print np.max(a, axis = 1)
# print np.argmax(a, axis = 1)

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
