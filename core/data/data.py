from json import loads, dumps
import numpy as np
import random
import os

class DataGenerator:
    def __init__(self, dir_, dictionary, img_path):
        with open(dir_, "r") as f:
            self.data = loads(f.read())
        self.img_path = img_path
        self.data_index = 0
        self.word_dictionary_size = len(dictionary)
        self.dictionary = dictionary
        self.rev_dictionary = {}
        for i in dictionary:
            self.rev_dictionary[dictionary[i]] = i

    def resetData(self):
        self.data_index = 0

    def shuffle(self):
        random.shuffle(self.data)

    def data(self):
        return self.data[self.data_index]

    def hasNext(self):
        return self.data_index != len(self.data)

    def next(self):
        self.data_index += 1
        return self.data[self.data_index - 1]

    def translateWordVector(self, size, index):
        ret = np.random.randn(size) * 0
        ret[index] = 1
        return ret

    def calcData(self):
        d = self.data[self.data_index]
        sentence = d['sentence'][0]
        fpath = os.path.join(self.img_path, d['image_feature'].split('/')[-1])
        image = np.load(fpath + ".npy")

        words = sentence.split(" ")
        # words = words[1:]
        input_ = np.random.randn(1, 1, self.word_dictionary_size, len(words) + 1) * 0
        output = np.random.randn(1, 1, self.word_dictionary_size, len(words) + 1) * 0
        image_feature = np.random.randn(1, 1, image.shape[0], len(words) + 1) * 0

        
        for index in range(0, len(words) + 1):
            if index == 0:
                input_[0, 0, self.word_dictionary_size - 2, index] = 1
                output[0, 0, self.dictionary[words[index]] - 1, index] = 1
            elif index == len(words):
                input_[0, 0, self.dictionary[words[index - 1]] - 1, index] = 1
                output[0, 0, self.word_dictionary_size - 1, index] = 1
            else:
                input_[0, 0, self.dictionary[words[index - 1]] - 1, index] = 1
                output[0, 0, self.dictionary[words[index]] - 1, index] = 1
            image_feature[0, 0, :, index] = image

        return (input_, output, image_feature)
    def translate(self, output):
        ret = ""
        for i in range(output.shape[3]):
            print np.max(output[0, 0, :, i], axis = 0)
            n = np.argmax(output[0, 0, :, i], axis = 0)
            ret = ret + " " + self.rev_dictionary[n + 1]
        return ret


    # def createTestBatch(self, batch_size):
    #     for index in range(0, batch_size):
    #         if self.hasNext():
    #             self.next()

english_char = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

class DatasetSimple(DataGenerator):
    def get_data_stream(self):
        for idx in range(len(self.data)):
            d = self.data[idx]
            if len(d['sentence']) == 0:
                continue
            words = self.deal_sentence(d['sentence'][0])
            if words == None:
                continue
            ipth = os.path.join(self.img_path, d['image_feature'].split('/')[-1]+'.npy')
            if not os.path.exists(ipth):
                continue
            img = np.load(ipth)

            _n = len(words)+1
            data = np.zeros((1,1,self.word_dictionary_size, _n), dtype='float32')
            label = np.zeros((1,1,self.word_dictionary_size, _n), dtype='float32')
            img_ft = np.zeros((1,1,img.shape[0],_n), dtype='float32')


            data[0,0,self.word_dictionary_size-2,0]=1
            for ii in range(1, _n):
                data[0,0,self.dictionary[words[ii-1]]-1, ii]=1

            label[0,0,self.word_dictionary_size-1, _n-1]=1
            for ii in range(0, _n-1):
                label[0,0,self.dictionary[words[ii]]-1, ii]=1

            for ii in range(0, _n):
                img_ft[0,0,:,ii]=img

            yield (data, label, img_ft)

    def deal_sentence(self, stc):
        words = []
        for ii in stc.split(' '):
            if len(ii)>0:
                for ch in ii:
                    if ch not in ii:
                        return None
                words.append(ii)
#        if len(words) > 10:
#            return None
        return words
