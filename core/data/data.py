from json import loads, dumps
import numpy as np
import random
import os
import math
import time
class DataGenerator:
    def __init__(self, dir_, dictionary, img_path):
        with open(dir_, "r") as f:
            self.data = loads(f.read())
        temp = []
        for item in self.data:
            if len(item["sentence"]) != 0:
                temp.append(item)
        self.data = temp
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

    def list(self, cnt):
        cnt = min(len(self.data), cnt)
        if cnt == -1:
            cnt = len(self.data)
        ret = []
        for i in range(0, cnt):
            ret.append(self.data[i]["sentence"][0])
        return ret

    def sortBySentenceLength(self):
        self.data = sorted(self.data, key = lambda a : len(a["sentence"][0]))

    # def createTestBatch(self, batch_size):
    #     for index in range(0, batch_size):
    #         if self.hasNext():
    #             self.next()
    def showProb(self, output, correct):
        ret = ""
        ret_corr = ""
        for i in range(output.shape[3]):
            print np.max(output[0, 0, :, i], axis = 0), output[0, 0, np.argmax(correct[0, 0, :, i], axis = 0), i]
            n = np.argmax(output[0, 0, :, i], axis = 0)
            ret = ret + " " + self.rev_dictionary[n + 1]
            ret_corr = ret_corr + " " + self.rev_dictionary[np.argmax(correct[0, 0, :, i], axis = 0) + 1]
        return ret + "\n" + ret_corr

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

class DataGeneratorFeature(DataGenerator):
    def calcData(self):
        d = self.data[self.data_index]
        sentence = d['sentence'][0]
        fpath = os.path.join(self.img_path, d['image_feature'].split('/')[-1])
        image = np.load(fpath + ".npy")

        words = sentence.split(" ")
        # words = words[1:]
        output = np.random.randn(1, 1, self.word_dictionary_size, 1) * 0
        image_feature = np.random.randn(1, 1, 4096, 1) * 0
        image_feature[0, 0, :, 0] = image

        for index in range(0, len(words)):
            output[0, 0, self.dictionary[words[index]] - 1, 0] = 1

        return (output, image_feature)


    def calcBatchData(self, batch_size):
        data_index_bak = self.data_index
        cnt = min(batch_size, len(self.data) - self.data_index)

        ret_output = np.random.randn(1, 1, self.word_dictionary_size, cnt) * 0
        ret_image_feature = np.random.randn(1, 1, 4096, cnt) * 0
        for index in range(0, cnt):
            a, b = self.calcData()
            ret_output[0, 0, :, index] = a[0, 0, :, 0]
            ret_image_feature[0, 0, :, index] = b[0, 0, :, 0]
            self.next()

        self.data_index = data_index_bak
        return (ret_output, ret_image_feature)


    def nextBatch(self, batch_size):
        for i in range(0, batch_size):
            if self.hasNext():
                self.next()


class PoemDataGenerator:
    def __init__(self, dir_, dictionary):
        with open(dir_, "r") as f:
            temp = loads(f.read())
        self.rev_dictionary = {}
        for i in dictionary:
            self.rev_dictionary[dictionary[i]] = i
        size = self.detectData(temp, self.rev_dictionary)
        self.data = []
        self.data_word_size = {i : 0 for i in range(len(dictionary))}
        for poem in temp:
            for i in range(3):
                s = poem[i]
                t = poem[i + 1]
                p = self.calcSentenceProb(size, s) + self.calcSentenceProb(size, t)
                print p
                if p < 120000 * random.random():
                    self.data.append((s, t))
                    for z in s:
                        self.data_word_size[z] += 1
        calc_list = []
        for i in self.data_word_size:
            calc_list.append(self.data_word_size[i])
        print max(calc_list)
        time.sleep(3)
        print calc_list
        self.data_index = 0
        self.word_dictionary_size = len(dictionary)
        self.dictionary = dictionary

    def calcSentenceProb(self, size, sentence):
        ret = 0
        for word in sentence:
            cnt = size[word][1]
            ret = ret + math.log(cnt) ** 5
            # if cnt <= 100:
            #     ret *= 100 / cnt
            #     continue
            # ret /= cnt / 100.0
        return ret

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

    def calcData(self):
        s = self.data[self.data_index][0]
        t = self.data[self.data_index][1]

        input_ = np.random.randn(1, 1, self.word_dictionary_size, 5) * 0
        output = np.random.randn(1, 1, self.word_dictionary_size, 5) * 0

        for index in range(0, 5):
            input_[0, 0, s[index], index] = 1
            output[0, 0, t[index], index] = 1

        return (input_, output)

    def list(self, cnt):
        cnt = min(len(self.data), cnt)
        if cnt == -1:
            cnt = len(self.data)
        ret = []
        for i in range(0, cnt):
            ret.append(self.data[i]["sentence"][0])
        return ret

    def showProb(self, output, correct):
        ret = ""
        ret_corr = ""
        for i in range(output.shape[3]):
            # print np.argmax(correct[0, 0, :, i], axis = 0)
            # print np.argmax(output[0, 0, :, i], axis = 0)
            # print np.max(output[0, 0, :, i], axis = 0), output[0, 0, np.argmax(correct[0, 0, :, i], axis = 0), i]
            print output[0, 0, np.argmax(output[0, 0, :, i], axis = 0), i], output[0, 0, np.argmax(correct[0, 0, :, i], axis = 0), i]
            n = np.argmax(output[0, 0, :, i], axis = 0)
            ret = ret + " " + self.rev_dictionary[n]
            ret_corr = ret_corr + " " + self.rev_dictionary[np.argmax(correct[0, 0, :, i], axis = 0)]
        return ret + "\n" + ret_corr

    def detectData(self, data, rev_dictionary):
        size = []
        for i in range(0, len(rev_dictionary)):
            size.append((i, 0))
        for poem in data:
            for sentence in poem:
                for word in sentence:
                    size[word] = (word, size[word][1] + 1)
        p = sorted(size, key = lambda a : -a[1])
        with open("word_freq.txt", "w") as f:
            for i in p:
                f.write(str(i[1]) + " ")
                f.write(rev_dictionary[i[0]].encode("utf8") + "\n")
        return size