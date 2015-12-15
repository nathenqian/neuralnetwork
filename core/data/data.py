from json import loads, dumps
import numpy as np
import random
class DataGenerator:
    def __init__(self, dir_, dictionary):
        with open(dir_, "r") as f:
            self.data = loads(f.read())
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
        sentence = d["sentence"][0]
        image = np.load(d["image_feature"] + ".npy")

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
