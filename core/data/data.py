from json import loads, dumps


class Data:
    def __init__(self, img, des):
        self.img = img
        self.des = des


class DataGenerator:
    def __init__(self):
        self.data_index = 0
        self.data_list = []

    def resetData(self):
        self.data_index = 0

    def data(self):
        return self.data[self.data_index]

    def hasNext(self):
        return self.data_index != len(self.data_list)

    def next(self):
        self.data_index += 1
        return self.data[self.data_index - 1]
 
    def createTestBatch(self, batch_size):
        for index in range(0, batch_size):
            if self.hasNext():
                self.next()
