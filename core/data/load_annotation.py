import os
import sys
from json import dumps
sys.path.append("/Users/nathenqian/Documents/code/nueral/caffe")
from caffe_net import CaffeNet
word_dictionary = {}
word_index = 0

unit_list = []

def createImgDir(str_):
    start = str_.find("<IMAGE>") + len("<IMAGE>")
    end = str_.find("</IMAGE>")
    global base_dir
    return os.path.join(base_dir, str_[start:end])

def createSentences(str_):
    start = str_.find("<DESCRIPTION>") + len("<DESCRIPTION>")
    end = str_.find("</DESCRIPTION>")
    sentences = str_[start:end].split(";")[:-1]
    for i in range(0, len(sentences)):
        sentence = sentences[i]
        while sentence[0] == " ":
            sentence = sentence[1:]
        while sentence[-1] == " ":
            sentence = sentence[:-1]
        sentences[i] = sentence
    return sentences

def processRead(str_):
    global unit_list, npy_dir
    sentences = createSentences(str_)
    image_dir = createImgDir(str_)
    feature_dir = os.path.join(npy_dir, image_dir.split("/")[-1])[: -3] + "config"
    unit_list.append({"sentence" : sentences, "image_dir" : image_dir, "image_feature" : feature_dir})
    global word_index, word_dictionary
    for sentence in sentences:
        word_list = sentence.split(" ")
        for i in word_list:
            if i not in word_dictionary:
                word_index += 1
                word_dictionary[i] = word_index

def getFilesDir(dir_):
    base_files = os.listdir(dir_)
    files = []
    for f in base_files:
        if f[-3:] == "eng":
            files.append(os.path.join(dir_, f))
    return files

def readFile(dir_):
    content = ""
    with open(dir_, "r") as f:
        content = f.read()
    return content

annotation_dir = "/Users/nathenqian/Documents/code/nueral/iaprtc12/annotations_complete_eng"
npy_dir = "/Users/nathenqian/Documents/code/nueral/iaprtc12/image_npy"
base_dir = "/Users/nathenqian/Documents/code/nueral/iaprtc12"
def calcWordNumber():
    for index in range(0, 41):
        print "calcWordNumber processing " + str(index)
        dir_ = annotation_dir
        if index < 10:
            dir_ = os.path.join(dir_, "0" + str(index))
        else:
            dir_ = os.path.join(dir_, str(index))
        files = getFilesDir(dir_)
        for f in files:
            file_content = readFile(f)
            processRead(file_content)
    print "total words number is " + str(word_index) # 6378

def main(config_dir):
    for index in range(0, 1):
        # print "calcWordNumber processing " + str(index)
        dir_ = annotation_dir
        if index < 10:
            dir_ = os.path.join(dir_, "0" + str(index))
        else:
            dir_ = os.path.join(dir_, str(index))
        files = getFilesDir(dir_)
        for f in files:
            file_content = readFile(f)
            processRead(file_content)
    with open(config_dir, "w") as f:
        global unit_list
        f.write(dumps(unit_list, indent = 4))
    # print "total words number is " + str(word_index) # 6378
# calcWordNumber()
# caffe_net = CaffeNet()
main("/Users/nathenqian/Documents/code/nueral/iaprtc12/data_config.conf")
