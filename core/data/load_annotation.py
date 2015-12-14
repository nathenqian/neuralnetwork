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
        sentence = sentence.replace(",", "")
        sentence = sentence.replace("(", "")
        sentence = sentence.replace(")", "")
        sentence = sentence.replace('"', "")
        sentence = sentence.replace("'", "")
        sentence = sentence.replace("!", "")
        sentence = sentence.replace("?", "")
        while sentence[0] == " ":
            sentence = sentence[1:]
        while sentence[-1] == " ":
            sentence = sentence[:-1]
        while (sentence.replace("  ", " ") != sentence):
            sentence = sentence.replace("  ", " ")
        sentences[i] = sentence
    return sentences

def processRead(str_):
    global unit_list, npy_dir
    sentences = createSentences(str_)
    image_dir = createImgDir(str_)
    error = 0
    try:
        f = open(image_dir, "r")
    except Exception, e:
        error = 1
    if error == 1:
        return
    feature_dir = os.path.join(npy_dir, image_dir.split("/")[-1])[: -3] + "config"
    unit_list.append({"sentence" : sentences, "image_dir" : image_dir, "image_feature" : feature_dir})
    
    # for sentence in sentences:
    #     word_list = sentence.split(" ")
    #     for i in word_list:
    #         if i not in word_dictionary:
    #             word_index += 1
    #             word_dictionary[i] = word_index

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


    invalid_list = {}
    for i in range(0, len(unit_list)):
        try:
            s = dumps(unit_list[i], indent = 4)
        except Exception as e:
            invalid_list[i] = 1
    final_list = []
    for i in range(0, len(unit_list)):
        if i not in invalid_list:
            final_list.append(unit_list[i])

    global word_index, word_dictionary
    for i in final_list:
        for sentence in i["sentence"]:
            word_list = sentence.split(" ")
            for i in word_list:
                if i not in word_dictionary:
                    word_index += 1
                    word_dictionary[i] = word_index

    with open("/Users/nathenqian/Documents/code/nueral/iaprtc12/word_dictionary.txt", "w") as f:
        f.write(dumps(word_dictionary, indent = 4))

    print "total words number is " + str(word_index) # 6378

def main(index):
    print "process index " + str(index)
    global unit_list
    unit_list = []
    for index in range(index, index + 1):
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

    invalid_list = {}
    for i in range(0, len(unit_list)):
        try:
            s = dumps(unit_list[i], indent = 4)
        except Exception as e:
            invalid_list[i] = 1
    final_list = []
    for i in range(0, len(unit_list)):
        if i not in invalid_list:
            final_list.append(unit_list[i])
    if index < 10:
        digit = "0" + str(index)
    else:
        digit = str(index)
    config_dir = "/Users/nathenqian/Documents/code/nueral/iaprtc12/data_config_" + digit + ".conf"
    with open(config_dir, "w") as f:
        f.write(dumps(final_list, indent = 4))
            # f.write(str(unit_list[2214]))
            # if (i == 2214):
                # break
    # print "total words number is " + str(word_index) # 6378
calcWordNumber()
# caffe_net = CaffeNet()
# for i in range(0, 41):
    # main(i)
