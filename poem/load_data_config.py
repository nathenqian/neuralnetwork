from json import dumps


word_dictionary = {}
file_written = []
word_cnt = {}

with open("raw_train.data", "r") as f:
    content = f.readline()
    while content != "":
        sentences = content.split("\n")[0].split("\t")
        valid = True
        for sentence in sentences:
            if sentence.find("<") != -1:
                valid = False
        if valid:
            poem = []
            for sentence in sentences:
                words = sentence.split(" ")
                if len(words) != 5:
                    continue
                sentence_list = []
                for word in words:
                    if word not in word_dictionary:
                        word_cnt[word] = (len(word_dictionary), 0)
                        word_dictionary[word] = len(word_dictionary)
                    sentence_list.append(word_dictionary[word])
                    word_cnt[word] = (word_cnt[word][0], word_cnt[word][1] + 1)

                poem.append(sentence_list)
            if len(poem) > 0:
                file_written.append(poem)
        content = f.readline()

word_cnt_list = []
for i in word_cnt:
    word_cnt_list.append(word_cnt[i])
word_cnt_list = sorted(word_cnt_list, key = lambda a : -a[1])


limited = 40
new_word_dictionay = {}
new_word_index = 1
for i in word_cnt_list:
    if i[1] <= limited:
        new_word_dictionay[i[0]] = 0
    else:
        new_word_dictionay[i[0]] = new_word_index
        new_word_index += 1

for poem in file_written:
    for sentence in poem:
        for i in range(0, len(sentence_list)):
            sentence[i] = new_word_dictionay[sentence[i]]
temp = {}
for i in word_dictionary:
    if new_word_dictionay[word_dictionary[i]] != 0:
        temp[i] = new_word_dictionay[word_dictionary[i]]

word_dictionary = temp
word_dictionary["unknown"] = 0

with open("train.data2", "w") as f:
    f.write(dumps(file_written, indent = 4))
print "dictionary size is %s" % (str(len(word_dictionary)))
with open("dictionary.data2", "w") as f:
    f.write(dumps(word_dictionary, indent = 4))
