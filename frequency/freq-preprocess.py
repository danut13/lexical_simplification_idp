import pickle
import math


def sigmoid(x):
    z = math.exp(-x)
    return 1 / (1 + z)


filename1 = 'eng_wikipedia_2016_1M/eng_wikipedia_2016_1M-words.txt'
filename2 = 'eng_news_2016_1M/eng_news_2016_1M-words.txt'
eng_dict = {}
freq_limit = 50

# Reads from word frequency of english wikipedia and saves it into a dictionary
with open(filename1) as f:
    for line in f:
        line = line.strip()
        list_line = line.split('\t')
        word_ID = int(list_line[0])
        word = list_line[1].lower()
        word_count = int(list_line[3])
        if word_ID > 100 and word_count > freq_limit:
            if word in eng_dict:
                eng_dict[word] += word_count
            else:
                eng_dict[word] = word_count
        if word_ID > 100 and word_count < freq_limit:
            break

# Reads from word frequency of english news and saves it into the same dictionary
with open(filename1) as f:
    for line in f:
        line = line.strip()
        list_line = line.split('\t')
        word_ID = int(list_line[0])
        word = list_line[1].lower()
        word_count = int(list_line[3])
        if word_ID > 100 and word_count > freq_limit:
            if word in eng_dict:
                eng_dict[word] += word_count
            else:
                eng_dict[word] = word_count
        if word_ID > 100 and word_count < freq_limit:
            break

# Normalization
eng_dict_norm = {}

for word, word_count in eng_dict.items():
    count_scaled = (sigmoid(word_count/1000) - 0.5) * 2
    eng_dict_norm[word] = count_scaled

with open('word-freq-eng.pkl', 'wb') as f:
    pickle.dump(eng_dict_norm, f)
