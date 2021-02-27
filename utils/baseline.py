import re
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from nltk.corpus import cmudict
from nltk.corpus import wordnet as wn
from curses.ascii import isdigit


class Baseline(object):

    def __init__(self, language):
        self.language = language

        if language == 'english':
            self.avg_word_length = 5.3
            self.lowfreqchar = re.compile('j|k|q|v|x|z')
            self.lowfreqdict = {'j': 0.847, 'k': 0.228, 'q': 0.905, 'v': 0.022,
                                'x': 0.85, 'z': 0.926}
            self.highfreqchar = re.compile('e|t|a|o|i')
            self.highfreqdict = {'e': 1.27, 't': 0.906, 'a': 0.817, 'o': 0.751,
                                 'i': 0.697}
            self.syll = cmudict.dict()
            self.vowels = re.compile('[aeiouyAEIOUY]')
            self.consonants = re.compile('[qwrtypsdfghjklzxcvbnmQWRTYPSDFGHJKLZXCVBNM]')
            self.mul_vow = re.compile('[aeiouyAEIOUY]{3,}')
            self.mul_cons = re.compile('[qwrtypsdfghjklzxcvbnmQWRTYPSDFGHJKLZXCVBNM]{3,}')
            self.trans_cons = re.compile(
                '[qwrtypsdfghjklzxcvbnmQWRTYPSDFGHJKLZXCVBNM](?![qwrtypsdfghjklzxcvbnmQWRTYPSDFGHJKLZXCVBNM])')
            self.trans_vow = re.compile('[aeiouyAEIOUY](?![aeiouyAEIOUY])')
            self.model_file = 'frequency/word-freq-eng.pkl'
            self.nms = 1
            self.nm = 4

        with open(self.model_file, 'rb') as f:
            self.word_freq = pickle.load(f)
        self.doubleletter = re.compile(r'([a-zA-Z])\1')
        self.compiled = re.compile('\.|\,|\'|\"|\(|\)|«|»|’')
        self.models = [SVC(kernel="linear", C=0.025), KNeighborsClassifier(3),
                       SVC(gamma=2, C=1),
                       DecisionTreeClassifier(max_depth=5),
                       RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
                       AdaBoostClassifier(), LogisticRegression(),
                       MLPClassifier(alpha=1, learning_rate='adaptive', max_iter=5000),
                       GaussianNB(), QuadraticDiscriminantAnalysis()][self.nms:self.nm]
        self.names = ["Linear SVM", "Nearest Neighbors", "RBF SVM",
                      "Decision Tree", "Random Forest", "AdaBoost", "Logistic Regression",
                      "Neural Net", "Naive Bayes", "QDA"][self.nms:self.nm]
        print('Number of models:', len(self.models))

    def extract_features(self, words):
        len_chars = len(words) / self.avg_word_length
        len_tokens = len(words.split(' '))

        chars = re.findall(self.lowfreqchar, words)
        lf_sum = 1
        for char in chars:
            lf_sum += self.lowfreqdict[char]

        chars = re.findall(self.highfreqchar, words)
        hf_sum = 1
        for char in chars:
            hf_sum += self.highfreqdict[char]

        num_syll = 0
        for word in words.split():
            try:
                num_syll += self.syllables(word)
            except:
                pass

        num_vowels = len(re.findall(self.vowels, words))

        num_consonants = len(re.findall(self.consonants, words))

        num_mul_vow = len(re.findall(self.mul_vow, words))

        num_mul_cons = len(re.findall(self.mul_cons, words))

        num_double_char = len(re.findall(self.doubleletter, words))

        num_vow_to_cons = re.findall(self.trans_cons, words)
        num_cons_to_vow = re.findall(self.trans_vow, words)
        num_total_trans = len(num_vow_to_cons) + len(num_cons_to_vow)

        words_list = words.lower().split()

        target_freq = 0

        for word in words_list:
            if word in self.word_freq:
                target_freq += self.word_freq[word]
            else:
                target_freq += 0.05

        senses = 0
        synonyms = 0
        hypernyms = 0
        hyponyms = 0
        len_def = 0
        num_POS = 0

        if self.language == 'english':
            for word in words_list:
                word_synset = wn.synsets(word)
                senses += len(word_synset)
                synonyms += sum([len(syn.lemmas()) for syn in word_synset])
                hypernyms += sum([len(syn.hypernyms()) for syn in word_synset])
                hyponyms += sum([len(syn.hyponyms()) for syn in word_synset])
                if senses != 0:
                    len_def += sum([len(syn.definition().split()) for syn in word_synset]) / senses
                num_POS += len(set([syn.pos() for syn in word_synset]))
        if senses == 0:
            for word in words_list:
                try:
                    word_synset = wn.synsets(wn.morphy(word))
                    senses += len(word_synset)  # senses
                    synonyms += sum([len(syn.lemmas()) for syn in word_synset])  # synonyms
                    hypernyms += sum([len(syn.hypernyms()) for syn in word_synset])  # hypernyms
                    hyponyms += sum([len(syn.hyponyms()) for syn in word_synset])  # hyponyms
                    if senses != 0:
                        len_def += sum([len(syn.definition().split()) for syn in word_synset]) / senses
                    num_POS += len(set([syn.pos() for syn in word_synset]))
                except:
                    pass

        features = [len_chars, len_tokens, lf_sum / len_chars, hf_sum / len_chars,
                    num_syll / len_tokens, num_vowels / len_tokens,
                    num_consonants / len_tokens, num_mul_vow / len_tokens,
                    num_mul_cons / len_tokens, num_double_char / len_tokens,
                    num_total_trans / len_chars, target_freq / len_tokens,
                    senses / len_tokens, synonyms / len_tokens, hypernyms / len_tokens,
                    hyponyms / len_tokens, len_def / len_tokens, num_POS / len_tokens]

        return features

    def syllables(self, word):
        count = 0
        if word.lower() in self.syll:
            return [len([y for y in x if isdigit(y[-1])]) for x in self.syll[word.lower()]][0]
        vowels = 'aeiouy'
        word = word.lower().strip(".:;?!")
        if word[0] in vowels:
            count += 1
        for index in range(1,len(word)):
            if word[index] in vowels and word[index-1] not in vowels:
                count += 1
        if word.endswith('e'):
            count -= 1
        if word.endswith('le') and (word[-3] not in vowels):
            count += 1
        if count == 0:
            count += 1
        return count

    def train(self, trainset):
        X = []
        y = []
        i = 0
        for sent in trainset:
            target = re.sub(self.compiled, '', sent['target_word'])
            X.append(self.extract_features(target))
            y.append(sent['gold_label'])

        i = 0
        for model in self.models:
            print('Training: ', self.names[i])
            model.fit(X, y)
            i += 1

    def test_word(self, word):
        M = []
        word = re.sub(self.compiled, '', word)
        M.append(self.extract_features(word))

        predictions = []
        for model in self.models:
            predictions.append(int(model.predict(M)[0]))

        if sum(predictions) > int(len(predictions) / 2):
            return '1'
        return '0'

    def test(self, testset):
        X = []
        M = []
        for sent in testset:
            target = re.sub(self.compiled, '', sent['target_word'])
            M.append(self.extract_features(target))

        i = 0
        for model in self.models:
            X.append((self.names[i], model.predict(M)))
            i += 1
        return X
