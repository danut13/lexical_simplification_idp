import re
import nltk


def cleaner(word):
    word = re.sub('[^a-zA-Z0-9]', ' ', word)
    return word


def simplify_phrase(phrase, baseline, simplifier):
    wordsToBeReplaced = []
    phrase = cleaner(phrase)
    tokenized_text = nltk.word_tokenize(phrase)
    tagged_text = nltk.pos_tag(tokenized_text)
    for word in tagged_text:
        if len(word[0]) > 3 and baseline.test_word(word[0]) == '1' \
                and not bool(re.search(r'\d', word[0])) and word[1] != 'NNP':
            wordsToBeReplaced.append((word[0], simplifier.simplify_phrase(phrase + '.', word[0])))
    return replaced_words(phrase, wordsToBeReplaced)


def replaced_words(phrase, words_to_be_replaced):
    for pair in words_to_be_replaced:
        phrase = re.sub(pair[0], pair[1], phrase)

    return phrase
