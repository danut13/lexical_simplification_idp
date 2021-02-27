import re


def cleaner(word):
    #word = re.sub('[\W]', ' ', word)
    word = re.sub('[^a-zA-Z0-9]', ' ', word)
    return word.strip()


def simplify_phrase(phrase, baseline, simplifier):
    wordsToBeReplaced = []
    phrase = cleaner(phrase)
    for word in phrase.split():
        if len(word) > 4 and baseline.test_word(word) == '1' and not bool(re.search(r'\d', word)):
            wordsToBeReplaced.append((word, simplifier.simplify_phrase(phrase+'.', word)))
    return replaced_words(phrase, wordsToBeReplaced)


def replaced_words(phrase, words_to_be_replaced):
    for pair in words_to_be_replaced:
        phrase = re.sub(pair[0], pair[1], phrase)

    return phrase
