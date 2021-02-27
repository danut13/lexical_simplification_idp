from utils.phrasesimplifier import simplify_phrase
import re


def simplify_text(text, baseline, simplifier):
    simplified_phrase = ""
    sentences = text.split(".")
    for sentence in sentences:
        sentence = sentence.lstrip()
        punctuation_marks_and_spaces = remember_punctuation_marks_and_spaces(sentence)
        simplified_phrase += reconstruct_phrase(simplify_phrase(sentence, baseline, simplifier),
                                                punctuation_marks_and_spaces)
        simplified_phrase += '. '
    return simplified_phrase


def remember_punctuation_marks_and_spaces(phrase):
    to_be_replaced_later = []
    for char in phrase:
        if re.match('[^a-zA-Z0-9]', char):
            to_be_replaced_later.append(char)

    return to_be_replaced_later


def reconstruct_phrase(phrase, to_be_replaced):
    space_counter = 0
    phrase = list(phrase)
    for char_index in range(0, len(phrase)):
        if phrase[char_index].isspace():
            if not to_be_replaced[space_counter].isspace():
                phrase[char_index] = to_be_replaced[space_counter]
            space_counter += 1

    return "".join(phrase)
