from utils.simplifier import Simplifier
from utils.phrasesimplifier import simplify_phrase
import csv


def simplify_whole_phrases(data, baseline):
    simplifier = Simplifier()
    phrases = make_phrases_dataset(data.devset)
    simplified = []
    i = 0
    for sentence in phrases:
        print(i)
        simplified.append("")
        sentences = sentence.split('~')
        for s in sentences:
            print(s)
            simplified[i] += (simplify_phrase(s, baseline, simplifier))
            simplified[i] += '. '
        i += 1

    file_path = "datasets/simplified/English_all.tsv"
    with open(file_path, 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for j in range(0, len(phrases)):
            tsv_writer.writerow([phrases[j]])
            tsv_writer.writerow([simplified[j]])
            tsv_writer.writerow([" "])


def make_phrases_dataset(data):
    previous_phrase = ""
    previous_hit_id = ""
    sentences = []
    i = -1
    for sent in data:
        phrase = sent['sentence']
        hit_id = sent['hit_id']
        if phrase != previous_phrase and hit_id == previous_hit_id:
            sentences[i] += "~"
            sentences[i] += phrase
        elif phrase != previous_phrase and hit_id != previous_hit_id:
            i += 1
            sentences.append("")
            sentences[i] += phrase
        previous_phrase = phrase
        previous_hit_id = hit_id
    return sentences


def simplify_word_by_word(prediction, data):
    simplifier = Simplifier()
    write_dict = []
    for index in range(0, len(prediction)):
        if prediction[index] == '1':
            target = data[index]['target_word']
            simplified_phrase = simplifier.simplify_phrase(data[index]['sentence'], target)
            if data[index]['gold_label'] == '1':
                write_dict.append(
                    {"original": data[index]['sentence'], "simplified": simplified_phrase, "complex_word": target
                        , "positive": "true"}
                )
            else:
                write_dict.append(
                    {"original": data[index]['sentence'], "simplified": simplified_phrase, "complex_word": target,
                     "positive": "negative"}
                )
    write_to_simplified(write_dict)


def write_to_simplified(dict_write):
    file_path = "datasets/simplified/English.tsv"
    with open(file_path, 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for row in dict_write:
            tsv_writer.writerow([row['original'], row['complex_word']])
            tsv_writer.writerow([row['simplified'], row["positive"]])
            tsv_writer.writerow([" "])
