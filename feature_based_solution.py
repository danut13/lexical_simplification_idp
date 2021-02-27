from utils.dataset import Dataset
from utils.baseline import Baseline
from utils.scorer import report_score
from utils.simplifier import Simplifier
from utils.datasetsimplifier import simplify_word_by_word
from utils.datasetsimplifier import simplify_whole_phrases
from utils.textsimplifier import simplify_text
from random import seed
import numpy as np


def execute_demo_feature(language):
    data = Dataset(language)

    print("{}: {} training - {} dev".format(language, len(data.trainset), len(data.devset)))

    print('Feature based models')
    baseline = Baseline(language)

    print('Training models')
    baseline.train(data.trainset)

    print('Predicting labels')
    predictions = baseline.test(data.devset)

    predictions_int = []
    for pred in predictions:
        pred_int = []
        for val in pred[1]:
            pred_int.append(int(val))
        predictions_int.append(pred_int)

    gold_labels = [sent['gold_label'] for sent in data.devset]

    print('Calculating scores')
    for pred in predictions:
        print('Scores for', pred[0])
        report_score(gold_labels, pred[1])

    print('Scores for hard voting with all models')
    avg_pred_int = np.mean(np.array(predictions_int), axis=0).tolist()
    avg_pred = [str(round(val)) for val in avg_pred_int]
    report_score(gold_labels, avg_pred)

    #### SIMPLIFY WORD BY WORD DATASET ####

    # simplify_word_by_word(avg_pred, data.devset)

    #### SIMPLIFY WHOLE PHRASES OF DATASET ####

    # simplify_whole_phrases(data.devset, baseline)

    return baseline


if __name__ == '__main__':
    seed(100)
    model = execute_demo_feature('english')
    simplifier = Simplifier()
    text = """\
The church is an example of red-brick Eclecticism. It has the elements of ancient architecture, 
and of modernism (in particular, large semicircular window openings of the refectory). 
The building has a cruciform appearance, five onion domes covered with tent are strictly proportional. 
The bell tower is attached directly to the church building and is located at its western entrance.
"""
    print(simplify_text(text, model, simplifier))

