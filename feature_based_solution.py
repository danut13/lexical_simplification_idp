from utils.dataset import Dataset
from utils.baseline import Baseline
from utils.scorer import report_score
from utils.simplifier import Simplifier
from utils.datasetsimplifier import simplify_word_by_word
from utils.datasetsimplifier import simplify_whole_phrases
from utils.textsimplifier import simplify_text
from random import seed
import numpy as np


def execute_demo_feature(language, request):
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
    if request == 'baseline':
        return baseline
    if request == 'predictions':
        return predictions_int


if __name__ == '__main__':
    seed(100)
    model = execute_demo_feature('english', 'baseline')
    simplifier = Simplifier()
    text = """The Red Ensign Group is a collaboration of United Kingdom shipping registries including British 
    Overseas Territories and Crown dependencies. It takes its name from the Red Ensign ("Red Duster") flag flown by 
    British civil merchant ships. Its stated purpose is to combine resources to maintain safety and quality across 
    the British fleet. As of 2018 it ranked the ninth largest such group in the world, with approximately 1,
    300 vessels.[1] Sir Alan Massey of the UK Maritime and Coastguard Agency commented: ".. keeping [ships] inside 
    the REG family means that you still have some influence over their quality and performance... We can take 
    administrative measures against members of [it] if we want to so as to ensure that safety is brought up to the 
    necessary standards..."[2] The vessels also receive British Consular assistance and protection.[3] """
    print(simplify_text(text, model, simplifier))

