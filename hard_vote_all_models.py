from feature_based_solution import execute_demo_feature
from word2vec_solution import execute_demo_w2v
from utils.scorer import report_score
from utils.dataset import Dataset
import numpy as np


def hard_vote(language):

    data = Dataset(language)
    gold_labels = [sent['gold_label'] for sent in data.devset]
    predictions = []
    for pred in execute_demo_feature(language):
        predictions.append(pred)
    for pred in execute_demo_w2v(language):
        predictions.append(pred)
    print('Scores for hard voting with both types of models')
    avg_pred_all_int = np.mean(np.array(predictions), axis=0).tolist()
    avg_pred_all = [str(round(val)) for val in avg_pred_all_int]
    report_score(gold_labels, avg_pred_all)


if __name__ == '__main__':
    hard_vote('english')