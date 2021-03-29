from utils.dataset import Dataset
from utils.scorer import report_score
from utils.wordvec import Word2vec
from random import seed
import numpy as np


def execute_demo_w2v(language):

    data = Dataset(language)
    print('Word2vec based models')
    print('Loading w2v')
    w2v = Word2vec(language)

    print('Training models')
    w2v.train(data.trainset)

    print('Predicting labels')
    predictions_w2v = w2v.test(data.devset)

    predictions_w2v_int = []
    for pred in predictions_w2v:
        pred_int = []
        for val in pred[1]:
            pred_int.append(int(val))
        predictions_w2v_int.append(pred_int)

    gold_labels = [sent['gold_label'] for sent in data.devset]

    print('Calculating scores')
    for pred in predictions_w2v:
        print('Scores for', pred[0])
        report_score(gold_labels, pred[1])

    print('Scores for hard voting with all models')
    avg_pred_w2v_int = np.mean(np.array(predictions_w2v_int), axis=0).tolist()
    avg_pred_w2v = [str(round(val)) for val in avg_pred_w2v_int]
    report_score(gold_labels, avg_pred_w2v)

    return predictions_w2v_int


if __name__ == '__main__':
    seed(100)
    execute_demo_w2v('english')
