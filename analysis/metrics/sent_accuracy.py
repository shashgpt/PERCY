import math
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import random
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

def calculate_sent_acc(results_one_rule):
    sent_acc = []
    results_one_rule = pd.DataFrame(results_one_rule)
    sentences = list(results_one_rule['sentence'])
    sent_predictions = list(results_one_rule['sentiment_prediction_output'])
    sent_labels = list(results_one_rule['sentiment_label'])
    rule_labels = list(results_one_rule['rule_label'])
    contrasts = list(results_one_rule['contrast'])
    for index, sentence in enumerate(sentences):
        if sent_predictions[index] != sent_labels[index]:
            sent_acc.append(0)
        elif sent_predictions[index] == sent_labels[index]:
            sent_acc.append(1)

    y_true = sent_labels
    y_pred = sent_predictions
    target_names = ['class 0', 'class 1']
    classification_rpt = classification_report(y_true, y_pred, target_names=target_names, digits=4, output_dict=True)

    weighted_precision = classification_rpt['weighted avg']['precision']
    weighted_recall = classification_rpt['weighted avg']['recall']
    weighted_f1_score = classification_rpt['weighted avg']['f1-score']

    return sent_acc, weighted_precision, weighted_recall, weighted_f1_score