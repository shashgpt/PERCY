from functools import update_wrapper
from scipy.stats import ttest_ind
import math
import numpy as np
import pandas as pd
import random

def zero_scores_for_punctuations(tokens, scores):
    for index, token in enumerate(tokens):
        if token == "," or token == '`' or token == "'":
            try:
                scores[index] = 0
            except:
                continue
    return scores

def calculate_lime_acc(results_one_rule, results_explanations, K=5):

    EA_values = []
    EA_values_pearson = []
    a_scores = []
    b_scores = []
    counter = 0
    results_one_rule = pd.DataFrame(results_one_rule)
    results_explanations = pd.DataFrame(results_explanations)
    
    sentences = list(results_one_rule['sentence'])
    sent_predictions = list(results_one_rule['sentiment_prediction_output'])
    sent_labels = list(results_one_rule['sentiment_label'])
    rule_labels = list(results_one_rule['rule_label'])
    contrasts = list(results_one_rule['contrast'])
    try:
        features = list(results_explanations["features"])
    except:
        features = list(results_explanations["sentence"])
    explanations = list(results_explanations["LIME_explanation_normalised"])

    # Anecdotal examples
    corr_sent_pred_wrong_percy_score = {"sentence":[],
                                        "sentiment":[],
                                        "a_conjunct":[],
                                        "b_conjunct":[],
                                        "a_conjunct_explanations":[],
                                        "b_conjunct_explanations":[],
                                        "a_conjunct_score":[],
                                        "b_conjunct_score":[]}
    
    for index, sentence in enumerate(sentences):
        
        # Select LIME and SHAP explanations corresponding to those tokens
        exp = explanations[index]

        # Check 1: sentiment prediction = sentiment label
        if sent_predictions[index] != sent_labels[index]:
            EA_values.append(0)
            EA_values_pearson.append(0)
            continue
        
        # Check 2: Drop the sentences for which LIME explanation couldn't be calculated
        if explanations[index] == "couldn't process":
            EA_values_pearson.append(0)
            continue

        # Check if A&B conjuncts contains 1 token atleast
        tokenized_sentence = sentence.split()
        rule_word_index = tokenized_sentence.index("but")
        A_conjunct = tokenized_sentence[:rule_word_index]
        B_conjunct = tokenized_sentence[rule_word_index+1:len(tokenized_sentence)]
        A_conjunct_exp = exp[0:rule_word_index]
        B_conjunct_exp = exp[rule_word_index+1:len(tokenized_sentence)]
        if len(A_conjunct) == 0 or len(B_conjunct) == 0 or len(A_conjunct_exp) == 0 or len(B_conjunct_exp)==0:
            EA_values_pearson.append(0)
            continue
        
        A_conjunct_selected = []
        B_conjunct_selected = []
        # A_conjunct_exp = zero_scores_for_punctuations(A_conjunct, A_conjunct_exp)
        # B_conjunct_exp = zero_scores_for_punctuations(B_conjunct, B_conjunct_exp)
        A_conjunct_exp_sorted = sorted(A_conjunct_exp, reverse=True) # Sorting tokens in descending order
        B_conjunct_exp_sorted = sorted(B_conjunct_exp, reverse=True)
        A_conjunct_exp_tokens = A_conjunct_exp_sorted[0:K] # select top 5 tokens from A and B for being consistent (reviewers may argue that some sentences might have more no of tokens in A or B in the dataset)
        B_conjunct_exp_tokens = B_conjunct_exp_sorted[0:K]
        for value_index, value in enumerate(A_conjunct_exp_tokens):
            A_conjunct_selected.append(A_conjunct[A_conjunct_exp.index(value)])
        for value_index, value in enumerate(B_conjunct_exp_tokens):
            B_conjunct_selected.append(B_conjunct[B_conjunct_exp.index(value)])
        p_value = ttest_ind(A_conjunct_exp_tokens, B_conjunct_exp_tokens)[1] # Pvalue test to reject the null hypothesis (How does it apply in LIME-scores?)

        # scores = []
        # for i, score in enumerate(A_conjunct_exp):
        #     if score in A_conjunct_exp_tokens:
        #         scores.append(score)
        #     else:
        #         scores.append(0.0)
        # scores.append(0.0)
        # for i, score in enumerate(B_conjunct_exp):
        #     if score in B_conjunct_exp_tokens:
        #         scores.append(score)
        #     else:
        #         scores.append(0.0)
        # scores = A_conjunct_exp + [exp[rule_word_index]] + B_conjunct_exp
        scores = exp

        if np.mean(A_conjunct_exp_tokens) < np.mean(B_conjunct_exp_tokens) and p_value < 0.05: # Check if both sum amd max are consistent
            # print("\n")
            # print("sentence: ", sentence)
            # print("sentiment: ", sent_labels[index])
            # print("A conjunct: ", A_conjunct_selected)
            # print("B conjunct: ", B_conjunct_selected)
            # print("A conjunct explanations: ", A_conjunct_exp_tokens)
            # print("B conjunct explanations: ", B_conjunct_exp_tokens)
            # print("A conjunct score: ", np.mean(A_conjunct_exp_tokens))
            # print("B conjunct score: ", np.mean(B_conjunct_exp_tokens))
            # print("features: ", features[index])
            # print("no of features: ", len(features[index]))
            # print("Scores: ", scores)
            # print("no of scores: ", len(scores))
            # print("\n")
            EA_value = 1
            EA_values.append(EA_value)
            EA_values_pearson.append(EA_value)
            a_scores.append(np.mean(A_conjunct_exp_tokens))
            b_scores.append(np.mean(B_conjunct_exp_tokens))

        # elif np.mean(A_conjunct_exp_tokens) > np.mean(B_conjunct_exp_tokens) and p_value < 0.05:
                # print("\n")
                # print("sentence: ", sentence)
                # print("sentiment: ", sent_labels[index])
                # print("A conjunct: ", A_conjunct_selected)
                # print("B conjunct: ", B_conjunct_selected)
                # print("A conjunct explanations: ", A_conjunct_exp_tokens)
                # print("B conjunct explanations: ", B_conjunct_exp_tokens)
                # print("A conjunct score: ", np.mean(A_conjunct_exp_tokens))
                # print("B conjunct score: ", np.mean(B_conjunct_exp_tokens))
                # print("features: ", features[index])
                # print("no of features: ", len(features[index]))
                # print("Scores: ", scores)
                # print("no of scores: ", len(scores))
                # print("\n")
                # EA_value = 0
                # EA_values.append(EA_value)
        # elif np.mean(A_conjunct_exp_tokens) > np.mean(B_conjunct_exp_tokens) and np.max(A_conjunct_exp_tokens) < np.max(B_conjunct_exp_tokens):
        #     print("\n")
        #     print("sentence: ", sentence)
        #     print("sentiment: ", sent_labels[index])
        #     print("A conjunct: ", A_conjunct_selected)
        #     print("B conjunct: ", B_conjunct_selected)
        #     print("A conjunct explanations: ", A_conjunct_exp_tokens)
        #     print("B conjunct explanations: ", B_conjunct_exp_tokens)
        #     print("A conjunct score: ", np.mean(A_conjunct_exp_tokens))
        #     print("B conjunct score: ", np.mean(B_conjunct_exp_tokens))
        #     print("features: ", features[index])
        #     print("no of features: ", len(features[index]))
        #     print("Scores: ", scores)
        #     print("no of scores: ", len(scores))
        #     print("\n")
        else:
            EA_value = 0
            EA_values.append(EA_value)
            EA_values_pearson.append(EA_value)

    return EA_values, EA_values_pearson, a_scores, b_scores