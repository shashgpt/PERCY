from functools import update_wrapper
import math
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import random

def calculate_lipschitz_scores_lime(results_one_rule, results_explanations):
    
    # Values to return (corrects distributions)
    filtered_lime_lipschitz_values = []
    
    results_one_rule = pd.DataFrame(results_one_rule)
    results_explanations = pd.DataFrame(results_explanations)
    
    sentences = list(results_one_rule['sentence'])
    sent_predictions = list(results_one_rule['sentiment_prediction_output'])
    sent_labels = list(results_one_rule['sentiment_label'])
    rule_labels = list(results_one_rule['rule_label'])
    contrasts = list(results_one_rule['contrast'])
    explanations = list(results_explanations["LIME_explanation_normalised"])
    lime_lipschitz_values = list(results_explanations["LIME_lipschtiz_value"])
    
    for index, sentence in enumerate(sentences):

        # Select SHAP explanations corresponding to those tokens
        exp = explanations[index]
        
        # Check 1: Sentiment prediction = sentiment label
        if sent_predictions[index] != sent_labels[index]:
            continue
        
        # Check 2: Drop the sentences for which LIME explanation couldn't be calculated
        if lime_lipschitz_values[index] == "couldn't process":
            continue
        
        # Check if A&B conjuncts contains 1 token atleast
        if rule_labels[index] == 1:
            tokenized_sentence = sentence.split()
            rule_word_index = tokenized_sentence.index("but")
            A_conjunct = tokenized_sentence[:rule_word_index]
            B_conjunct = tokenized_sentence[rule_word_index+1:len(tokenized_sentence)]
            A_conjunct_exp = exp[0:rule_word_index]
            B_conjunct_exp = exp[rule_word_index+1:len(tokenized_sentence)]
            if len(A_conjunct) == 0 or len(B_conjunct) == 0 or len(A_conjunct_exp) == 0 or len(B_conjunct_exp)==0:
                continue
        
        # Append lipschitz values
        filtered_lime_lipschitz_values.append(lime_lipschitz_values[index])
        
    return filtered_lime_lipschitz_values

def calculate_lipschitz_scores_shap(results_one_rule, results_explanations):
    
    # Values to return (corrects distributions)
    filtered_lipschitz_values = []
    flag = 0
    results_one_rule = pd.DataFrame(results_one_rule)
    results_explanations = pd.DataFrame(results_explanations)
    
    sentences = list(results_one_rule['sentence'])
    sent_predictions = list(results_one_rule['sentiment_prediction_output'])
    sent_labels = list(results_one_rule['sentiment_label'])
    rule_labels = list(results_one_rule['rule_label'])
    contrasts = list(results_one_rule['contrast'])
    try:
        explanations = list(results_explanations["SHAP_explanation_normalised"])
        lipschitz_values = list(results_explanations["SHAP_lipschtiz_value"])
    except:
        flag = 1
        explanations = list(results_explanations["LIME_explanation_normalised"])
        lipschitz_values = list(results_explanations["LIME_lipschtiz_value"])
    
    for index, sentence in enumerate(sentences):

        # Select SHAP explanations corresponding to those tokens
        exp = explanations[index]
        
        # Check 1: Sentiment prediction = sentiment label
        if sent_predictions[index] != sent_labels[index]:
            continue
            
        # Check 2: Drop the sentences for which LIME explanation couldn't be calculated
        if lipschitz_values[index] == "couldn't process":
            continue
        
        # Check if A&B conjuncts contains 1 token atleast
        if rule_labels[index] == 1:
            tokenized_sentence = sentence.split()
            rule_word_index = tokenized_sentence.index("but")
            A_conjunct = tokenized_sentence[:rule_word_index]
            B_conjunct = tokenized_sentence[rule_word_index+1:len(tokenized_sentence)]
            A_conjunct_exp = exp[0:rule_word_index]
            B_conjunct_exp = exp[rule_word_index+1:len(tokenized_sentence)]
            if len(A_conjunct) == 0 or len(B_conjunct) == 0 or len(A_conjunct_exp) == 0 or len(B_conjunct_exp)==0:
                continue
        
        # Append lipschitz values
        filtered_lipschitz_values.append(lipschitz_values[index])
    
    return filtered_lipschitz_values

def calculate_lipschitz_scores_int_grad(results_one_rule, results_explanations):
    
    # Values to return (corrects distributions)
    filtered_lipschitz_values = []
    flag = 0
    results_one_rule = pd.DataFrame(results_one_rule)
    results_explanations = pd.DataFrame(results_explanations)
    
    sentences = list(results_one_rule['sentence'])
    sent_predictions = list(results_one_rule['sentiment_prediction_output'])
    sent_labels = list(results_one_rule['sentiment_label'])
    rule_labels = list(results_one_rule['rule_label'])
    contrasts = list(results_one_rule['contrast'])
    try:
        explanations = list(results_explanations["INT_GRAD_explanation_normalised"])
        lipschitz_values = list(results_explanations["Int_grad_lipschtiz_value"])
    except:
        flag = 1
        explanations = list(results_explanations["LIME_explanation_normalised"])
        lipschitz_values = list(results_explanations["LIME_lipschtiz_value"])
    
    for index, sentence in enumerate(sentences):
        
        # Select SHAP explanations corresponding to those tokens
        exp = explanations[index]

        # Check 1: Sentiment prediction = sentiment label
        if sent_predictions[index] != sent_labels[index]:
            continue
            
        # Check 2: Drop the sentences for which LIME explanation couldn't be calculated
        if explanations[index] == "couldn't process":
            continue
        
        # Check if A&B conjuncts contains 1 token atleast
        if rule_labels[index] == 1:
            tokenized_sentence = sentence.split()
            rule_word_index = tokenized_sentence.index("but")
            A_conjunct = tokenized_sentence[:rule_word_index]
            B_conjunct = tokenized_sentence[rule_word_index+1:len(tokenized_sentence)]
            A_conjunct_exp = exp[0:rule_word_index]
            B_conjunct_exp = exp[rule_word_index+1:len(tokenized_sentence)]
            if len(A_conjunct) == 0 or len(B_conjunct) == 0 or len(A_conjunct_exp) == 0 or len(B_conjunct_exp)==0:
                continue
        
        # Append lipschitz values
        filtered_lipschitz_values.append(lipschitz_values[index])
    
    return filtered_lipschitz_values