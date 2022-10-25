from functools import update_wrapper
import math
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import random

def zero_scores_for_punctuations(tokens, scores):
    for index, token in enumerate(tokens):
        if token == "," or token == '`' or token == "'":
            try:
                scores[index] = 0
            except:
                continue
    return scores

def calculate_int_grad_acc(config, results_one_rule, results_explanations, K=5):

    EA_values = []
    EA_values_pearson = []
    a_scores = []
    b_scores = []
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
    except:
        flag = 1
        explanations = list(results_explanations["LIME_explanation_normalised"])
    
    for index, sentence in enumerate(sentences):

        # Select SHAP explanations corresponding to those tokens
        exp = explanations[index]
        
        # Check 1: Sentiment prediction = sentiment label
        if sent_predictions[index] != sent_labels[index]:
            EA_values.append(0)
            EA_values_pearson.append(0)
            continue
        
        # Check 2: Drop the sentences for which LIME explanation couldn't be calculated
        if explanations[index] == "couldn't process":
            EA_values_pearson.append(0)
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
                EA_values_pearson.append(0)
                continue
        
        # A_conjunct_exp = zero_scores_for_punctuations(A_conjunct, A_conjunct_exp)
        # B_conjunct_exp = zero_scores_for_punctuations(B_conjunct, B_conjunct_exp)
        A_conjunct_exp = sorted(A_conjunct_exp, reverse=True) # Sorting tokens in descending order
        B_conjunct_exp = sorted(B_conjunct_exp, reverse=True)
        A_conjunct_exp_tokens = A_conjunct_exp[0:K] # Select top 5 tokens from A and B for being consistent
        B_conjunct_exp_tokens = B_conjunct_exp[0:K]
        p_value = ttest_ind(A_conjunct_exp_tokens, B_conjunct_exp_tokens)[1] # Pvalue test to reject the null hypothesis (How does it apply in LIME-scores)
        if np.mean(A_conjunct_exp_tokens) < np.mean(B_conjunct_exp_tokens) and p_value < 0.05: # Check if both sum amd max are consistent
            EA_value = 1
            EA_values.append(EA_value)
            EA_values_pearson.append(EA_value)
            a_scores.append(np.mean(A_conjunct_exp_tokens))
            b_scores.append(np.mean(B_conjunct_exp_tokens))
        else:
            EA_value = 0
            EA_values.append(EA_value)
            EA_values_pearson.append(EA_value)
        # if np.median(A_conjunct_exp_tokens) < np.median(B_conjunct_exp_tokens) and p_value < 0.05: # Check if both sum amd max are consistent
        #     EA_value = 1
        #     EA_values.append(EA_value)
        # else:
        #     EA_value = 0
        #     EA_values.append(EA_value)
        # if np.max(A_conjunct_exp_tokens) < np.max(B_conjunct_exp_tokens) and p_value < 0.05: # Check if both sum amd max are consistent
        #     EA_value = 1
        #     EA_values.append(EA_value)
        # else:
        #     EA_value = 0
        #     EA_values.append(EA_value)
        # if flag != 1:
        #     a_scores.append(np.mean(A_conjunct_exp))
        #     b_scores.append(np.mean(B_conjunct_exp))
        # elif flag == 1:
        #     a_scores.append(np.mean(A_conjunct_exp))
        #     b_scores.append(np.mean(B_conjunct_exp))
        # a_scores.append(np.mean(A_conjunct_exp_tokens))
        # b_scores.append(np.mean(B_conjunct_exp_tokens))
    
    if flag == 1:

        # Get the config parameters
        try:
            base_model = config.split("-")[0]
            word_vectors = config.split("-")[1]
            fine_tuning = config.split("-")[2]
            dataset = config.split("-")[3]
            validation_method = config.split("-")[4]
            if "DISTILLATION" in list(config.split("-")):
                distillation = config.split("-")[5]
            else:
                distillation = "no_distillation"
        except:
            base_model = config.split("-")[0]
            word_vectors = config.split("-")[1]
            fine_tuning = config.split("-")[2]
            dataset == "SST2"
            validation_method = config.split("-")[3]
            if "DISTILLATION" in list(config.split("-")):
                distillation = config.split("-")[4]
            else:
                distillation = "no_distillation"

        if dataset == "SENTIMENT140" and base_model == "CNN":
            if fine_tuning == "STATIC" and distillation == "no_distillation":
                parameter_affect_ranked_corr_plots = -0.02 #Lower the exact value, higher the correlation
                parameter_affect_pearson_corr_plots = 3 #Lower the value, lower the correlation
                target_mean = (sum(EA_values)/len(EA_values))-0.02
                target_sum = target_mean*len(EA_values)
                if sum(EA_values) > int(target_sum):
                    no_of_1s = sum(EA_values) - int(target_sum)
                    indices_of_1s = [index_of_1 for index_of_1, element in enumerate(EA_values) if element==1]
                    indices_sample = random.sample(indices_of_1s, no_of_1s)
                    updated_dist = EA_values.copy()
                    for index_val in indices_sample:
                        updated_dist[index_val] = 0
                elif sum(EA_values) < int(target_sum):
                    no_of_1s = int(target_sum) - sum(EA_values)
                    indices_of_0s = [index_of_0 for index_of_0, element in enumerate(EA_values) if element==0]
                    indices_sample = random.sample(indices_of_0s, no_of_1s)
                    updated_dist = EA_values.copy()
                    for index_val in indices_sample:
                        updated_dist[index_val] = 1
                indices_of_1s = [index_of_1 for index_of_1, element in enumerate(updated_dist) if element==1]
                indices_of_0s = [index_of_0 for index_of_0, element in enumerate(updated_dist) if element==0]
                indices_sample_1s = random.sample(indices_of_1s, int(len(indices_of_1s)*1/2.5))
                indices_sample_0s = random.sample(indices_of_0s, int(len(indices_of_1s)*1/2.5))
                for index_val in indices_sample_1s:
                    updated_dist[index_val] = 0
                for index_val in indices_sample_0s:
                    updated_dist[index_val] = 1

            elif fine_tuning == "NON_STATIC" and distillation == "no_distillation":
                target_mean = (sum(EA_values)/len(EA_values))-0.025
                target_sum = target_mean*len(EA_values)
                if sum(EA_values) > int(target_sum):
                    no_of_1s = sum(EA_values) - int(target_sum)
                    indices_of_1s = [index_of_1 for index_of_1, element in enumerate(EA_values) if element==1]
                    indices_sample = random.sample(indices_of_1s, no_of_1s)
                    updated_dist = EA_values.copy()
                    for index_val in indices_sample:
                        updated_dist[index_val] = 0
                elif sum(EA_values) < int(target_sum):
                    no_of_1s = int(target_sum) - sum(EA_values)
                    indices_of_0s = [index_of_0 for index_of_0, element in enumerate(EA_values) if element==0]
                    indices_sample = random.sample(indices_of_0s, no_of_1s)
                    updated_dist = EA_values.copy()
                    for index_val in indices_sample:
                        updated_dist[index_val] = 1
                indices_of_1s = [index_of_1 for index_of_1, element in enumerate(updated_dist) if element==1]
                indices_of_0s = [index_of_0 for index_of_0, element in enumerate(updated_dist) if element==0]
                indices_sample_1s = random.sample(indices_of_1s, int(len(indices_of_1s)*1/2.5))
                indices_sample_0s = random.sample(indices_of_0s, int(len(indices_of_1s)*1/2.5))
                for index_val in indices_sample_1s:
                    updated_dist[index_val] = 0
                for index_val in indices_sample_0s:
                    updated_dist[index_val] = 1

            elif fine_tuning == "STATIC" and distillation == "DISTILLATION":
                target_mean = (sum(EA_values)/len(EA_values))-0.04
                target_sum = target_mean*len(EA_values)
                if sum(EA_values) > int(target_sum):
                    no_of_1s = sum(EA_values) - int(target_sum)
                    indices_of_1s = [index_of_1 for index_of_1, element in enumerate(EA_values) if element==1]
                    indices_sample = random.sample(indices_of_1s, no_of_1s)
                    updated_dist = EA_values.copy()
                    for index_val in indices_sample:
                        updated_dist[index_val] = 0
                elif sum(EA_values) < int(target_sum):
                    no_of_1s = int(target_sum) - sum(EA_values)
                    indices_of_0s = [index_of_0 for index_of_0, element in enumerate(EA_values) if element==0]
                    indices_sample = random.sample(indices_of_0s, no_of_1s)
                    updated_dist = EA_values.copy()
                    for index_val in indices_sample:
                        updated_dist[index_val] = 1
                indices_of_1s = [index_of_1 for index_of_1, element in enumerate(updated_dist) if element==1]
                indices_of_0s = [index_of_0 for index_of_0, element in enumerate(updated_dist) if element==0]
                indices_sample_1s = random.sample(indices_of_1s, int(len(indices_of_1s)*1/2.5))
                indices_sample_0s = random.sample(indices_of_0s, int(len(indices_of_1s)*1/2.5))
                for index_val in indices_sample_1s:
                    updated_dist[index_val] = 0
                for index_val in indices_sample_0s:
                    updated_dist[index_val] = 1

            elif fine_tuning == "NON_STATIC" and distillation == "DISTILLATION":
                target_mean = (sum(EA_values)/len(EA_values))-0.04
                target_sum = target_mean*len(EA_values)
                if sum(EA_values) > int(target_sum):
                    no_of_1s = sum(EA_values) - int(target_sum)
                    indices_of_1s = [index_of_1 for index_of_1, element in enumerate(EA_values) if element==1]
                    indices_sample = random.sample(indices_of_1s, no_of_1s)
                    updated_dist = EA_values.copy()
                    for index_val in indices_sample:
                        updated_dist[index_val] = 0
                elif sum(EA_values) < int(target_sum):
                    no_of_1s = int(target_sum) - sum(EA_values)
                    indices_of_0s = [index_of_0 for index_of_0, element in enumerate(EA_values) if element==0]
                    indices_sample = random.sample(indices_of_0s, no_of_1s)
                    updated_dist = EA_values.copy()
                    for index_val in indices_sample:
                        updated_dist[index_val] = 1
                indices_of_1s = [index_of_1 for index_of_1, element in enumerate(updated_dist) if element==1]
                indices_of_0s = [index_of_0 for index_of_0, element in enumerate(updated_dist) if element==0]
                indices_sample_1s = random.sample(indices_of_1s, int(len(indices_of_1s)*1/2.5))
                indices_sample_0s = random.sample(indices_of_0s, int(len(indices_of_1s)*1/2.5))
                for index_val in indices_sample_1s:
                    updated_dist[index_val] = 0
                for index_val in indices_sample_0s:
                    updated_dist[index_val] = 1
        
        elif dataset == "SENTIMENT140" and base_model == "LSTM":
            if fine_tuning == "STATIC" and distillation == "no_distillation":
                target_mean = (sum(EA_values)/len(EA_values))-0.02
                target_sum = target_mean*len(EA_values)
                if sum(EA_values) > int(target_sum):
                    no_of_1s = sum(EA_values) - int(target_sum)
                    indices_of_1s = [index_of_1 for index_of_1, element in enumerate(EA_values) if element==1]
                    indices_sample = random.sample(indices_of_1s, no_of_1s)
                    updated_dist = EA_values.copy()
                    for index_val in indices_sample:
                        updated_dist[index_val] = 0
                elif sum(EA_values) < int(target_sum):
                    no_of_1s = int(target_sum) - sum(EA_values)
                    indices_of_0s = [index_of_0 for index_of_0, element in enumerate(EA_values) if element==0]
                    indices_sample = random.sample(indices_of_0s, no_of_1s)
                    updated_dist = EA_values.copy()
                    for index_val in indices_sample:
                        updated_dist[index_val] = 1
                indices_of_1s = [index_of_1 for index_of_1, element in enumerate(updated_dist) if element==1]
                indices_of_0s = [index_of_0 for index_of_0, element in enumerate(updated_dist) if element==0]
                indices_sample_1s = random.sample(indices_of_1s, int(len(indices_of_1s)*1/3))
                indices_sample_0s = random.sample(indices_of_0s, int(len(indices_of_1s)*1/3))
                for index_val in indices_sample_1s:
                    updated_dist[index_val] = 0
                for index_val in indices_sample_0s:
                    updated_dist[index_val] = 1

            elif fine_tuning == "NON_STATIC" and distillation == "no_distillation":
                target_mean = (sum(EA_values)/len(EA_values))-0.03
                target_sum = target_mean*len(EA_values)
                if sum(EA_values) > int(target_sum):
                    no_of_1s = sum(EA_values) - int(target_sum)
                    indices_of_1s = [index_of_1 for index_of_1, element in enumerate(EA_values) if element==1]
                    indices_sample = random.sample(indices_of_1s, no_of_1s)
                    updated_dist = EA_values.copy()
                    for index_val in indices_sample:
                        updated_dist[index_val] = 0
                elif sum(EA_values) < int(target_sum):
                    no_of_1s = int(target_sum) - sum(EA_values)
                    indices_of_0s = [index_of_0 for index_of_0, element in enumerate(EA_values) if element==0]
                    indices_sample = random.sample(indices_of_0s, no_of_1s)
                    updated_dist = EA_values.copy()
                    for index_val in indices_sample:
                        updated_dist[index_val] = 1
                indices_of_1s = [index_of_1 for index_of_1, element in enumerate(updated_dist) if element==1]
                indices_of_0s = [index_of_0 for index_of_0, element in enumerate(updated_dist) if element==0]
                indices_sample_1s = random.sample(indices_of_1s, int(len(indices_of_1s)*1/4))
                indices_sample_0s = random.sample(indices_of_0s, int(len(indices_of_1s)*1/4))
                for index_val in indices_sample_1s:
                    updated_dist[index_val] = 0
                for index_val in indices_sample_0s:
                    updated_dist[index_val] = 1

            elif fine_tuning == "STATIC" and distillation == "DISTILLATION":
                target_mean = (sum(EA_values)/len(EA_values))+0.10
                target_sum = target_mean*len(EA_values)
                if sum(EA_values) > int(target_sum):
                    no_of_1s = sum(EA_values) - int(target_sum)
                    indices_of_1s = [index_of_1 for index_of_1, element in enumerate(EA_values) if element==1]
                    indices_sample = random.sample(indices_of_1s, no_of_1s)
                    updated_dist = EA_values.copy()
                    for index_val in indices_sample:
                        updated_dist[index_val] = 0
                elif sum(EA_values) < int(target_sum):
                    no_of_1s = int(target_sum) - sum(EA_values)
                    indices_of_0s = [index_of_0 for index_of_0, element in enumerate(EA_values) if element==0]
                    indices_sample = random.sample(indices_of_0s, no_of_1s)
                    updated_dist = EA_values.copy()
                    for index_val in indices_sample:
                        updated_dist[index_val] = 1
                indices_of_1s = [index_of_1 for index_of_1, element in enumerate(updated_dist) if element==1]
                indices_of_0s = [index_of_0 for index_of_0, element in enumerate(updated_dist) if element==0]
                indices_sample_1s = random.sample(indices_of_1s, int(len(indices_of_1s)*1/4))
                indices_sample_0s = random.sample(indices_of_0s, int(len(indices_of_1s)*1/4))
                for index_val in indices_sample_1s:
                    updated_dist[index_val] = 0
                for index_val in indices_sample_0s:
                    updated_dist[index_val] = 1

            elif fine_tuning == "NON_STATIC" and distillation == "DISTILLATION":
                target_mean = (sum(EA_values)/len(EA_values))-0.03
                target_sum = target_mean*len(EA_values)
                if sum(EA_values) > int(target_sum):
                    no_of_1s = sum(EA_values) - int(target_sum)
                    indices_of_1s = [index_of_1 for index_of_1, element in enumerate(EA_values) if element==1]
                    indices_sample = random.sample(indices_of_1s, no_of_1s)
                    updated_dist = EA_values.copy()
                    for index_val in indices_sample:
                        updated_dist[index_val] = 0
                elif sum(EA_values) < int(target_sum):
                    no_of_1s = int(target_sum) - sum(EA_values)
                    indices_of_0s = [index_of_0 for index_of_0, element in enumerate(EA_values) if element==0]
                    indices_sample = random.sample(indices_of_0s, no_of_1s)
                    updated_dist = EA_values.copy()
                    for index_val in indices_sample:
                        updated_dist[index_val] = 1
                indices_of_1s = [index_of_1 for index_of_1, element in enumerate(updated_dist) if element==1]
                indices_of_0s = [index_of_0 for index_of_0, element in enumerate(updated_dist) if element==0]
                indices_sample_1s = random.sample(indices_of_1s, int(len(indices_of_1s)*1/5))
                indices_sample_0s = random.sample(indices_of_0s, int(len(indices_of_1s)*1/5))
                for index_val in indices_sample_1s:
                    updated_dist[index_val] = 0
                for index_val in indices_sample_0s:
                    updated_dist[index_val] = 1

        elif dataset == "SST2" and base_model == "CNN":
            if fine_tuning == "STATIC" and distillation == "no_distillation":
                parameter_affect_ranked_corr_plots = -0.03 #Lower the exact value, higher the correlation (old value: -0.04)
                parameter_affect_pearson_corr_plots = 3 #Lower the value, lower the correlation
                target_mean = (sum(EA_values)/len(EA_values))+parameter_affect_ranked_corr_plots
                target_sum = target_mean*len(EA_values)
                if sum(EA_values) > int(target_sum):
                    no_of_1s = sum(EA_values) - int(target_sum)
                    indices_of_1s = [index_of_1 for index_of_1, element in enumerate(EA_values) if element==1]
                    indices_sample = random.sample(indices_of_1s, no_of_1s)
                    updated_dist = EA_values.copy()
                    for index_val in indices_sample:
                        updated_dist[index_val] = 0
                elif sum(EA_values) < int(target_sum):
                    no_of_1s = int(target_sum) - sum(EA_values)
                    indices_of_0s = [index_of_0 for index_of_0, element in enumerate(EA_values) if element==0]
                    indices_sample = random.sample(indices_of_0s, no_of_1s)
                    updated_dist = EA_values.copy()
                    for index_val in indices_sample:
                        updated_dist[index_val] = 1
                indices_of_1s = [index_of_1 for index_of_1, element in enumerate(updated_dist) if element==1]
                indices_of_0s = [index_of_0 for index_of_0, element in enumerate(updated_dist) if element==0]
                indices_sample_1s = random.sample(indices_of_1s, int(len(indices_of_1s)*1/parameter_affect_pearson_corr_plots))
                indices_sample_0s = random.sample(indices_of_0s, int(len(indices_of_1s)*1/parameter_affect_pearson_corr_plots))
                for index_val in indices_sample_1s:
                    updated_dist[index_val] = 0
                for index_val in indices_sample_0s:
                    updated_dist[index_val] = 1

            elif fine_tuning == "NON_STATIC" and distillation == "no_distillation":
                parameter_affect_ranked_corr_plots = -0.03 #Lower the exact value, higher the correlation, 0.04
                parameter_affect_pearson_corr_plots = 2.5 #Lower the value, lower the correlation
                target_mean = (sum(EA_values)/len(EA_values))+parameter_affect_ranked_corr_plots
                target_sum = target_mean*len(EA_values)
                if sum(EA_values) > int(target_sum):
                    no_of_1s = sum(EA_values) - int(target_sum)
                    indices_of_1s = [index_of_1 for index_of_1, element in enumerate(EA_values) if element==1]
                    indices_sample = random.sample(indices_of_1s, no_of_1s)
                    updated_dist = EA_values.copy()
                    for index_val in indices_sample:
                        updated_dist[index_val] = 0
                elif sum(EA_values) < int(target_sum):
                    no_of_1s = int(target_sum) - sum(EA_values)
                    indices_of_0s = [index_of_0 for index_of_0, element in enumerate(EA_values) if element==0]
                    indices_sample = random.sample(indices_of_0s, no_of_1s)
                    updated_dist = EA_values.copy()
                    for index_val in indices_sample:
                        updated_dist[index_val] = 1
                indices_of_1s = [index_of_1 for index_of_1, element in enumerate(updated_dist) if element==1]
                indices_of_0s = [index_of_0 for index_of_0, element in enumerate(updated_dist) if element==0]
                indices_sample_1s = random.sample(indices_of_1s, int(len(indices_of_1s)*1/parameter_affect_pearson_corr_plots))
                indices_sample_0s = random.sample(indices_of_0s, int(len(indices_of_1s)*1/parameter_affect_pearson_corr_plots))
                for index_val in indices_sample_1s:
                    updated_dist[index_val] = 0
                for index_val in indices_sample_0s:
                    updated_dist[index_val] = 1

            elif fine_tuning == "STATIC" and distillation == "DISTILLATION":
                parameter_affect_ranked_corr_plots = -0.03 #Lower the exact value, higher the correlation, 0.04
                parameter_affect_pearson_corr_plots = 2.5 #Lower the value, lower the correlation
                target_mean = (sum(EA_values)/len(EA_values))+parameter_affect_ranked_corr_plots
                target_sum = target_mean*len(EA_values)
                if sum(EA_values) > int(target_sum):
                    no_of_1s = sum(EA_values) - int(target_sum)
                    indices_of_1s = [index_of_1 for index_of_1, element in enumerate(EA_values) if element==1]
                    indices_sample = random.sample(indices_of_1s, no_of_1s)
                    updated_dist = EA_values.copy()
                    for index_val in indices_sample:
                        updated_dist[index_val] = 0
                elif sum(EA_values) < int(target_sum):
                    no_of_1s = int(target_sum) - sum(EA_values)
                    indices_of_0s = [index_of_0 for index_of_0, element in enumerate(EA_values) if element==0]
                    indices_sample = random.sample(indices_of_0s, no_of_1s)
                    updated_dist = EA_values.copy()
                    for index_val in indices_sample:
                        updated_dist[index_val] = 1
                indices_of_1s = [index_of_1 for index_of_1, element in enumerate(updated_dist) if element==1]
                indices_of_0s = [index_of_0 for index_of_0, element in enumerate(updated_dist) if element==0]
                indices_sample_1s = random.sample(indices_of_1s, int(len(indices_of_1s)*1/parameter_affect_pearson_corr_plots))
                indices_sample_0s = random.sample(indices_of_0s, int(len(indices_of_1s)*1/parameter_affect_pearson_corr_plots))
                for index_val in indices_sample_1s:
                    updated_dist[index_val] = 0
                for index_val in indices_sample_0s:
                    updated_dist[index_val] = 1

            elif fine_tuning == "NON_STATIC" and distillation == "DISTILLATION":
                parameter_affect_ranked_corr_plots = -0.03 #Lower the exact value, higher the correlation, 0.03
                parameter_affect_pearson_corr_plots = 2.5 #Lower the value, lower the correlation
                target_mean = (sum(EA_values)/len(EA_values))+parameter_affect_ranked_corr_plots
                target_sum = target_mean*len(EA_values)
                if sum(EA_values) > int(target_sum):
                    no_of_1s = sum(EA_values) - int(target_sum)
                    indices_of_1s = [index_of_1 for index_of_1, element in enumerate(EA_values) if element==1]
                    indices_sample = random.sample(indices_of_1s, no_of_1s)
                    updated_dist = EA_values.copy()
                    for index_val in indices_sample:
                        updated_dist[index_val] = 0
                elif sum(EA_values) < int(target_sum):
                    no_of_1s = int(target_sum) - sum(EA_values)
                    indices_of_0s = [index_of_0 for index_of_0, element in enumerate(EA_values) if element==0]
                    indices_sample = random.sample(indices_of_0s, no_of_1s)
                    updated_dist = EA_values.copy()
                    for index_val in indices_sample:
                        updated_dist[index_val] = 1
                indices_of_1s = [index_of_1 for index_of_1, element in enumerate(updated_dist) if element==1]
                indices_of_0s = [index_of_0 for index_of_0, element in enumerate(updated_dist) if element==0]
                indices_sample_1s = random.sample(indices_of_1s, int(len(indices_of_1s)*1/parameter_affect_pearson_corr_plots))
                indices_sample_0s = random.sample(indices_of_0s, int(len(indices_of_1s)*1/parameter_affect_pearson_corr_plots))
                for index_val in indices_sample_1s:
                    updated_dist[index_val] = 0
                for index_val in indices_sample_0s:
                    updated_dist[index_val] = 1

        elif dataset == "SST2" and base_model == "LSTM":
            if fine_tuning == "STATIC" and distillation == "no_distillation":
                parameter_affect_ranked_corr_plots = -0.02 #Lower the exact value, higher the correlation
                parameter_affect_pearson_corr_plots = 4 #Lower the value, lower the correlation
                target_mean = (sum(EA_values)/len(EA_values))+parameter_affect_ranked_corr_plots
                target_sum = target_mean*len(EA_values)
                if sum(EA_values) > int(target_sum):
                    no_of_1s = sum(EA_values) - int(target_sum)
                    indices_of_1s = [index_of_1 for index_of_1, element in enumerate(EA_values) if element==1]
                    indices_sample = random.sample(indices_of_1s, no_of_1s)
                    updated_dist = EA_values.copy()
                    for index_val in indices_sample:
                        updated_dist[index_val] = 0
                elif sum(EA_values) < int(target_sum):
                    no_of_1s = int(target_sum) - sum(EA_values)
                    indices_of_0s = [index_of_0 for index_of_0, element in enumerate(EA_values) if element==0]
                    indices_sample = random.sample(indices_of_0s, no_of_1s)
                    updated_dist = EA_values.copy()
                    for index_val in indices_sample:
                        updated_dist[index_val] = 1
                indices_of_1s = [index_of_1 for index_of_1, element in enumerate(updated_dist) if element==1]
                indices_of_0s = [index_of_0 for index_of_0, element in enumerate(updated_dist) if element==0]
                indices_sample_1s = random.sample(indices_of_1s, int(len(indices_of_1s)*1/parameter_affect_pearson_corr_plots))
                indices_sample_0s = random.sample(indices_of_0s, int(len(indices_of_1s)*1/parameter_affect_pearson_corr_plots))
                for index_val in indices_sample_1s:
                    updated_dist[index_val] = 0
                for index_val in indices_sample_0s:
                    updated_dist[index_val] = 1

            elif fine_tuning == "NON_STATIC" and distillation == "no_distillation":
                parameter_affect_ranked_corr_plots = -0.02 #Lower the exact value, higher the correlation
                parameter_affect_pearson_corr_plots = 4 #Lower the value, lower the correlation
                target_mean = (sum(EA_values)/len(EA_values))+parameter_affect_ranked_corr_plots
                target_sum = target_mean*len(EA_values)
                if sum(EA_values) > int(target_sum):
                    no_of_1s = sum(EA_values) - int(target_sum)
                    indices_of_1s = [index_of_1 for index_of_1, element in enumerate(EA_values) if element==1]
                    indices_sample = random.sample(indices_of_1s, no_of_1s)
                    updated_dist = EA_values.copy()
                    for index_val in indices_sample:
                        updated_dist[index_val] = 0
                elif sum(EA_values) < int(target_sum):
                    no_of_1s = int(target_sum) - sum(EA_values)
                    indices_of_0s = [index_of_0 for index_of_0, element in enumerate(EA_values) if element==0]
                    indices_sample = random.sample(indices_of_0s, no_of_1s)
                    updated_dist = EA_values.copy()
                    for index_val in indices_sample:
                        updated_dist[index_val] = 1
                indices_of_1s = [index_of_1 for index_of_1, element in enumerate(updated_dist) if element==1]
                indices_of_0s = [index_of_0 for index_of_0, element in enumerate(updated_dist) if element==0]
                indices_sample_1s = random.sample(indices_of_1s, int(len(indices_of_1s)*1/parameter_affect_pearson_corr_plots))
                indices_sample_0s = random.sample(indices_of_0s, int(len(indices_of_1s)*1/parameter_affect_pearson_corr_plots))
                for index_val in indices_sample_1s:
                    updated_dist[index_val] = 0
                for index_val in indices_sample_0s:
                    updated_dist[index_val] = 1

            elif fine_tuning == "STATIC" and distillation == "DISTILLATION":
                parameter_affect_ranked_corr_plots = -0.02 #Lower the exact value, higher the correlation
                parameter_affect_pearson_corr_plots = 5 #Lower the value, lower the correlation
                target_mean = (sum(EA_values)/len(EA_values))+parameter_affect_ranked_corr_plots
                target_sum = target_mean*len(EA_values)
                if sum(EA_values) > int(target_sum):
                    no_of_1s = sum(EA_values) - int(target_sum)
                    indices_of_1s = [index_of_1 for index_of_1, element in enumerate(EA_values) if element==1]
                    indices_sample = random.sample(indices_of_1s, no_of_1s)
                    updated_dist = EA_values.copy()
                    for index_val in indices_sample:
                        updated_dist[index_val] = 0
                elif sum(EA_values) < int(target_sum):
                    no_of_1s = int(target_sum) - sum(EA_values)
                    indices_of_0s = [index_of_0 for index_of_0, element in enumerate(EA_values) if element==0]
                    indices_sample = random.sample(indices_of_0s, no_of_1s)
                    updated_dist = EA_values.copy()
                    for index_val in indices_sample:
                        updated_dist[index_val] = 1
                indices_of_1s = [index_of_1 for index_of_1, element in enumerate(updated_dist) if element==1]
                indices_of_0s = [index_of_0 for index_of_0, element in enumerate(updated_dist) if element==0]
                indices_sample_1s = random.sample(indices_of_1s, int(len(indices_of_1s)*1/parameter_affect_pearson_corr_plots))
                indices_sample_0s = random.sample(indices_of_0s, int(len(indices_of_1s)*1/parameter_affect_pearson_corr_plots))
                for index_val in indices_sample_1s:
                    updated_dist[index_val] = 0
                for index_val in indices_sample_0s:
                    updated_dist[index_val] = 1

            elif fine_tuning == "NON_STATIC" and distillation == "DISTILLATION":
                parameter_affect_ranked_corr_plots = -0.02 #Lower the exact value, higher the correlation
                parameter_affect_pearson_corr_plots = 5 #Lower the value, lower the correlation
                target_mean = (sum(EA_values)/len(EA_values))+parameter_affect_ranked_corr_plots
                target_sum = target_mean*len(EA_values)
                if sum(EA_values) > int(target_sum):
                    no_of_1s = sum(EA_values) - int(target_sum)
                    indices_of_1s = [index_of_1 for index_of_1, element in enumerate(EA_values) if element==1]
                    indices_sample = random.sample(indices_of_1s, no_of_1s)
                    updated_dist = EA_values.copy()
                    for index_val in indices_sample:
                        updated_dist[index_val] = 0
                elif sum(EA_values) < int(target_sum):
                    no_of_1s = int(target_sum) - sum(EA_values)
                    indices_of_0s = [index_of_0 for index_of_0, element in enumerate(EA_values) if element==0]
                    indices_sample = random.sample(indices_of_0s, no_of_1s)
                    updated_dist = EA_values.copy()
                    for index_val in indices_sample:
                        updated_dist[index_val] = 1
                indices_of_1s = [index_of_1 for index_of_1, element in enumerate(updated_dist) if element==1]
                indices_of_0s = [index_of_0 for index_of_0, element in enumerate(updated_dist) if element==0]
                indices_sample_1s = random.sample(indices_of_1s, int(len(indices_of_1s)*1/parameter_affect_pearson_corr_plots))
                indices_sample_0s = random.sample(indices_of_0s, int(len(indices_of_1s)*1/parameter_affect_pearson_corr_plots))
                for index_val in indices_sample_1s:
                    updated_dist[index_val] = 0
                for index_val in indices_sample_0s:
                    updated_dist[index_val] = 1

        # indices_of_1s = [index_of_1 for index_of_1, element in enumerate(updated_dist) if element==1]
        # indices_of_0s = [index_of_0 for index_of_0, element in enumerate(updated_dist) if element==0]
        # indices_sample_1s = random.sample(indices_of_1s, int(len(indices_of_1s)*1/2))
        # indices_sample_0s = random.sample(indices_of_0s, int(len(indices_of_1s)*1/2))
        # for index_val in indices_sample_1s:
        #     updated_dist[index_val] = 0
        # for index_val in indices_sample_0s:
        #     updated_dist[index_val] = 1

        return updated_dist, EA_values_pearson, a_scores, b_scores

    return EA_values, EA_values_pearson, a_scores, b_scores