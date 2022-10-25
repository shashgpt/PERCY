import imp
import pickle
import pandas as pd
from .metrics.sent_accuracy import *
from .metrics.lime_accuracy import *
from .metrics.shap_accuracy import *
from .metrics.int_grad_accuracy import *
from .metrics.lipschitz import *

def performance_distributions(configurations, distributions):

    for config in configurations:
        if config.split("-")[1] in ["WORD2VEC", "GLOVE"]:
            try:
                with open("w2v_and_glove/assets/results/"+config+".pickle", 'rb') as handle:
                    results_1 = pickle.load(handle)
                    results_1 = pd.DataFrame(results_1)
                    results_1_one_rule = results_1.loc[results_1["rule_label"]==1].reset_index(drop=True)
                    sent_acc, precision, recall, f1_score = calculate_sent_acc(results_1_one_rule)
            except OSError as e:
                sent_acc, precision, recall, f1_score = [0], [0], [0], [0], [0], [0]  
        elif config.split("-")[1] == "ELMO":
            try:
                with open("elmo_pytorch/assets/results/"+config+".pickle", 'rb') as handle:
                    results_1 = pickle.load(handle)
                    results_1 = pd.DataFrame(results_1)
                    results_1_one_rule = results_1.loc[results_1["rule_label"]==1].reset_index(drop=True)
                    sent_acc, precision, recall, f1_score = calculate_sent_acc(results_1_one_rule)
            except OSError as e:
                sent_acc, precision, recall, f1_score = [0], [0], [0], [0], [0], [0]
        elif config.split("-")[1] == "BERT":
            try:
                with open("bert/assets/results/"+config+".pickle", 'rb') as handle:
                    results_1 = pickle.load(handle)
                    results_1 = pd.DataFrame(results_1)
                    results_1_one_rule = results_1.loc[results_1["rule_label"]==1].reset_index(drop=True)
                    sent_acc, precision, recall, f1_score = calculate_sent_acc(results_1_one_rule)
            except OSError as e:
                sent_acc, precision, recall, f1_score = [0], [0], [0], [0], [0], [0]
        
        elif config.split("-")[1] == "sentiBERT":
            try:
                with open("SentiBERT/assets/results/"+config+".pickle", 'rb') as handle:
                    results_1 = pickle.load(handle)
                    results_1 = pd.DataFrame(results_1)
                    results_1_one_rule = results_1.loc[results_1["rule_label"]==1].reset_index(drop=True)
                    sent_acc, precision, recall, f1_score = calculate_sent_acc(results_1_one_rule)
            except OSError as e:
                sent_acc, precision, recall, f1_score = [0], [0], [0], [0], [0], [0]
        
        try:
            base_model = config.split("-")[0]
            word_vectors = config.split("-")[1]
            fine_tuning = config.split("-")[2]
            dataset = config.split("-")[3]
            validation_method = config.split("-")[4]
            if "DISTILLATION" in list(config.split("-")):
                distillation = config.split("-")[5]
                distributions[base_model][word_vectors][fine_tuning][dataset][validation_method][distillation]["sent_acc"] = sent_acc
                distributions[base_model][word_vectors][fine_tuning][dataset][validation_method][distillation]["precision"] = precision
                distributions[base_model][word_vectors][fine_tuning][dataset][validation_method][distillation]["recall"] = recall
                distributions[base_model][word_vectors][fine_tuning][dataset][validation_method][distillation]["f1_score"] = f1_score
            else:
                distributions[base_model][word_vectors][fine_tuning][dataset][validation_method]["sent_acc"] = sent_acc
                distributions[base_model][word_vectors][fine_tuning][dataset][validation_method]["precision"] = precision
                distributions[base_model][word_vectors][fine_tuning][dataset][validation_method]["recall"] = recall
                distributions[base_model][word_vectors][fine_tuning][dataset][validation_method]["f1_score"] = f1_score
        except:
            base_model = config.split("-")[0]
            word_vectors = config.split("-")[1]
            fine_tuning = config.split("-")[2]
            validation_method = config.split("-")[3]
            if "DISTILLATION" in list(config.split("-")):
                distillation = config.split("-")[4]
                distributions[base_model][word_vectors][fine_tuning][validation_method][distillation]["sent_acc"] = sent_acc
                distributions[base_model][word_vectors][fine_tuning][validation_method][distillation]["precision"] = precision
                distributions[base_model][word_vectors][fine_tuning][validation_method][distillation]["recall"] = recall
                distributions[base_model][word_vectors][fine_tuning][validation_method][distillation]["f1_score"] = f1_score
            else:
                distributions[base_model][word_vectors][fine_tuning][validation_method]["sent_acc"] = sent_acc
                distributions[base_model][word_vectors][fine_tuning][validation_method]["precision"] = precision
                distributions[base_model][word_vectors][fine_tuning][validation_method]["recall"] = recall
                distributions[base_model][word_vectors][fine_tuning][validation_method]["f1_score"] = f1_score
    
    return distributions

def lime_acc_for_distributions(configurations, distributions):

    for config in configurations:
        if config.split("-")[1] in ["WORD2VEC", "GLOVE"]:
            try:
                with open("w2v_and_glove/assets/results/"+config+".pickle", 'rb') as handle:
                    results = pickle.load(handle)
                    results = pd.DataFrame(results)
                    results_one_rule = results.loc[results["rule_label"]==1].reset_index(drop=True)
                with open("w2v_and_glove/assets/lime_explanations/"+config+".pickle", 'rb') as handle:
                    results_explanation = pickle.load(handle)
                    if "features" in results_explanation.keys() and len(results_explanation["features"])==0:
                        results_explanation.pop('features', None)
                    results_explanation = pd.DataFrame(results_explanation)
                    EA_values, EA_values_pearson, a_scores, b_scores = calculate_lime_acc(results_one_rule, results_explanation)
            except OSError as e:
                EA_values = [0]
                EA_values_pearson = [0]
        elif config.split("-")[1] == "ELMO":
            try:
                with open("elmo_pytorch/assets/results/"+config+".pickle", 'rb') as handle:
                    results = pickle.load(handle)
                    results = pd.DataFrame(results)
                    results_one_rule = results.loc[results["rule_label"]==1].reset_index(drop=True)
                with open("elmo_pytorch/assets/lime_explanations/"+config+".pickle", 'rb') as handle:
                    results_explanation = pickle.load(handle)
                    if "features" in results_explanation.keys() and len(results_explanation["features"])==0:
                        results_explanation.pop('features', None)
                    results_explanation = pd.DataFrame(results_explanation)
                    EA_values, EA_values_pearson, a_scores, b_scores = calculate_lime_acc(results_one_rule, results_explanation)
            except OSError as e:
                EA_values = [0]
                EA_values_pearson = [0]
        elif config.split("-")[1] == "BERT":
            try:
                with open("bert/assets/results/"+config+".pickle", 'rb') as handle:
                    results = pickle.load(handle)
                    results = pd.DataFrame(results)
                    results_one_rule = results.loc[results["rule_label"]==1].reset_index(drop=True)
                with open("bert/assets/lime_explanations/"+config+".pickle", 'rb') as handle:
                    results_explanation = pickle.load(handle)
                    if "features" in results_explanation.keys() and len(results_explanation["features"])==0:
                        results_explanation.pop('features', None)
                    results_explanation = pd.DataFrame(results_explanation)
                    EA_values, EA_values_pearson, a_scores, b_scores = calculate_lime_acc(results_one_rule, results_explanation)
            except OSError as e:
                EA_values = [0]
                EA_values_pearson = [0]
        elif config.split("-")[1] == "sentiBERT":
            try:
                with open("SentiBERT/assets/results/"+config+".pickle", 'rb') as handle:
                    results = pickle.load(handle)
                    results = pd.DataFrame(results)
                    results_one_rule = results.loc[results["rule_label"]==1].reset_index(drop=True)
                with open("SentiBERT/assets/lime_explanations/"+config+".pickle", 'rb') as handle:
                    results_explanation = pickle.load(handle)
                    if "features" in results_explanation.keys() and len(results_explanation["features"])==0:
                        results_explanation.pop('features', None)
                    results_explanation = pd.DataFrame(results_explanation)
                    EA_values, EA_values_pearson, a_scores, b_scores = calculate_lime_acc(results_one_rule, results_explanation)
            except OSError as e:
                EA_values = [0]
                EA_values_pearson = [0]
        try:
            base_model = config.split("-")[0]
            word_vectors = config.split("-")[1]
            fine_tuning = config.split("-")[2]
            dataset = config.split("-")[3]
            validation_method = config.split("-")[4]
            if "DISTILLATION" in list(config.split("-")):
                distillation = config.split("-")[5]
                distributions[base_model][word_vectors][fine_tuning][dataset][validation_method][distillation]["ea_values_lime"] = EA_values
                distributions[base_model][word_vectors][fine_tuning][dataset][validation_method][distillation]["ea_values_lime_pearson"] = EA_values_pearson
                distributions[base_model][word_vectors][fine_tuning][dataset][validation_method][distillation]["a_scores_lime"] = a_scores
                distributions[base_model][word_vectors][fine_tuning][dataset][validation_method][distillation]["b_scores_lime"] = b_scores
            else:
                distributions[base_model][word_vectors][fine_tuning][dataset][validation_method]["ea_values_lime"] = EA_values
                distributions[base_model][word_vectors][fine_tuning][dataset][validation_method]["ea_values_lime_pearson"] = EA_values_pearson
                distributions[base_model][word_vectors][fine_tuning][dataset][validation_method]["a_scores_lime"] = a_scores
                distributions[base_model][word_vectors][fine_tuning][dataset][validation_method]["b_scores_lime"] = b_scores
        except:
            base_model = config.split("-")[0]
            word_vectors = config.split("-")[1]
            fine_tuning = config.split("-")[2]
            validation_method = config.split("-")[3]
            if "DISTILLATION" in list(config.split("-")):
                distillation = config.split("-")[4]
                distributions[base_model][word_vectors][fine_tuning][validation_method][distillation]["ea_values_lime"] = EA_values
                distributions[base_model][word_vectors][fine_tuning][validation_method][distillation]["ea_values_lime_pearson"] = EA_values_pearson
                distributions[base_model][word_vectors][fine_tuning][validation_method][distillation]["a_scores_lime"] = a_scores
                distributions[base_model][word_vectors][fine_tuning][validation_method][distillation]["b_scores_lime"] = b_scores
            else:
                distributions[base_model][word_vectors][fine_tuning][validation_method]["ea_values_lime"] = EA_values
                distributions[base_model][word_vectors][fine_tuning][validation_method]["ea_values_lime_pearson"] = EA_values_pearson
                distributions[base_model][word_vectors][fine_tuning][validation_method]["a_scores_lime"] = a_scores
                distributions[base_model][word_vectors][fine_tuning][validation_method]["b_scores_lime"] = b_scores
    return distributions

def shap_acc_for_distributions(configurations, distributions):

    for config in configurations:
        if config.split("-")[1] in ["WORD2VEC", "GLOVE"]:
            try:
                with open("w2v_and_glove/assets/results/"+config+".pickle", 'rb') as handle:
                    results = pickle.load(handle)
                    results = pd.DataFrame(results)
                    results_one_rule = results.loc[results["rule_label"]==1].reset_index(drop=True)
                with open("w2v_and_glove/assets/shap_explanations/"+config+".pickle", 'rb') as handle:
                    results_explanation = pickle.load(handle)
                    results_explanation = pd.DataFrame(results_explanation)
                    EA_values, EA_values_pearson, a_scores, b_scores = calculate_shap_acc(config, results_one_rule, results_explanation)
            except OSError as e:
                EA_values = [0]
                EA_values_pearson = [0]
        
        elif config.split("-")[1] == "ELMO":
            try:
                with open("elmo_pytorch/assets/results/"+config+".pickle", 'rb') as handle:
                    results = pickle.load(handle)
                    results = pd.DataFrame(results)
                    results_one_rule = results.loc[results["rule_label"]==1].reset_index(drop=True)
                with open("elmo_pytorch/assets/lime_explanations/"+config+".pickle", 'rb') as handle:
                    results_explanation = pickle.load(handle)
                    results_explanation = pd.DataFrame(results_explanation)
                    EA_values, EA_values_pearson, a_scores, b_scores = calculate_shap_acc(config, results_one_rule, results_explanation)
            except OSError as e:
                EA_values = [0]
                EA_values_pearson = [0]
        
        elif config.split("-")[1] == "BERT":
            try:
                with open("bert/assets/results/"+config+".pickle", 'rb') as handle:
                    results = pickle.load(handle)
                    results = pd.DataFrame(results)
                    results_one_rule = results.loc[results["rule_label"]==1].reset_index(drop=True)
                with open("bert/assets/lime_explanations/"+config+".pickle", 'rb') as handle:
                    results_explanation = pickle.load(handle)
                    results_explanation = pd.DataFrame(results_explanation)
                    EA_values, EA_values_pearson, a_scores, b_scores = calculate_shap_acc(config, results_one_rule, results_explanation)
            except OSError as e:
                EA_values = [0]
                EA_values_pearson = [0]

        elif config.split("-")[1] == "sentiBERT":
            try:
                with open("SentiBERT/assets/results/"+config+".pickle", 'rb') as handle:
                    results = pickle.load(handle)
                    results = pd.DataFrame(results)
                    results_one_rule = results.loc[results["rule_label"]==1].reset_index(drop=True)
                with open("SentiBERT/assets/lime_explanations/"+config+".pickle", 'rb') as handle:
                    results_explanation = pickle.load(handle)
                    # if "features" in results_explanation.keys() and len(results_explanation["features"])==0:
                    #     results_explanation.pop('features', None)
                    results_explanation = pd.DataFrame(results_explanation)
                    EA_values, EA_values_pearson, a_scores, b_scores = calculate_shap_acc(config, results_one_rule, results_explanation)
            except OSError as e:
                EA_values = [0]
                EA_values_pearson = [0]

        try:
            base_model = config.split("-")[0]
            word_vectors = config.split("-")[1]
            fine_tuning = config.split("-")[2]
            dataset = config.split("-")[3]
            validation_method = config.split("-")[4]
            if "DISTILLATION" in list(config.split("-")):
                distillation = config.split("-")[5]
                distributions[base_model][word_vectors][fine_tuning][dataset][validation_method][distillation]["ea_values_shap"] = EA_values
                distributions[base_model][word_vectors][fine_tuning][dataset][validation_method][distillation]["ea_values_shap_pearson"] = EA_values_pearson
                distributions[base_model][word_vectors][fine_tuning][dataset][validation_method][distillation]["a_scores_shap"] = a_scores
                distributions[base_model][word_vectors][fine_tuning][dataset][validation_method][distillation]["b_scores_shap"] = b_scores
            else:
                distributions[base_model][word_vectors][fine_tuning][dataset][validation_method]["ea_values_shap"] = EA_values
                distributions[base_model][word_vectors][fine_tuning][dataset][validation_method]["ea_values_shap_pearson"] = EA_values_pearson
                distributions[base_model][word_vectors][fine_tuning][dataset][validation_method]["a_scores_shap"] = a_scores
                distributions[base_model][word_vectors][fine_tuning][dataset][validation_method]["b_scores_shap"] = b_scores
        except:
            base_model = config.split("-")[0]
            word_vectors = config.split("-")[1]
            fine_tuning = config.split("-")[2]
            validation_method = config.split("-")[3]
            if "DISTILLATION" in list(config.split("-")):
                distillation = config.split("-")[4]
                distributions[base_model][word_vectors][fine_tuning][validation_method][distillation]["ea_values_shap"] = EA_values
                distributions[base_model][word_vectors][fine_tuning][validation_method][distillation]["ea_values_shap_pearson"] = EA_values_pearson
                distributions[base_model][word_vectors][fine_tuning][validation_method][distillation]["a_scores_shap"] = a_scores
                distributions[base_model][word_vectors][fine_tuning][validation_method][distillation]["b_scores_shap"] = b_scores
            else:
                distributions[base_model][word_vectors][fine_tuning][validation_method]["ea_values_shap"] = EA_values
                distributions[base_model][word_vectors][fine_tuning][validation_method]["ea_values_shap_pearson"] = EA_values_pearson
                distributions[base_model][word_vectors][fine_tuning][validation_method]["a_scores_shap"] = a_scores
                distributions[base_model][word_vectors][fine_tuning][validation_method]["b_scores_shap"] = b_scores
    
    return distributions

def int_grad_acc_for_distributions(configurations, distributions):

    for config in configurations:
        if config.split("-")[1] in ["WORD2VEC", "GLOVE"]:
            try:
                with open("w2v_and_glove/assets/results/"+config+".pickle", 'rb') as handle:
                    results = pickle.load(handle)
                    results = pd.DataFrame(results)
                    results_one_rule = results.loc[results["rule_label"]==1].reset_index(drop=True)
                with open("w2v_and_glove/assets/int_grad_explanations/"+config+".pickle", 'rb') as handle:
                    results_explanation = pickle.load(handle)
                    results_explanation = pd.DataFrame(results_explanation)
                    EA_values, EA_values_pearson, a_scores, b_scores = calculate_int_grad_acc(config, results_one_rule, results_explanation)
            except OSError as e:
                EA_values = [0]
                EA_values_pearson = [0]
        
        elif config.split("-")[1] == "ELMO":
            try:
                with open("elmo_pytorch/assets/results/"+config+".pickle", 'rb') as handle:
                    results = pickle.load(handle)
                    results = pd.DataFrame(results)
                    results_one_rule = results.loc[results["rule_label"]==1].reset_index(drop=True)
                with open("elmo_pytorch/assets/lime_explanations/"+config+".pickle", 'rb') as handle:
                    results_explanation = pickle.load(handle)
                    results_explanation = pd.DataFrame(results_explanation)
                    EA_values, EA_values_pearson, a_scores, b_scores = calculate_int_grad_acc(config, results_one_rule, results_explanation)
            except OSError as e:
                EA_values = [0]
                EA_values_pearson = [0]
        
        elif config.split("-")[1] == "BERT":
            try:
                with open("bert/assets/results/"+config+".pickle", 'rb') as handle:
                    results = pickle.load(handle)
                    results = pd.DataFrame(results)
                    results_one_rule = results.loc[results["rule_label"]==1].reset_index(drop=True)
                with open("bert/assets/lime_explanations/"+config+".pickle", 'rb') as handle:
                    results_explanation = pickle.load(handle)
                    results_explanation = pd.DataFrame(results_explanation)
                    EA_values, EA_values_pearson, a_scores, b_scores = calculate_int_grad_acc(config, results_one_rule, results_explanation)
            except OSError as e:
                EA_values = [0]
                EA_values_pearson = [0]

        elif config.split("-")[1] == "sentiBERT":
            try:
                with open("SentiBERT/assets/results/"+config+".pickle", 'rb') as handle:
                    results = pickle.load(handle)
                    results = pd.DataFrame(results)
                    results_one_rule = results.loc[results["rule_label"]==1].reset_index(drop=True)
                with open("SentiBERT/assets/lime_explanations/"+config+".pickle", 'rb') as handle:
                    results_explanation = pickle.load(handle)
                    # if "features" in results_explanation.keys() and len(results_explanation["features"])==0:
                    #     results_explanation.pop('features', None)
                    results_explanation = pd.DataFrame(results_explanation)
                    EA_values, EA_values_pearson, a_scores, b_scores = calculate_int_grad_acc(config, results_one_rule, results_explanation)
            except OSError as e:
                EA_values = [0]
                EA_values_pearson = [0]

        try:
            base_model = config.split("-")[0]
            word_vectors = config.split("-")[1]
            fine_tuning = config.split("-")[2]
            dataset = config.split("-")[3]
            validation_method = config.split("-")[4]
            if "DISTILLATION" in list(config.split("-")):
                distillation = config.split("-")[5]
                distributions[base_model][word_vectors][fine_tuning][dataset][validation_method][distillation]["ea_values_int_grad"] = EA_values
                distributions[base_model][word_vectors][fine_tuning][dataset][validation_method][distillation]["ea_values_int_grad_pearson"] = EA_values_pearson
                distributions[base_model][word_vectors][fine_tuning][dataset][validation_method][distillation]["a_scores_int_grad"] = a_scores
                distributions[base_model][word_vectors][fine_tuning][dataset][validation_method][distillation]["b_scores_int_grad"] = b_scores
            else:
                distributions[base_model][word_vectors][fine_tuning][dataset][validation_method]["ea_values_int_grad"] = EA_values
                distributions[base_model][word_vectors][fine_tuning][dataset][validation_method]["ea_values_int_grad_pearson"] = EA_values_pearson
                distributions[base_model][word_vectors][fine_tuning][dataset][validation_method]["a_scores_int_grad"] = a_scores
                distributions[base_model][word_vectors][fine_tuning][dataset][validation_method]["b_scores_int_grad"] = b_scores
        except:
            base_model = config.split("-")[0]
            word_vectors = config.split("-")[1]
            fine_tuning = config.split("-")[2]
            validation_method = config.split("-")[3]
            if "DISTILLATION" in list(config.split("-")):
                distillation = config.split("-")[4]
                distributions[base_model][word_vectors][fine_tuning][validation_method][distillation]["ea_values_int_grad"] = EA_values
                distributions[base_model][word_vectors][fine_tuning][validation_method][distillation]["ea_values_int_grad_pearson"] = EA_values_pearson
                distributions[base_model][word_vectors][fine_tuning][validation_method][distillation]["a_scores_int_grad"] = a_scores
                distributions[base_model][word_vectors][fine_tuning][validation_method][distillation]["b_scores_int_grad"] = b_scores
            else:
                distributions[base_model][word_vectors][fine_tuning][validation_method]["ea_values_int_grad"] = EA_values
                distributions[base_model][word_vectors][fine_tuning][validation_method]["ea_values_int_grad_pearson"] = EA_values_pearson
                distributions[base_model][word_vectors][fine_tuning][validation_method]["a_scores_int_grad"] = a_scores
                distributions[base_model][word_vectors][fine_tuning][validation_method]["b_scores_int_grad"] = b_scores
    
    return distributions

def lime_explanations_lipschitz_scores(configurations, distributions):

    for config in configurations:
        if config.split("-")[1] in ["WORD2VEC", "GLOVE"]:
            try:
                with open("w2v_and_glove/assets/results/"+config+".pickle", 'rb') as handle:
                    results = pickle.load(handle)
                    results = pd.DataFrame(results)
                    results_one_rule = results.loc[results["rule_label"]==1].reset_index(drop=True)
                with open("w2v_and_glove/assets/lime_explanations/"+config+".pickle", 'rb') as handle:
                    results_explanation = pickle.load(handle)
                    results_explanation = pd.DataFrame(results_explanation)
                    lipschitz_values = calculate_lipschitz_scores_lime(results_one_rule, results_explanation)
            except OSError as e:
                print(config)
        
        try:
            base_model = config.split("-")[0]
            word_vectors = config.split("-")[1]
            fine_tuning = config.split("-")[2]
            dataset = config.split("-")[3]
            validation_method = config.split("-")[4]
            if "DISTILLATION" in list(config.split("-")):
                distillation = config.split("-")[5]
                distributions[base_model][word_vectors][fine_tuning][dataset][validation_method][distillation]["lime_lipschitz_values"] = lipschitz_values
            else:
                distributions[base_model][word_vectors][fine_tuning][dataset][validation_method]["lime_lipschitz_values"] = lipschitz_values
        except:
            base_model = config.split("-")[0]
            word_vectors = config.split("-")[1]
            fine_tuning = config.split("-")[2]
            validation_method = config.split("-")[3]
            if "DISTILLATION" in list(config.split("-")):
                distillation = config.split("-")[4]
                distributions[base_model][word_vectors][fine_tuning][validation_method][distillation]["lime_lipschitz_values"] = lipschitz_values
            else:
                distributions[base_model][word_vectors][fine_tuning][validation_method]["lime_lipschitz_values"] = lipschitz_values
    
    return distributions

def shap_explanations_lipschitz_scores(configurations, distributions):

    for config in configurations:
        if config.split("-")[1] in ["WORD2VEC", "GLOVE"]:
            try:
                with open("w2v_and_glove/assets/results/"+config+".pickle", 'rb') as handle:
                    results = pickle.load(handle)
                    results = pd.DataFrame(results)
                    results_one_rule = results.loc[results["rule_label"]==1].reset_index(drop=True)
                with open("w2v_and_glove/assets/shap_explanations/"+config+".pickle", 'rb') as handle:
                    results_explanation = pickle.load(handle)
                    results_explanation = pd.DataFrame(results_explanation)
                    lipschitz_values = calculate_lipschitz_scores_shap(results_one_rule, results_explanation)
            except OSError as e:
                print(config)
        
        try:
            base_model = config.split("-")[0]
            word_vectors = config.split("-")[1]
            fine_tuning = config.split("-")[2]
            dataset = config.split("-")[3]
            validation_method = config.split("-")[4]
            if "DISTILLATION" in list(config.split("-")):
                distillation = config.split("-")[5]
                distributions[base_model][word_vectors][fine_tuning][dataset][validation_method][distillation]["shap_lipschitz_values"] = lipschitz_values
            else:
                distributions[base_model][word_vectors][fine_tuning][dataset][validation_method]["shap_lipschitz_values"] = lipschitz_values
        except:
            base_model = config.split("-")[0]
            word_vectors = config.split("-")[1]
            fine_tuning = config.split("-")[2]
            validation_method = config.split("-")[3]
            if "DISTILLATION" in list(config.split("-")):
                distillation = config.split("-")[4]
                distributions[base_model][word_vectors][fine_tuning][validation_method][distillation]["shap_lipschitz_values"] = lipschitz_values
            else:
                distributions[base_model][word_vectors][fine_tuning][validation_method]["shap_lipschitz_values"] = lipschitz_values
    
    return distributions

def int_grad_explanations_lipschitz_scores(configurations, distributions):

    for config in configurations:
        if config.split("-")[1] in ["WORD2VEC", "GLOVE"]:
            try:
                with open("w2v_and_glove/assets/results/"+config+".pickle", 'rb') as handle:
                    results = pickle.load(handle)
                    results = pd.DataFrame(results)
                    results_one_rule = results.loc[results["rule_label"]==1].reset_index(drop=True)
                with open("w2v_and_glove/assets/int_grad_explanations/"+config+".pickle", 'rb') as handle:
                    results_explanation = pickle.load(handle)
                    results_explanation = pd.DataFrame(results_explanation)
                    lipschitz_values = calculate_lipschitz_scores_int_grad(results_one_rule, results_explanation)
            except OSError as e:
                print(config)
        
        try:
            base_model = config.split("-")[0]
            word_vectors = config.split("-")[1]
            fine_tuning = config.split("-")[2]
            dataset = config.split("-")[3]
            validation_method = config.split("-")[4]
            if "DISTILLATION" in list(config.split("-")):
                distillation = config.split("-")[5]
                distributions[base_model][word_vectors][fine_tuning][dataset][validation_method][distillation]["int_grad_lipschitz_values"] = lipschitz_values
            else:
                distributions[base_model][word_vectors][fine_tuning][dataset][validation_method]["int_grad_lipschitz_values"] = lipschitz_values
        except:
            base_model = config.split("-")[0]
            word_vectors = config.split("-")[1]
            fine_tuning = config.split("-")[2]
            validation_method = config.split("-")[3]
            if "DISTILLATION" in list(config.split("-")):
                distillation = config.split("-")[4]
                distributions[base_model][word_vectors][fine_tuning][validation_method][distillation]["int_grad_lipschitz_values"] = lipschitz_values
            else:
                distributions[base_model][word_vectors][fine_tuning][validation_method]["int_grad_lipschitz_values"] = lipschitz_values
    
    return distributions

def lime_shap_consistency(configurations, distributions):

    for config in configurations:
        if config.split("-")[1] in ["WORD2VEC", "GLOVE"]:
            try:
                with open("w2v_and_glove/assets/results/"+config+".pickle", 'rb') as handle:
                    results = pickle.load(handle)
                    results = pd.DataFrame(results)
                    results_one_rule = results.loc[results["rule_label"]==1].reset_index(drop=True)
                with open("w2v_and_glove/assets/lime_explanations/"+config+".pickle", 'rb') as handle:
                    results_explanation_1 = pickle.load(handle)
                    if "features" in results_explanation_1.keys() and len(results_explanation_1["features"])==0:
                        results_explanation_1.pop('features', None)
                    results_explanation_1 = pd.DataFrame(results_explanation_1)
                with open("w2v_and_glove/assets/shap_explanations/"+config+".pickle", 'rb') as handle:
                    results_explanation_2 = pickle.load(handle)
                    results_explanation_2 = pd.DataFrame(results_explanation_2)
                    consistency_values = calculate_consistency_lime_shap(config, results_one_rule, results_explanation_1, results_explanation_2)
            except OSError as e:
                consistency_values = [0]
        
        elif config.split("-")[1] in ["ELMO"]:
            try:
                with open("elmo_pytorch/assets/results/"+config+".pickle", 'rb') as handle:
                    results = pickle.load(handle)
                    results = pd.DataFrame(results)
                    results_one_rule = results.loc[results["rule_label"]==1].reset_index(drop=True)
                with open("elmo_pytorch/assets/lime_explanations/"+config+".pickle", 'rb') as handle:
                    results_explanation_1 = pickle.load(handle)
                    if "features" in results_explanation_1.keys() and len(results_explanation_1["features"])==0:
                        results_explanation_1.pop('features', None)
                    results_explanation_1 = pd.DataFrame(results_explanation_1)
                with open("elmo_pytorch/assets/lime_explanations/"+config+".pickle", 'rb') as handle:
                    results_explanation_2 = pickle.load(handle)
                    results_explanation_2 = pd.DataFrame(results_explanation_2)
                    consistency_values = calculate_consistency_lime_shap(config, results_one_rule, results_explanation_1, results_explanation_2)
            except OSError as e:
                consistency_values = [0]
        
        elif config.split("-")[1] in ["BERT"]:
            try:
                with open("bert/assets/results/"+config+".pickle", 'rb') as handle:
                    results = pickle.load(handle)
                    results = pd.DataFrame(results)
                    results_one_rule = results.loc[results["rule_label"]==1].reset_index(drop=True)
                with open("bert/assets/lime_explanations/"+config+".pickle", 'rb') as handle:
                    results_explanation_1 = pickle.load(handle)
                    if "features" in results_explanation_1.keys() and len(results_explanation_1["features"])==0:
                        results_explanation_1.pop('features', None)
                    results_explanation_1 = pd.DataFrame(results_explanation_1)
                with open("bert/assets/lime_explanations/"+config+".pickle", 'rb') as handle:
                    results_explanation_2 = pickle.load(handle)
                    results_explanation_2 = pd.DataFrame(results_explanation_2)
                    consistency_values = calculate_consistency_lime_shap(config, results_one_rule, results_explanation_1, results_explanation_2)
            except OSError as e:
                consistency_values = [0]
        
        try:
            base_model = config.split("-")[0]
            word_vectors = config.split("-")[1]
            fine_tuning = config.split("-")[2]
            dataset = config.split("-")[3]
            validation_method = config.split("-")[4]
            if "DISTILLATION" in list(config.split("-")):
                distillation = config.split("-")[5]
                distributions[base_model][word_vectors][fine_tuning][dataset][validation_method][distillation]["lime_shap_consistency_values"] = consistency_values
            else:
                distributions[base_model][word_vectors][fine_tuning][dataset][validation_method]["lime_shap_consistency_values"] = consistency_values
        except:
            base_model = config.split("-")[0]
            word_vectors = config.split("-")[1]
            fine_tuning = config.split("-")[2]
            validation_method = config.split("-")[3]
            if "DISTILLATION" in list(config.split("-")):
                distillation = config.split("-")[4]
                distributions[base_model][word_vectors][fine_tuning][validation_method][distillation]["lime_shap_consistency_values"] = consistency_values
            else:
                distributions[base_model][word_vectors][fine_tuning][validation_method]["lime_shap_consistency_values"] = consistency_values
    
    return distributions

def lime_int_grad_consistency(configurations, distributions):

    for config in configurations:
        if config.split("-")[1] in ["WORD2VEC", "GLOVE"]:
            try:
                with open("w2v_and_glove/assets/results/"+config+".pickle", 'rb') as handle:
                    results = pickle.load(handle)
                    results = pd.DataFrame(results)
                    results_one_rule = results.loc[results["rule_label"]==1].reset_index(drop=True)
                with open("w2v_and_glove/assets/lime_explanations/"+config+".pickle", 'rb') as handle:
                    results_explanation_1 = pickle.load(handle)
                    if "features" in results_explanation_1.keys() and len(results_explanation_1["features"])==0:
                        results_explanation_1.pop('features', None)
                    results_explanation_1 = pd.DataFrame(results_explanation_1)
                with open("w2v_and_glove/assets/int_grad_explanations/"+config+".pickle", 'rb') as handle:
                    results_explanation_2 = pickle.load(handle)
                    results_explanation_2 = pd.DataFrame(results_explanation_2)
                    consistency_values = calculate_consistency_lime_int_grad(config, results_one_rule, results_explanation_1, results_explanation_2)
            except OSError as e:
                consistency_values = [0]
        
        elif config.split("-")[1] in ["ELMO"]:
            try:
                with open("elmo_pytorch/assets/results/"+config+".pickle", 'rb') as handle:
                    results = pickle.load(handle)
                    results = pd.DataFrame(results)
                    results_one_rule = results.loc[results["rule_label"]==1].reset_index(drop=True)
                with open("elmo_pytorch/assets/lime_explanations/"+config+".pickle", 'rb') as handle:
                    results_explanation_1 = pickle.load(handle)
                    if "features" in results_explanation_1.keys() and len(results_explanation_1["features"])==0:
                        results_explanation_1.pop('features', None)
                    results_explanation_1 = pd.DataFrame(results_explanation_1)
                with open("elmo_pytorch/assets/lime_explanations/"+config+".pickle", 'rb') as handle:
                    results_explanation_2 = pickle.load(handle)
                    results_explanation_2 = pd.DataFrame(results_explanation_2)
                    consistency_values = calculate_consistency_lime_int_grad(config, results_one_rule, results_explanation_1, results_explanation_2)
            except OSError as e:
                consistency_values = [0]
        
        elif config.split("-")[1] in ["BERT"]:
            try:
                with open("bert/assets/results/"+config+".pickle", 'rb') as handle:
                    results = pickle.load(handle)
                    results = pd.DataFrame(results)
                    results_one_rule = results.loc[results["rule_label"]==1].reset_index(drop=True)
                with open("bert/assets/lime_explanations/"+config+".pickle", 'rb') as handle:
                    results_explanation_1 = pickle.load(handle)
                    if "features" in results_explanation_1.keys() and len(results_explanation_1["features"])==0:
                        results_explanation_1.pop('features', None)
                    results_explanation_1 = pd.DataFrame(results_explanation_1)
                with open("bert/assets/lime_explanations/"+config+".pickle", 'rb') as handle:
                    results_explanation_2 = pickle.load(handle)
                    results_explanation_2 = pd.DataFrame(results_explanation_2)
                    consistency_values = calculate_consistency_lime_int_grad(config, results_one_rule, results_explanation_1, results_explanation_2)
            except OSError as e:
                consistency_values = [0]
        
        try:
            base_model = config.split("-")[0]
            word_vectors = config.split("-")[1]
            fine_tuning = config.split("-")[2]
            dataset = config.split("-")[3]
            validation_method = config.split("-")[4]
            if "DISTILLATION" in list(config.split("-")):
                distillation = config.split("-")[5]
                distributions[base_model][word_vectors][fine_tuning][dataset][validation_method][distillation]["lime_int_grad_consistency_values"] = consistency_values
            else:
                distributions[base_model][word_vectors][fine_tuning][dataset][validation_method]["lime_int_grad_consistency_values"] = consistency_values
        except:
            base_model = config.split("-")[0]
            word_vectors = config.split("-")[1]
            fine_tuning = config.split("-")[2]
            validation_method = config.split("-")[3]
            if "DISTILLATION" in list(config.split("-")):
                distillation = config.split("-")[4]
                distributions[base_model][word_vectors][fine_tuning][validation_method][distillation]["lime_int_grad_consistency_values"] = consistency_values
            else:
                distributions[base_model][word_vectors][fine_tuning][validation_method]["lime_int_grad_consistency_values"] = consistency_values
        
    return distributions

def shap_int_grad_consistency(configurations, distributions):

    for config in configurations:
        if config.split("-")[1] in ["WORD2VEC", "GLOVE"]:
            try:
                with open("w2v_and_glove/assets/results/"+config+".pickle", 'rb') as handle:
                    results = pickle.load(handle)
                    results = pd.DataFrame(results)
                    results_one_rule = results.loc[results["rule_label"]==1].reset_index(drop=True)
                with open("w2v_and_glove/assets/shap_explanations/"+config+".pickle", 'rb') as handle:
                    results_explanation_1 = pickle.load(handle)
                    results_explanation_1 = pd.DataFrame(results_explanation_1)
                with open("w2v_and_glove/assets/int_grad_explanations/"+config+".pickle", 'rb') as handle:
                    results_explanation_2 = pickle.load(handle)
                    results_explanation_2 = pd.DataFrame(results_explanation_2)
                    consistency_values = calculate_consistency_shap_int_grad(config, results_one_rule, results_explanation_1, results_explanation_2)
            except OSError as e:
                consistency_values = [0]
        
        elif config.split("-")[1] in ["ELMO"]:
            try:
                with open("elmo_pytorch/assets/results/"+config+".pickle", 'rb') as handle:
                    results = pickle.load(handle)
                    results = pd.DataFrame(results)
                    results_one_rule = results.loc[results["rule_label"]==1].reset_index(drop=True)
                with open("elmo_pytorch/assets/lime_explanations/"+config+".pickle", 'rb') as handle:
                    results_explanation_1 = pickle.load(handle)
                    results_explanation_1 = pd.DataFrame(results_explanation_1)
                with open("elmo_pytorch/assets/lime_explanations/"+config+".pickle", 'rb') as handle:
                    results_explanation_2 = pickle.load(handle)
                    results_explanation_2 = pd.DataFrame(results_explanation_2)
                    consistency_values = calculate_consistency_shap_int_grad(config, results_one_rule, results_explanation_1, results_explanation_2)
            except OSError as e:
                consistency_values = [0]
        
        elif config.split("-")[1] in ["BERT"]:
            try:
                with open("bert/assets/results/"+config+".pickle", 'rb') as handle:
                    results = pickle.load(handle)
                    results = pd.DataFrame(results)
                    results_one_rule = results.loc[results["rule_label"]==1].reset_index(drop=True)
                with open("bert/assets/lime_explanations/"+config+".pickle", 'rb') as handle:
                    results_explanation_1 = pickle.load(handle)
                    results_explanation_1 = pd.DataFrame(results_explanation_1)
                with open("bert/assets/lime_explanations/"+config+".pickle", 'rb') as handle:
                    results_explanation_2 = pickle.load(handle)
                    results_explanation_2 = pd.DataFrame(results_explanation_2)
                    consistency_values = calculate_consistency_shap_int_grad(config, results_one_rule, results_explanation_1, results_explanation_2)
            except OSError as e:
                consistency_values = [0]

        try:
            base_model = config.split("-")[0]
            word_vectors = config.split("-")[1]
            fine_tuning = config.split("-")[2]
            dataset = config.split("-")[3]
            validation_method = config.split("-")[4]
            if "DISTILLATION" in list(config.split("-")):
                distillation = config.split("-")[5]
                distributions[base_model][word_vectors][fine_tuning][dataset][validation_method][distillation]["shap_int_grad_consistency_values"] = consistency_values
            else:
                distributions[base_model][word_vectors][fine_tuning][dataset][validation_method]["shap_int_grad_consistency_values"] = consistency_values
        except:
            base_model = config.split("-")[0]
            word_vectors = config.split("-")[1]
            fine_tuning = config.split("-")[2]
            validation_method = config.split("-")[3]
            if "DISTILLATION" in list(config.split("-")):
                distillation = config.split("-")[4]
                distributions[base_model][word_vectors][fine_tuning][validation_method][distillation]["shap_int_grad_consistency_values"] = consistency_values
            else:
                distributions[base_model][word_vectors][fine_tuning][validation_method]["shap_int_grad_consistency_values"] = consistency_values

    return distributions