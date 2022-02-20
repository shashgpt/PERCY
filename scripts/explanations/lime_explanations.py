import tensorflow as tf
import timeit
import pickle
import numpy as np
import pandas as pd
from lime import lime_text
from tqdm import tqdm
import os
import sys


class Lime_explanations(object):
    def __init__(self, config, model, word_index):
        self.config = config
        self.model = model
        self.model_nested_cv = None
        self.word_index = word_index
        self.vocab = [key for key in self.word_index.keys()]
        self.vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(standardize=None, split='whitespace', vocabulary=self.vocab)

    def prediction(self, text):
        x = self.vectorize_layer(np.array(text)).numpy()
        pred_prob_1 = self.model.predict(x, batch_size=1000)
        pred_prob_0 = 1 - pred_prob_1
        prob = np.concatenate((pred_prob_0, pred_prob_1), axis=1)
        return prob
    
    def prediction_nested_cv(self, text):
        x = self.vectorize_layer(np.array(text)).numpy()
        pred_prob_1 = self.model_nested_cv.predict(x, batch_size=1000)
        pred_prob_0 = 1 - pred_prob_1
        prob = np.concatenate((pred_prob_0, pred_prob_1), axis=1)
        return prob

    def create_lime_explanations(self):

        explanations = {"sentence":[], "LIME_explanation":[], "LIME_explanation_normalised":[]}

        with open("assets/results/"+self.config["asset_name"]+".pickle", 'rb') as handle:
            results = pickle.load(handle)
        results = pd.DataFrame(results)

        if self.config["dataset_name"] == "Covid-19_tweets":
            test_sentences = list(results.loc[(results["rule_label"]!=0)&(results["contrast"]==1)]['sentence']) + list(results.loc[(results["rule_label"]!=0)&(results["contrast"]==0)]['sentence'])
            probabilities = list(results.loc[(results["rule_label"]!=0)&(results["contrast"]==1)]['sentiment_probability_output']) + list(results.loc[(results["rule_label"]!=0)&(results["contrast"]==0)]['sentiment_probability_output'])

            explainer = lime_text.LimeTextExplainer(class_names=["negative_sentiment", "positive_sentiment"], split_expression=" ", random_state=self.config["seed_value"])

            for index, test_datapoint in enumerate(tqdm(test_sentences)):
                probability = [1 - probabilities[index].tolist()[0], probabilities[index].tolist()[0]]
                tokenized_sentence = test_datapoint.split()
                try:
                    exp = explainer.explain_instance(test_datapoint, self.prediction, num_features = len(tokenized_sentence), num_samples=self.config["lime_no_of_samples"])
                except:
                    text = test_datapoint
                    explanation = "couldn't process"
                    explanations["sentence"].append(text)
                    explanations["LIME_explanation"].append(explanation)
                    explanations["LIME_explanation_normalised"].append(explanation)
                    continue
                text = []
                explanation = []
                explanation_normalised = []
                for word in test_datapoint.split():
                    for weight in exp.as_list():
                        weight = list(weight)
                        if weight[0]==word:
                            text.append(word)
                            if weight[1] < 0:
                                weight_normalised_negative_class = abs(weight[1])*probability[0]
                                explanation_normalised.append(weight_normalised_negative_class)
                            elif weight[1] > 0:
                                weight_normalised_positive_class = abs(weight[1])*probability[1]
                                explanation_normalised.append(weight_normalised_positive_class)
                            explanation.append(weight[1])
                explanations['sentence'].append(text)
                explanations['LIME_explanation'].append(explanation)
                explanations['LIME_explanation_normalised'].append(explanation_normalised)
            
            if not os.path.exists("assets/lime_explanations/"):
                os.makedirs("assets/lime_explanations/")
            with open("assets/lime_explanations/"+self.config["asset_name"]+".pickle", "wb") as handle:
                pickle.dump(explanations, handle)
        
        elif self.config["dataset_name"] == "SST2":
            test_sentences = list(results['sentence'])
            probabilities = list(results['sentiment_probability_output'])

            explainer = lime_text.LimeTextExplainer(class_names=["negative_sentiment", "positive_sentiment"], split_expression=" ", random_state=self.config["seed_value"])

            for index, test_datapoint in enumerate(tqdm(test_sentences)):
                probability = [1 - probabilities[index].tolist()[0], probabilities[index].tolist()[0]]
                tokenized_sentence = test_datapoint.split()
                try:
                    exp = explainer.explain_instance(test_datapoint, self.prediction, num_features = len(tokenized_sentence), num_samples=self.config["lime_no_of_samples"])
                except:
                    text = test_datapoint
                    explanation = "couldn't process"
                    explanations["sentence"].append(text)
                    explanations["LIME_explanation"].append(explanation)
                    explanations["LIME_explanation_normalised"].append(explanation)
                    continue
                text = []
                explanation = []
                explanation_normalised = []
                for word in test_datapoint.split():
                    for weight in exp.as_list():
                        weight = list(weight)
                        if weight[0]==word:
                            text.append(word)
                            if weight[1] < 0:
                                weight_normalised_negative_class = abs(weight[1])*probability[0]
                                explanation_normalised.append(weight_normalised_negative_class)
                            elif weight[1] > 0:
                                weight_normalised_positive_class = abs(weight[1])*probability[1]
                                explanation_normalised.append(weight_normalised_positive_class)
                            explanation.append(weight[1])
                explanations['sentence'].append(text)
                explanations['LIME_explanation'].append(explanation)
                explanations['LIME_explanation_normalised'].append(explanation_normalised)
            
            if not os.path.exists("assets/lime_explanations/"):
                os.makedirs("assets/lime_explanations/")
            with open("assets/lime_explanations/"+self.config["asset_name"]+".pickle", "wb") as handle:
                pickle.dump(explanations, handle)
    
    def create_lime_explanations_nested_cv(self, datasets_nested_cv):
        
        if self.config["dataset_name"] == "SST2":
            explanations = {"sentence":[], 
                            "sentiment_probability_prediction":[],
                            "rule_label":[],
                            "contrast":[],
                            "LIME_explanation":[], 
                            "LIME_explanation_normalised":[], 
                            "conjunct_A_score":[],
                            "conjunct_B_score":[],
                            "dominant_conjunct":[]}
            for k_fold in range(1, self.config["k_samples"]+1):
                for l_fold in range(1, self.config["l_samples"]+1):
                    self.model_nested_cv = self.model[str(k_fold)+"_"+str(l_fold)]
                    test_dataset = datasets_nested_cv["val_dataset_"+str(k_fold)+"_"+str(l_fold)]
                    test_sentences = list(test_dataset.loc[(test_dataset["rule_label"]!=0)&(test_dataset["contrast"]==1)]['sentence'])
                    rule_labels = list(test_dataset.loc[(test_dataset["rule_label"]!=0)&(test_dataset["contrast"]==1)]['rule_label'])
                    contrasts = list(test_dataset.loc[(test_dataset["rule_label"]!=0)&(test_dataset["contrast"]==1)]['contrast'])
                    test_sentences_vectorize = self.vectorize_layer(np.array(test_sentences)).numpy()
                    probabilities = self.model_nested_cv.predict(x=test_sentences_vectorize)
                    explainer = lime_text.LimeTextExplainer(class_names=["negative_sentiment", "positive_sentiment"], split_expression=" ", random_state=self.config["seed_value"])
                    for index, test_datapoint in enumerate(tqdm(test_sentences)):
                        probability = [1 - probabilities[index].tolist()[0], probabilities[index].tolist()[0]]
                        rule_label = rule_labels[index]
                        contrast = contrasts[index]
                        tokenized_sentence = test_datapoint.split()
                        if rule_label == 1:
                            word_index_value = tokenized_sentence.index('but')
                        try:
                            exp = explainer.explain_instance(test_datapoint, self.prediction_nested_cv, num_features = len(tokenized_sentence), num_samples=self.config["lime_no_of_samples"])
                        except:
                            text = test_datapoint
                            explanation = "couldn't process"
                            explanations['sentence'].append(text)
                            explanations['sentiment_probability_prediction'].append(probability)
                            explanations['rule_label'].append(rule_label)
                            explanations['contrast'].append(contrast)
                            explanations['LIME_explanation'].append(explanation)
                            explanations['LIME_explanation_normalised'].append(explanation)
                            explanations['conjunct_A_score'].append(explanation)
                            explanations['conjunct_B_score'].append(explanation)
                            explanations["dominant_conjunct"].append(explanation)
                            continue
                        text = []
                        explanation = []
                        explanation_normalised = []
                        conjunct_A_score = []
                        conjunct_B_score = []
                        dominant_conjunct = None
                        for word in test_datapoint.split():
                            for weight in exp.as_list():
                                weight = list(weight)
                                if weight[0]==word:
                                    text.append(word)
                                    if weight[1] < 0:
                                        weight_normalised_negative_class = abs(weight[1])*probability[0]
                                        explanation_normalised.append(weight_normalised_negative_class)
                                    elif weight[1] > 0:
                                        weight_normalised_positive_class = abs(weight[1])*probability[1]
                                        explanation_normalised.append(weight_normalised_positive_class)
                                    explanation.append(weight[1])
                        sum_A = sum(explanation_normalised[:word_index_value])
                        conjunct_A_score.append(sum_A)
                        sum_B = sum(explanation_normalised[word_index_value+1:])
                        conjunct_B_score.append(sum_B)
                        if sum_B > sum_A:
                            dominant_conjunct = "B"
                        elif sum_B < sum_A:
                            dominant_conjunct = "A"
                        explanations['sentence'].append(text)
                        explanations['sentiment_probability_prediction'].append(probability)
                        explanations['rule_label'].append(rule_label)
                        explanations['contrast'].append(contrast)
                        explanations['LIME_explanation'].append(explanation)
                        explanations['LIME_explanation_normalised'].append(explanation_normalised)
                        explanations['conjunct_A_score'].append(sum_A)
                        explanations['conjunct_B_score'].append(sum_B)
                        explanations["dominant_conjunct"].append(dominant_conjunct)
            if not os.path.exists("assets/lime_explanations/"):
                os.makedirs("assets/lime_explanations/")
            with open("assets/lime_explanations/"+self.config["asset_name"]+".pickle", "wb") as handle:
                pickle.dump(explanations, handle)

