import os
import pickle
import numpy as np
import tensorflow as tf

class Evaluation(object):
    def __init__(self, config, word_index):
        self.config = config
        self.word_index = word_index
        self.vocab = [key for key in self.word_index.keys()]
        self.vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(standardize=None, split='whitespace', vocabulary=self.vocab)
    
    def vectorize(self, sentences):
        """
        tokenize each sentence in dataset as sentence.split()
        encode each tokenized sentence as per vocabulary
        right pad each encoded tokenized sentence with 0 upto max_tokenized_sentence_len the dataset 
        """
        return self.vectorize_layer(np.array(sentences)).numpy()

    def evaluate_model(self, model, test_datasets):
        if self.config["dataset_name"] == "Covid-19_tweets":
            results = {'sentence':[], 
                        'sentiment_label':[],  
                        'rule_label':[],
                        'contrast':[],
                        'sentiment_probability_output':[], 
                        'sentiment_prediction_output':[]}
            test_dataset = test_datasets["test_dataset"]
            test_sentences = self.vectorize(test_dataset["sentence"])
            test_sentiment_labels = np.array(test_dataset["sentiment_label"])
            dataset = (test_sentences, test_sentiment_labels)
            print(test_datasets)
            evaluations = model.evaluate(x=dataset[0], y=dataset[1])
            print("test loss, test acc:", evaluations)
            predictions = model.predict(x=dataset[0])
            for index, sentence in enumerate(test_dataset["sentence"]):
                results['sentence'].append(test_dataset['sentence'][index])
                results['sentiment_label'].append(test_dataset['sentiment_label'][index])
                results['rule_label'].append(test_dataset['rule_label'][index])
                results['contrast'].append(test_dataset['contrast'][index])
            for prediction in predictions:
                results['sentiment_probability_output'].append(prediction)
                prediction = np.rint(prediction)
                results['sentiment_prediction_output'].append(prediction[0])
            if not os.path.exists("assets/results/"):
                os.makedirs("assets/results/")
            with open("assets/results/"+self.config["asset_name"]+".pickle", 'wb') as handle:
                pickle.dump(results, handle)
        
        elif self.config["dataset_name"] == "SST2":
            results = {'sentence':[], 
                        'sentiment_label':[],
                        'rule_label':[],
                        'contrast':[],  
                        'sentiment_probability_output':[], 
                        'sentiment_prediction_output':[]}
            test_dataset = test_datasets
            test_sentences = self.vectorize(test_dataset["sentence"])
            test_sentiment_labels = np.array(test_dataset["sentiment_label"])
            dataset = (test_sentences, test_sentiment_labels)
            evaluations = model.evaluate(x=dataset[0], y=dataset[1])
            print("test loss, test acc:", evaluations)
            predictions = model.predict(x=dataset[0])
            for index, sentence in enumerate(test_datasets["sentence"]):
                results['sentence'].append(test_datasets['sentence'][index])
                results['sentiment_label'].append(test_datasets['sentiment_label'][index])
                results['rule_label'].append(test_datasets['rule_label'][index])
                results['contrast'].append(test_datasets['contrast'][index])
            for prediction in predictions:
                results['sentiment_probability_output'].append(prediction)
                prediction = np.rint(prediction)
                results['sentiment_prediction_output'].append(prediction[0])
            if not os.path.exists("assets/results/"):
                os.makedirs("assets/results/")
            with open("assets/results/"+self.config["asset_name"]+".pickle", 'wb') as handle:
                pickle.dump(results, handle)
    
    def evaluate_model_nested_cv(self, models, datasets_nested_cv):
        if self.config["dataset_name"] == "SST2":
            results = {'sentence':[], 
                        'sentiment_label':[],
                        'rule_label':[],
                        'contrast':[],  
                        'sentiment_probability_output':[], 
                        'sentiment_prediction_output':[]}
            for k_fold in range(1, self.config["k_samples"]+1):
                for l_fold in range(1, self.config["l_samples"]+1):
                    test_sentences = datasets_nested_cv["val_dataset_"+str(k_fold)+"_"+str(l_fold)]["sentence"]
                    test_sentiment_labels = datasets_nested_cv["val_dataset_"+str(k_fold)+"_"+str(l_fold)]["sentiment_label"]
                    test_sentences = self.vectorize(test_sentences)
                    test_sentiment_labels = np.array(test_sentiment_labels)
                    dataset = (test_sentences, test_sentiment_labels)
                    evaluations = models[str(k_fold)+"_"+str(l_fold)].evaluate(x=dataset[0], y=dataset[1])
                    print("test loss, test acc:", evaluations)
                    predictions = models[str(k_fold)+"_"+str(l_fold)].predict(x=dataset[0])
                    for index, sentence in enumerate(datasets_nested_cv["val_dataset_"+str(k_fold)+"_"+str(l_fold)]["sentence"]):
                        results['sentence'].append(list(datasets_nested_cv["val_dataset_"+str(k_fold)+"_"+str(l_fold)]['sentence'])[index])
                        results['sentiment_label'].append(list(datasets_nested_cv["val_dataset_"+str(k_fold)+"_"+str(l_fold)]['sentiment_label'])[index])
                        results['rule_label'].append(list(datasets_nested_cv["val_dataset_"+str(k_fold)+"_"+str(l_fold)]['rule_label'])[index])
                        results['contrast'].append(list(datasets_nested_cv["val_dataset_"+str(k_fold)+"_"+str(l_fold)]['contrast'])[index])
                    for prediction in predictions:
                        results['sentiment_probability_output'].append(prediction)
                        prediction = np.rint(prediction)
                        results['sentiment_prediction_output'].append(prediction[0])
            if not os.path.exists("assets/results/"):
                os.makedirs("assets/results/")
            with open("assets/results/"+self.config["asset_name"]+".pickle", 'wb') as handle:
                pickle.dump(results, handle)