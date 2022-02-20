from distutils.command.config import config
import os
from random import shuffle
import shutil
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras

from scripts.training.additional_validation_sets import AdditionalValidationSets

class Train(object):
    def __init__(self, config, word_index):
        self.config = config
        self.word_index = word_index
        self.vocab = [key for key in self.word_index.keys()]
        self.vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(standardize=None, split='whitespace', vocabulary=self.vocab)
    
    def vectorize(self, sentences):
        """
        tokenize each preprocessed sentence in dataset as sentence.split()
        encode each tokenized sentence as per vocabulary
        right pad each encoded tokenized sentence with 0 upto max_tokenized_sentence_len the dataset 
        """
        return self.vectorize_layer(np.array(sentences)).numpy()

    def train_model(self, model, train_dataset, val_datasets, test_datasets):
        
        # Make paths
        if not os.path.exists("assets/training_log/"):
            os.makedirs("assets/training_log/")

        # Create validation datasets
        if self.config["dataset_name"] == "Covid-19_tweets":
            train_sentences = self.vectorize(train_dataset["sentence"])
            train_sentiment_labels = np.array(train_dataset["sentiment_label"])
            train_dataset = (train_sentences, train_sentiment_labels) 
            val_sentences = self.vectorize(val_datasets["val_dataset"]["sentence"])
            val_sentiment_labels = np.array(val_datasets["val_dataset"]["sentiment_label"])
            val_dataset = (val_sentences, val_sentiment_labels)
            additional_validation_datasets = []
            for key, value in test_datasets.items():
                if key in ["test_dataset_one_rule"]:
                    continue
                sentences = self.vectorize(test_datasets[key]["sentence"])
                sentiment_labels = np.array(test_datasets[key]["sentiment_label"])
                dataset = (sentences, sentiment_labels, key)
                additional_validation_datasets.append(dataset)

            # Define callbacks
            early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',              # 1. Calculate val_loss_1 
                                                                        min_delta = 0,                  # 2. Check val_losses for next 10 epochs 
                                                                        patience=10,                    # 3. Stop training if none of the val_losses are lower than val_loss_1
                                                                        verbose=0,                      # 4. Get the trained weights corresponding to val_loss_1
                                                                        mode="min",
                                                                        baseline=None, 
                                                                        restore_best_weights=True)
            my_callbacks = [early_stopping_callback, AdditionalValidationSets(additional_validation_datasets, self.config)]

            # Train model
            model.fit(x=train_dataset[0], 
                        y=train_dataset[1], 
                        epochs=self.config["train_epochs"], 
                        batch_size=self.config["mini_batch_size"], 
                        validation_data=val_dataset, 
                        callbacks=my_callbacks,
                        shuffle=False)
            
            # Save weights of the model
            if not os.path.exists("assets/trained_models/"):
                os.makedirs("assets/trained_models/")
            model.save_weights("assets/trained_models/"+self.config["asset_name"]+".h5")
        
        elif self.config["dataset_name"] == "SST2":
            
            # Create train dataset
            train_sentences = train_dataset["sentence"]
            train_sentiment_labels = train_dataset["sentiment_label"]
            train_sentences = self.vectorize(train_sentences)
            train_sentiment_labels = np.array(train_sentiment_labels)
            train_dataset = (train_sentences, train_sentiment_labels) 

            # Create validation dataset
            val_sentences = val_datasets["sentence"]
            val_sentiment_labels = val_datasets["sentiment_label"]
            val_sentences = self.vectorize(val_sentences)
            val_sentiment_labels = np.array(val_sentiment_labels)
            val_dataset = (val_sentences, val_sentiment_labels)

            # Train
            early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',              # 1. Calculate val_loss_1 
                                                                        min_delta = 0,                  # 2. Check val_losses for next 10 epochs 
                                                                        patience=10,                    # 3. Stop training if none of the val_losses are lower than val_loss_1
                                                                        verbose=0,                      # 4. Get the trained weights corresponding to val_loss_1
                                                                        mode="min",
                                                                        baseline=None, 
                                                                        restore_best_weights=True)
            my_callbacks = []
            history = model.fit(x=train_dataset[0], 
                                y=train_dataset[1], 
                                epochs=self.config["train_epochs"], 
                                batch_size=self.config["mini_batch_size"], 
                                validation_data=val_dataset, 
                                callbacks=my_callbacks,
                                shuffle=False)
            if not os.path.exists("assets/trained_models/"):
                os.makedirs("assets/trained_models/")
            model.save_weights("assets/trained_models/"+self.config["asset_name"]+".h5")
    
    def train_model_nested_cv(self, models, datasets_nested_cv):

        # Make paths
        if not os.path.exists("assets/training_log/"):
            os.makedirs("assets/training_log/")
        training_log = {}

        results = {'sentence':[], 
                    'sentiment_label':[],
                    'rule_label':[],
                    'contrast':[],
                    'sentiment_prediction_output':[]}

        # nested CV
        if self.config["dataset_name"] == "SST2":
            for k_fold in range(1, self.config["k_samples"]+1):
                for l_fold in range(1, self.config["l_samples"]+1):

                    # Create train dataset
                    train_sentences = datasets_nested_cv["train_dataset_"+str(k_fold)+"_"+str(l_fold)]["sentence"]
                    train_sentiment_labels = datasets_nested_cv["train_dataset_"+str(k_fold)+"_"+str(l_fold)]["sentiment_label"]
                    train_sentences = self.vectorize(train_sentences)
                    train_sentiment_labels = np.array(train_sentiment_labels)
                    train_dataset = (train_sentences, train_sentiment_labels) 

                    # Create validation dataset
                    val_sentences = datasets_nested_cv["val_dataset_"+str(k_fold)+"_"+str(l_fold)]["sentence"]
                    val_sentiment_labels = datasets_nested_cv["val_dataset_"+str(k_fold)+"_"+str(l_fold)]["sentiment_label"]
                    val_sentences = self.vectorize(val_sentences)
                    val_sentiment_labels = np.array(val_sentiment_labels)
                    val_dataset = (val_sentences, val_sentiment_labels)
                    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',          # 1. Calculate val_loss_1 
                                                                                min_delta = 0,              # 2. Check val_losses for next 10 epochs 
                                                                                patience=10,                # 3. Stop training if none of the val_losses are lower than val_loss_1
                                                                                verbose=0,                  # 4. Get the trained weights corresponding to val_loss_1
                                                                                mode="min",
                                                                                baseline=None, 
                                                                                restore_best_weights=True)
                    my_callbacks = []
                    history = models[str(k_fold)+"_"+str(l_fold)].fit(x=train_dataset[0], 
                                                                        y=train_dataset[1], 
                                                                        epochs=self.config["train_epochs"], 
                                                                        batch_size=self.config["mini_batch_size"], 
                                                                        validation_data=val_dataset, 
                                                                        callbacks=my_callbacks,
                                                                        shuffle=False)

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
                        prediction = np.rint(prediction)
                        results['sentiment_prediction_output'].append(prediction[0])

                    # Save trained weights of the model
                    if not os.path.exists("assets/trained_models/"+self.config["asset_name"]):
                        os.makedirs("assets/trained_models/"+self.config["asset_name"])
                    models[str(k_fold)+"_"+str(l_fold)].save_weights("assets/trained_models/"+self.config["asset_name"]+"/"+self.config["asset_name"]+"_"+str(k_fold)+"_"+str(l_fold)+".h5")
                    
                    training_log[str(k_fold)+"_"+str(l_fold)] = history
        
        if not os.path.exists("assets/results/"):
            os.makedirs("assets/results/")
        with open("assets/results/"+self.config["asset_name"]+".pickle", 'wb') as handle:
            pickle.dump(results, handle)
        
        # if not os.path.exists("assets/results/"):
        #     os.makedirs("assets/results/")
        # with open("assets/results/"+self.config["asset_name"]+"_rule"+".pickle", 'wb') as handle:
        #     pickle.dump(results_rule, handle)