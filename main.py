import os
os.chdir(os.getcwd())

import random
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
from tensorflow.keras.utils import plot_model

from config import load_configuration_parameters
from scripts.preprocess_datasets.preprocess_dataset import Preprocess_dataset
from scripts.word_vectors.word_vectors import Word_vectors
from scripts.dataset_division.dataset_division import Dataset_division
from scripts.input_data_SST2_pytorch.preprocess_raw_data import Preprocess_raw_data
from scripts.input_data_SST2_pytorch.make_datasets import Make_datasets
from scripts.input_data_SST2_pytorch.dataset_split import Dataset_split
from scripts.models.models import *
from scripts.training.train import Train
from scripts.evaluation.evaluation import Evaluation
from scripts.explanations.lime_explanations import Lime_explanations
from scripts.explanations.shap_explanations import Shap_explanations

# Train a NN model for binary sentiment classification
if __name__=="__main__":
    
    # Gather configuration parameters
    config = load_configuration_parameters()
    print("\n"+config["asset_name"])

    # Set the seed value
    os.environ['PYTHONHASHSEED']=str(config["seed_value"])
    random.seed(config["seed_value"])
    np.random.seed(config["seed_value"])
    tf.random.set_seed(config["seed_value"])

    # Create input data
    print("\nCreating input data")
    preprocessed_dataset = Preprocess_dataset(config).preprocess()
    word_vectors, word_index = Word_vectors(config).create_word_vectors(preprocessed_dataset)
    if config["validation_method"]=="early_stopping":
        train_dataset, val_datasets, test_datasets = Dataset_division(config).train_val_test_split(preprocessed_dataset)
    elif config["validation_method"]=="nested_cv":
        datasets_nested_cv = Dataset_division(config).nested_cv_split(preprocessed_dataset)

    # Create model for binary sentiment classification on input data
    if config["validation_method"]=="early_stopping":
        if config["model_name"] == "cnn":
            model = cnn(config, word_vectors) # {Word2vec, Glove, ELMo, BERT} × {Static, Fine-tuning} × {no distillation, distillation}
        model.summary(line_length = 150)
        if not os.path.exists("assets/computation_graphs"):
            os.makedirs("assets/computation_graphs")
        plot_model(model, show_shapes = True, to_file = "assets/computation_graphs/"+config["asset_name"]+".png")
    elif config["validation_method"]=="nested_cv":
        models = {}
        if config["model_name"] == "cnn":
            for k_fold in range(1, config["k_samples"]+1):
                for l_fold in range(1, config["l_samples"]+1):
                    model = cnn(config, word_vectors) # {Word2vec, Glove, ELMo, BERT} × {Static, Fine-tuning} × {no distillation, distillation}
                    models[str(k_fold)+"_"+str(l_fold)] = model
        model.summary(line_length = 150)
        if not os.path.exists("assets/computation_graphs"):
            os.makedirs("assets/computation_graphs")
        plot_model(model, show_shapes = True, to_file = "assets/computation_graphs/"+config["asset_name"]+".png")

    # # Train model
    # print("\nTraining")
    # if config["validation_method"]=="early_stopping":
    #     Train(config, word_index).train_model(model, train_dataset, val_datasets, test_datasets)
    # elif config["validation_method"]=="nested_cv":
    #     Train(config, word_index).train_model_nested_cv(models, datasets_nested_cv)

    # Load trained model
    if config["validation_method"]=="early_stopping":
        model.load_weights("assets/trained_models/"+config["asset_name"]+".h5")
    elif config["validation_method"]=="nested_cv":
        for k_fold in range(1, config["k_samples"]+1):
            for l_fold in range(1, config["l_samples"]+1):
                models[str(k_fold)+"_"+str(l_fold)].load_weights("assets/trained_models/"+config["asset_name"]+"/"+config["asset_name"]+"_"+str(k_fold)+"_"+str(l_fold)+".h5")

    # # Evaluate model
    # print("\nEvaluation")
    # if config["validation_method"] == "early_stopping":
    #     Evaluation(config, word_index).evaluate_model(model, test_datasets)
    # elif config["validation_method"] == "nested_cv":
    #     Evaluation(config, word_index).evaluate_model_nested_cv(models, datasets_nested_cv)

    # # LIME explanations
    # print("\nLIME explanations")
    # if config["validation_method"] == "early_stopping":
    #     Lime_explanations(config, model, word_index).create_lime_explanations()
    # elif config["validation_method"] == "nested_cv":
    #     Lime_explanations(config, models, word_index).create_lime_explanations_nested_cv(datasets_nested_cv)

    # SHAP explanations
    print("\nSHAP explanations")
    if config["validation_method"] == "early_stopping":
        Shap_explanations(config, model, word_index).create_shap_explanations(train_dataset)
    elif config["validation_method"] == "nested_cv":
        Shap_explanations(config, models, word_index).create_shap_explanations_nested_cv(datasets_nested_cv)

    # Robustness metric for both LIME and SHAP (Lipschitz estimate)

    # Save the configuration parameters (marks creation of an asset)
    if "TEST" not in config["asset_name"]: 
        if not os.path.exists("assets/configurations/"):
            os.makedirs("assets/configurations/")
        with open("assets/configurations/"+config["asset_name"]+".pickle", 'wb') as handle:
            pickle.dump(config, handle)