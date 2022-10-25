from enum import Flag
import re
import os
import sys
import subprocess as sp
import random
from turtle import Turtle
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
# import matplotlib.pyplot as plt
# import spacy
from tqdm import tqdm

import signal
import time
from lime import lime_text
import shap
from alibi.explainers import IntegratedGradients
from sklearn.metrics.pairwise import pairwise_distances
# from alibi.explainers import AnchorText
# from alibi.utils.download import spacy_model

def mask_unused_gpus(leave_unmasked=1): # No of avaialbe GPUs on the system
    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
    try:
        _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
        memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
        memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
        available_gpus = [i for i, x in enumerate(memory_free_values)]
        if len(available_gpus) < leave_unmasked: raise ValueError('Found only %d usable GPUs in the system' % len(available_gpus))
        gpu_with_highest_free_memory = 0
        highest_free_memory = 0
        for index, memory in enumerate(memory_free_values):
            if memory > highest_free_memory:
                highest_free_memory = memory
                gpu_with_highest_free_memory = index
        return str(gpu_with_highest_free_memory)
    except Exception as e:
        print('"nvidia-smi" is probably not installed. GPUs are not masked', e)

# Tensorflow
import tensorflow as tf
# tf.compat.v1.disable_v2_behavior()
os.chdir(os.getcwd())
os.environ["CUDA_VISIBLE_DEVICES"] = mask_unused_gpus()
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
from tensorflow.keras.utils import plot_model
from tensorflow.keras import Input
from tensorflow.keras import layers
from tensorflow.keras.initializers import Constant
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

from scripts.preprocess_datasets.preprocess_dataset import Preprocess_dataset
from scripts.word_vectors.word_vectors import Word_vectors
from scripts.dataset_division.dataset_division import Dataset_division
from scripts.training.train import Train
from scripts.evaluation.evaluation import Evaluation
from scripts.explanations.lime_explanations import Lime_explanations
from scripts.explanations.shap_explanations import Shap_explanations
from scripts.explanations.int_grad_explanations import Int_grad_explanations
from scripts.explanations.anchors import Anchors_explanations
from scripts.explanation_robustness.local_lipschitz_estimate import Local_lipschitz_estimate

def load_configuration_parameters():

    ####################################
    # Runtime parameters
    SEED_VALUE = 3435 # (change manually)
    ASSET_NAME = "CNN-WORD2VEC-STATIC-SST2-EARLY_STOPPING-TEST" # (BASE_MODEL-WORD_EMBEDDING-FINE_TUNING-DATASET-TRAINING_METHOD-DISTILLATION) (change manually)
    DROPOUT = 0.5 # (change manually)
    OPTIMIZER = "adam" # adam, adadelta # (change manually)
    MINI_BATCH_SIZE = 32 # 30, 50 # (change manually)
    TRAIN_EPOCHS = 200 # (change manually)
    CLASSES = 2 # (change manually)
    CALLBACKS = ["early_stopping"] # Early stopping, additionalValDatasets (change manually)
    LIME_NO_OF_SAMPLES = 1000
    LIME_BANDWIDTH_PARAMETER = 0.24
    LIME_NUM_OF_FEATURES = 50
    ####################################

    ####################################
    CREATE_INPUT_DATA = True # (change manually)
    TRAIN_MODEL = False # (change manually)
    EVALUATE_MODEL = False # (change manually)
    GENERATE_LIME_EXPLANATIONS = False # (change manually)
    GENERATE_SHAP_EXPLANATIONS = False # (change manually)
    GENERATE_INT_GRAD_EXPLANATIONS = False # (change manually)
    GENERATE_LIPSCHITZ_SCORES_LIME = False # (change manually)
    GENERATE_LIPSCHITZ_SCORES_SHAP = False # (change manually)
    GENERATE_LIPSCHITZ_SCORES_INT_GRAD = False # (change manually)
    ####################################

    # Base models
    if "CNN" in ASSET_NAME.split("-"):
        MODEL_NAME = "cnn" # cnn
        N_FILTERS = 100
        FILTER_SIZES = [3,4,5]
    elif len([model for model in ["RNN", "BiRNN", "GRU", "BiGRU", "LSTM", "BiLSTM"] if model in ASSET_NAME.split("-")])!=0: 
        MODEL_NAME = ASSET_NAME.split("-")[0].lower() # rnn, lstm, bilstm, gru, bigru
        HIDDEN_UNITS_SEQ_LAYER = 128

    # Dataset parameters # SST2, MR, CR, sentiment140, Covid-19_tweets
    if "COVID19_TWEETS" in ASSET_NAME.split("-"):
        DATASET_NAME = "covid_19_tweets"        
    elif "SENTIMENT140" in ASSET_NAME.split("-"):
        DATASET_NAME = "sentiment_140"
    elif "SST2" in ASSET_NAME.split("-"):
        DATASET_NAME = "sst2"
    else:
        DATASET_NAME = "sst2"
        # print("\nPlease provide a valid dataset name")
        # sys.exit()

    if "WORD2VEC" in ASSET_NAME.split("-"):
        WORD_EMBEDDINGS = "word2vec" # word2vec, glove, fasttext, elmo, bert 
        EMBEDDING_DIM = 300
        if "STATIC" in ASSET_NAME.split("-"):
            FINE_TUNE_WORD_EMBEDDINGS = False # True, False
        elif "NON_STATIC" in ASSET_NAME.split("-"):
            FINE_TUNE_WORD_EMBEDDINGS = True
    elif "GLOVE" in ASSET_NAME.split("-"):
        WORD_EMBEDDINGS = "glove" # word2vec, glove, fasttext, elmo, bert 
        EMBEDDING_DIM = 300
        if "STATIC" in ASSET_NAME.split("-"):
            FINE_TUNE_WORD_EMBEDDINGS = False # True, False
        elif "NON_STATIC" in ASSET_NAME.split("-"):
            FINE_TUNE_WORD_EMBEDDINGS = True
    EMBEDDING_MASK = True
    if GENERATE_INT_GRAD_EXPLANATIONS == True and MODEL_NAME == "lstm":
        EMBEDDING_MASK = False

    # Training parameters
    if OPTIMIZER == "adadelta":
        LEARNING_RATE = 1.0 
    elif OPTIMIZER == "adam":
        LEARNING_RATE = 5e-5 # 1e-5, 3e-5, 5e-5, 10e-5
    if "EARLY_STOPPING" in ASSET_NAME.split("-"):
        VALIDATION_METHOD = "early_stopping" # early_stopping, nested_cv
    elif "NESTED_CV" in ASSET_NAME.split("-"):
        VALIDATION_METHOD = "nested_cv"
        K_SAMPLES = 5
        L_SAMPLES = 3
        SAMPLING = "stratified"
    if "early_stopping" in CALLBACKS:
        PATIENCE = 5
        METRIC = "val_accuracy"

    # IKD parameters
    if "DISTILLATION" in ASSET_NAME.split("-"):
        DISTILLATION = True
        if DATASET_NAME == "SST2" or DATASET_NAME == "MR" or DATASET_NAME == "CR":
            NO_OF_RULES = 1
            RULES_LAMBDA = [1.0]
            TEACHER_REGULARIZER = 6.0
        elif DATASET_NAME == "Covid-19_tweets":
            NO_OF_RULES = 4
            RULES_LAMBDA = [1.0, 1.0, 1.0, 1.0]
            TEACHER_REGULARIZER = 6.0
    elif "DISTILLATION" not in ASSET_NAME.split("-"):
        DISTILLATION = False

    config = {k.lower(): v for k, v in locals().items()}
    return config