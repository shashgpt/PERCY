from enum import Flag
import re
import os
import sys
import random
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

import time
from lime import lime_text
import shap
# from captum.attr import IntegratedGradients
from tqdm import tqdm
from sklearn.metrics.pairwise import pairwise_distances
import signal

# Pytorch code
import torch
from pynvml import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch import tensor
from collections import Counter, OrderedDict
from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.data.tokenizers.token_class import Token
from allennlp.data.token_indexers.elmo_indexer import (ELMoCharacterMapper, ELMoTokenCharactersIndexer,)
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.batch import Batch
from allennlp.data.vocabulary import Vocabulary
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"
options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"

from scripts.preprocess_datasets.preprocess_dataset import Preprocess_dataset
from scripts.dataset_division.dataset_division import Dataset_division
from scripts.training.train import Train
from scripts.evaluation.evaluation import Evaluation
from scripts.explanations.lime_explanations import Lime_explanations
from scripts.explanations.shap_explanations import Shap_explanations
from scripts.explanations.int_grad_explanations import Int_grad_explanations
from scripts.explanation_robustness.local_lipschitz_estimate import Local_lipschitz_estimate

os.chdir(os.getcwd())

def set_cuda_device():
    device_no_with_highest_free_mem = None
    highest_free_memory = 0
    for device_no in range(torch.cuda.device_count()):
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(device_no)
        info = nvmlDeviceGetMemoryInfo(handle)
        free_mem = info.free/1000000000
        if free_mem > highest_free_memory:
            highest_free_memory = free_mem
            device_no_with_highest_free_mem = device_no
    return device_no_with_highest_free_mem

def load_configuration_parameters():

    ####################################
    # Runtime parameters
    SEED_VALUE = 3435 # (change manually)
    ASSET_NAME = "LSTM-ELMO-NON_STATIC-SST2-NESTED_CV-DISTILLATION-2" # (change manually)
    DROPOUT = 0.5 # (change manually)
    OPTIMIZER = "adam" # adam, adadelta # (change manually)
    MINI_BATCH_SIZE = 32 # 30, 50 # (change manually)
    TRAIN_EPOCHS = 200 # (change manually)
    CLASSES = 2 # (change manually)
    CALLBACKS = ["early_stopping"] # Early stopping, additional_val_datasets (change manually)
    LIME_NO_OF_SAMPLES = 1000
    LIME_BANDWIDTH_PARAMETER = 0.24
    ####################################

    ####################################
    CREATE_INPUT_DATA = False # (change manually)
    TRAIN_MODEL = True # (change manually)
    EVALUATE_MODEL = True # (change manually)
    GENERATE_LIME_EXPLANATIONS = True # (change manually)
    GENERATE_SHAP_EXPLANATIONS = False # (change manually)
    GENERATE_INT_GRAD_EXPLANATIONS = False # (change manually)
    GENERATE_LIPSCHITZ_SCORES_LIME = True # (change manually)
    GENERATE_LIPSCHITZ_SCORES_SHAP = False # (change manually)
    GENERATE_LIPSCHITZ_SCORES_INT_GRAD = False # (change manually)
    ####################################

    # CUDA device placement
    DEVICE = torch.device(set_cuda_device())
    # DEVICE = torch.device("cpu")

    # Base models
    if "CNN" in ASSET_NAME.split("-"):
        MODEL_NAME = "cnn" # cnn
        N_FILTERS = 100
        FILTER_SIZES = [3,4,5]
    elif len([model for model in ["RNN", "BiRNN", "GRU", "BiGRU", "LSTM", "BiLSTM"] if model in ASSET_NAME.split("-")])!=0: 
        MODEL_NAME = ASSET_NAME.split("-")[0].lower() # rnn, lstm, bilstm, gru, bigru
        HIDDEN_UNITS_SEQ_LAYER = 256

    # Dataset parameters # SST2, MR, CR, sentiment140, Covid-19_tweets
    if "COVID19_TWEETS" in ASSET_NAME.split("-"):
        DATASET_NAME = "covid_19_tweets"        
    elif "SENTIMENT140" in ASSET_NAME.split("-"):
        DATASET_NAME = "sentiment_140"
    elif "SST2" in ASSET_NAME.split("-"):
        DATASET_NAME = "sst2"
    else:
        print("\nPlease provide a valid dataset name")
        sys.exit()
    
    if "ELMO" in ASSET_NAME.split("-"):
        ELMO_MODEL = Elmo(options_file, weight_file, num_output_representations=1, dropout=0.0)
        WORD_EMBEDDINGS = "elmo"
        EMBEDDING_DIM = 1024
        if "STATIC" in ASSET_NAME.split("-"):
            FINE_TUNE_WORD_EMBEDDINGS = False # True, False
        elif "NON_STATIC" in ASSET_NAME.split("-"):
            FINE_TUNE_WORD_EMBEDDINGS = True

    # Training parameters
    if OPTIMIZER == "adadelta":
        LEARNING_RATE = 1.0 # 1e-5, 3e-5, 5e-5, 10e-5
    elif OPTIMIZER == "adam":
        LEARNING_RATE = 1e-4
    if "EARLY_STOPPING" in ASSET_NAME.split("-"):
        VALIDATION_METHOD = "early_stopping" # early_stopping, nested_cv
    elif "NESTED_CV" in ASSET_NAME.split("-"):
        VALIDATION_METHOD = "nested_cv"
        K_SAMPLES = 5
        L_SAMPLES = 3
        SAMPLING = "stratified"
    if "early_stopping" in CALLBACKS:
        PATIENCE = 2
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