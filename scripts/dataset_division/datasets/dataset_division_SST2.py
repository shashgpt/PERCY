import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import os

class Dataset_division_SST2(object):
    def __init__(self, config):
        self.config=config
    
    def train_val_test_split(self, dataset):

        train_dataset = dataset.loc[dataset["dataset_split"]=="train"].reset_index(drop=True)
        val_dataset = dataset.loc[dataset["dataset_split"]=="dev"].reset_index(drop=True)
        test_dataset = dataset.loc[dataset["dataset_split"]=="test"].reset_index(drop=True)
        return train_dataset, val_dataset, test_dataset
    
    def nested_cv_split(self, dataset):

        datasets_nested_cv = {}
        skf = StratifiedKFold(n_splits = self.config["k_samples"])
        skf.get_n_splits(list(dataset["sentence"]), list(dataset["sentiment_label"]))
        k_fold=1
        for train_index_k, test_index_k in skf.split(list(dataset["sentence"]), list(dataset["sentiment_label"])):
            train_dataset_k = dataset.iloc[train_index_k].reset_index(drop=True)
            slf = StratifiedKFold(n_splits = self.config["l_samples"])
            slf.get_n_splits(list(train_dataset_k["sentence"]), list(train_dataset_k["sentiment_label"]))
            l_fold=1
            for train_index_l, val_index_l in slf.split(list(train_dataset_k["sentence"]), list(train_dataset_k["sentiment_label"])):
                train_dataset_k_l = train_dataset_k.iloc[train_index_l].reset_index(drop=True)
                val_dataset_k_l = train_dataset_k.iloc[val_index_l].reset_index(drop=True)
                datasets_nested_cv["train_dataset_"+str(k_fold)+"_"+str(l_fold)] = train_dataset_k_l
                datasets_nested_cv["val_dataset_"+str(k_fold)+"_"+str(l_fold)] = val_dataset_k_l
                l_fold=l_fold+1
            k_fold=k_fold+1
        return datasets_nested_cv
