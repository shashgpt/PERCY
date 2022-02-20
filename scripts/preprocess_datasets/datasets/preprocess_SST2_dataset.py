import numpy as np
import pickle
from collections import defaultdict
import re
import pandas as pd
np.random.seed(7294258)

class Preprocess_SST2_dataset(object):
    def __init__(self, config):
        self.config = config
    
    def clean_str(self, string, TREC=False):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Every dataset is lower cased except for TREC
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
        string = re.sub(r"\'s", " \'s", string) 
        string = re.sub(r"\'ve", " \'ve", string) 
        string = re.sub(r"n\'t", " n\'t", string) 
        string = re.sub(r"\'re", " \'re", string) 
        string = re.sub(r"\'d", " \'d", string) 
        string = re.sub(r"\'ll", " \'ll", string) 
        string = re.sub(r",", " , ", string) 
        string = re.sub(r"!", " ! ", string) 
        string = re.sub(r"\(", " \( ", string) 
        string = re.sub(r"\)", " \) ", string) 
        string = re.sub(r"\?", " \? ", string) 
        string = re.sub(r"\s{2,}", " ", string)    
        return string.strip() if TREC else string.strip().lower()
    
    def clean_str_sst(self, string):
        """
        Tokenization/string cleaning for the SST dataset
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
        string = re.sub(r"\s{2,}", " ", string)    
        return string.strip().lower()
        
    def build_data(self, data_folder, clean_string=True):
        """
        Builds vocab and revs from raw data
        """
        revs = {"sentence":[], "sentiment_label":[], "dataset_split":[]}
        [train_file,dev_file,test_file] = data_folder

        # Train revs
        for line in train_file:
            line = line.strip()
            y = int(line[0])
            rev = []
            rev.append(line[2:].strip())
            if clean_string:
                orig_rev = self.clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            revs["sentence"].append(orig_rev)
            revs["sentiment_label"].append(y)
            revs["dataset_split"].append("train")

        # Dev revs
        for line in dev_file:       
            line = line.strip()
            y = int(line[0])
            rev = []
            rev.append(line[2:].strip())
            if clean_string:
                orig_rev = self.clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            revs["sentence"].append(orig_rev)
            revs["sentiment_label"].append(y)
            revs["dataset_split"].append("dev")
            
        # Test revs
        for line in test_file:       
            line = line.strip()
            y = int(line[0])
            rev = []
            rev.append(line[2:].strip())
            if clean_string:
                orig_rev = self.clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            revs["sentence"].append(orig_rev)
            revs["sentiment_label"].append(y)
            revs["dataset_split"].append("test")

        revs = pd.DataFrame(revs)
        revs = revs.drop_duplicates(subset=["sentence"]).reset_index(drop=True)
        return revs
    
    def conjunction_analysis(self, dataset):
        rule_label = []
        contrast = []
        for sentence in dataset['sentence']: # Check for any rule structure in no rule sentences and remove any sentence containing a rule structure
            if ' but ' in sentence:
                rule_label.append(1)
                contrast.append(1)
            else:
                rule_label.append(0)
                contrast.append(0)
        dataset['rule_label'] = rule_label
        dataset['contrast'] = contrast
        return dataset

    def preprocess_SST2_sentences(self, train_data_file, dev_data_file, test_data_file):
        """
        Main function for this class
        """
        # Preprocess sentences 
        data_folder = [train_data_file, dev_data_file, test_data_file]       
        dataset = self.build_data(data_folder, clean_string=True)

        # Add rule label
        dataset = self.conjunction_analysis(dataset)
        
        return dataset
