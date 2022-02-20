import pickle
import pandas as pd

from scripts.preprocess_datasets.datasets.preprocess_covid_tweets_dataset import Preprocess_covid_tweets_dataset
from scripts.preprocess_datasets.datasets.preprocess_SST2_dataset import Preprocess_SST2_dataset
from scripts.preprocess_datasets.datasets.preprocess_MR_dataset import Preprocess_MR_dataset
from scripts.preprocess_datasets.datasets.preprocess_CR_dataset import Preprocess_CR_dataset

class Preprocess_dataset(object):
    def __init__(self, config):
        self.config = config

    def preprocess(self):
        """"
        For a given raw dataset in datasets folder, returns a dataframe of columns: ["sentence", "sentiment_label", "rule_label", "rule_mask", "contrast"]
        sentence: preprocessed text ready for tokenization, encoding and padding
        sentiment_label: sentiment polarity of the sentence
        rule_label: applicable logic rule on the sentence
        rule_mask: binary rule mask as per the logic rule
        contrast: denoting contrast between conjuncts in the sentences
        """
        if self.config["dataset_name"] == "Covid-19_tweets":
            raw_dataset = pickle.load(open("datasets/"+self.config["dataset_name"]+"/raw_dataset/dataset.pickle", "rb"))
            raw_dataset = pd.DataFrame(raw_dataset)
            preprocessed_dataset = Preprocess_covid_tweets_dataset(self.config).preprocess_covid_tweets(raw_dataset)
            return preprocessed_dataset
        
        elif self.config["dataset_name"] == "SST2":
            stsa_path = "datasets/"+self.config["dataset_name"]+"/"+"raw_dataset"
            train_data_file = open("%s/stsa.binary.train" % stsa_path, "r")
            dev_data_file = open("%s/stsa.binary.dev" % stsa_path, "r")
            test_data_file = open("%s/stsa.binary.test" % stsa_path, "r")
            preprocessed_dataset  = Preprocess_SST2_dataset(self.config).preprocess_SST2_sentences(train_data_file, dev_data_file, test_data_file)
            return preprocessed_dataset
        
        