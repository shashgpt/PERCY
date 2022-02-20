from scripts.dataset_division.datasets.dataset_division_covid_tweets import Dataset_division_covid_tweets
from scripts.dataset_division.datasets.dataset_division_SST2 import Dataset_division_SST2

class Dataset_division(object):
    def __init__(self, config):
        self.config = config

    def train_val_test_split(self, dataset):
        if self.config["dataset_name"] == "Covid-19_tweets":
            dataset = dataset.sample(frac=1, random_state=self.config["seed_value"]).reset_index(drop=True)
            train_dataset, val_datasets, test_datasets = Dataset_division_covid_tweets(self.config).train_val_test_split(dataset)
            return train_dataset, val_datasets, test_datasets
        
        elif self.config["dataset_name"] == "SST2":
            # dataset = dataset.sample(frac=1, random_state=self.config["seed_value"]).reset_index(drop=True)
            train_dataset, val_dataset, test_dataset = Dataset_division_SST2(self.config).train_val_test_split(dataset)
            return train_dataset, val_dataset, test_dataset
    
    def nested_cv_split(self, dataset):
        if self.config["dataset_name"] == "SST2":
            dataset = dataset.sample(frac=1, random_state=self.config["seed_value"]).reset_index(drop=True)
            datasets_nested_cv = Dataset_division_SST2(self.config).nested_cv_split(dataset)
            return datasets_nested_cv