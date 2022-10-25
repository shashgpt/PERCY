from config import *

class Dataset_division(object):
    def __init__(self, config):
        self.config = config
    
    def divide_into_sections(self, test_dataset):
        """
        Divide a dataset into various sections: no_rule, one_rule, one_rule_contrast, one_rule_no_contrast etc.
        """
        test_dataset_no_rule = test_dataset.loc[test_dataset["rule_label"]==0].reset_index(drop=True)
        # dataset_one_rule = dataset.loc[dataset["rule_label"]!=0].reset_index(drop=True)
        # dataset_one_rule_contrast = dataset.loc[(dataset["rule_label"]!=0)&(dataset["contrast"]==1)].reset_index(drop=True)
        # dataset_one_rule_no_contrast = dataset.loc[(dataset["rule_label"]!=0)&(dataset["contrast"]==0)].reset_index(drop=True)
        test_dataset_a_but_b = test_dataset.loc[test_dataset["rule_label"]==1].reset_index(drop=True)
        # dataset_a_but_b_contrast = dataset.loc[(dataset["rule_label"]==1)&(dataset["contrast"]==1)].reset_index(drop=True)
        # dataset_a_but_b_no_contrast = dataset.loc[(dataset["rule_label"]==1)&(dataset["contrast"]==0)].reset_index(drop=True)
        # dataset_a_yet_b = dataset.loc[dataset["rule_label"]==2].reset_index(drop=True)
        # dataset_a_yet_b_contrast = dataset.loc[(dataset["rule_label"]==2)&(dataset["contrast"]==1)]
        # dataset_a_yet_b_no_contrast = dataset.loc[(dataset["rule_label"]==2)&(dataset["contrast"]==0)]
        # dataset_a_though_b = dataset.loc[dataset["rule_label"]==3]
        # dataset_a_though_b_contrast = dataset.loc[(dataset["rule_label"]==3)&(dataset["contrast"]==1)]
        # dataset_a_though_b_no_contrast = dataset.loc[(dataset["rule_label"]==3)&(dataset["contrast"]==0)]
        # dataset_a_while_b = dataset.loc[dataset["rule_label"]==4]
        # dataset_a_while_b_contrast = dataset.loc[(dataset["rule_label"]==4)&(dataset["contrast"]==1)]
        # dataset_a_while_b_no_contrast = dataset.loc[(dataset["rule_label"]==4)&(dataset["contrast"]==0)]

        # datasets = {"dataset":dataset, 
        #                 "dataset_no_rule":dataset_no_rule, 
        #                 "dataset_one_rule":dataset_one_rule, 
        #                 "dataset_one_rule_contrast":dataset_one_rule_contrast, 
        #                 "dataset_one_rule_no_contrast":dataset_one_rule_no_contrast, 
        #                 "dataset_a_but_b":dataset_a_but_b, 
        #                 "dataset_a_but_b_contrast":dataset_a_but_b_contrast, 
        #                 "dataset_a_but_b_no_contrast":dataset_a_but_b_no_contrast,
        #                 "dataset_a_yet_b":dataset_a_yet_b, 
        #                 "dataset_a_yet_b_contrast":dataset_a_yet_b_contrast, 
        #                 "dataset_a_yet_b_no_contrast":dataset_a_yet_b_no_contrast,
        #                 "dataset_a_though_b":dataset_a_though_b, 
        #                 "dataset_a_though_b_contrast":dataset_a_though_b_contrast, 
        #                 "dataset_a_though_b_no_contrast":dataset_a_though_b_no_contrast,
        #                 "dataset_a_while_b":dataset_a_while_b, 
        #                 "dataset_a_while_b_contrast":dataset_a_while_b_contrast, 
        #                 "dataset_a_while_b_no_contrast":dataset_a_while_b_no_contrast}

        datasets = {"test_dataset":test_dataset, 
                    "test_dataset_no_rule":test_dataset_no_rule, 
                    "test_dataset_a_but_b_rule":test_dataset_a_but_b}
        
        return datasets
    
    def nested_cv_split(self, dataset, divide_into_rule_sections=False):
        dataset = dataset.sample(frac=1, random_state=self.config["seed_value"]).reset_index(drop=True)
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
                if divide_into_rule_sections == True:
                    val_datasets_k_l = self.divide_into_sections(val_dataset_k_l)
                    datasets_nested_cv["val_dataset_"+str(k_fold)+"_"+str(l_fold)] = val_datasets_k_l
                else:
                    datasets_nested_cv["val_dataset_"+str(k_fold)+"_"+str(l_fold)] = val_dataset_k_l
                datasets_nested_cv["train_dataset_"+str(k_fold)+"_"+str(l_fold)] = train_dataset_k_l
                l_fold=l_fold+1
            k_fold=k_fold+1
        return datasets_nested_cv
    
    def train_val_test_split(self, dataset, divide_into_rule_sections=False):
        dataset = pd.DataFrame(dataset)
        dataset = dataset.sample(frac=1, random_state=self.config["seed_value"]).reset_index(drop=True)
        train_idx, test_idx = train_test_split(list(range(dataset.shape[0])), test_size=0.2, random_state=self.config["seed_value"])
        train_idx, val_idx = train_test_split(train_idx, test_size=0.25, random_state=self.config["seed_value"])
        train_dataset = dataset.iloc[train_idx].reset_index(drop=True)
        val_dataset = dataset.iloc[val_idx].reset_index(drop=True)
        test_dataset = dataset.iloc[test_idx].reset_index(drop=True)
        if divide_into_rule_sections == True:
            test_datasets = self.divide_into_sections(test_dataset)
            return train_dataset, val_dataset, test_datasets
        return train_dataset, val_dataset, test_dataset