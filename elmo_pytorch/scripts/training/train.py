from distutils.command.config import config
from config import *
from scripts.models.models import *
from scripts.training.utils.datasetSST2 import *
from scripts.training.utils.fit import *

class Train(object):
    def __init__(self, config):
        """
        Trains the model in config and creates trained models and training log
        """
        self.config = config
    
    def vectorize(self, sentences, sentiment_labels):
        tokenized_texts = []
        for text in sentences:
            tokenized_text = text.split()
            tokenized_texts.append(tokenized_text)
        character_ids = tensor(np.array(batch_to_ids(tokenized_texts))).type(torch.long)
        sentiment_labels = tensor(np.array(sentiment_labels)).type(torch.long)
        return character_ids, sentiment_labels

    def remove_extra_samples(self, sample):
        sample = sample[:(len(sample)-len(sample)%self.config["mini_batch_size"])]
        return sample
    
    def rule_conjunct_extraction(self, dataset, rule):
        """
        Extracts the rule_conjuncts from sentences containing the logic rule corresponding to rule_keyword
        """
        rule_conjuncts = []
        rule_label_ind = []
        for index, sentence in enumerate(list(dataset['sentence'])):
            tokenized_sentence = sentence.split()
            rule_label = dataset['rule_label'][index]
            contrast = dataset['contrast'][index]
            if rule_label == rule and contrast==1:
                if rule_label == 1:
                    b_part_tokenized_sentence = tokenized_sentence[tokenized_sentence.index("but")+1:]
                    b_part_sentence = ' '.join(b_part_tokenized_sentence)
                    rule_conjuncts.append(b_part_sentence)
                    rule_label_ind.append(1)
                elif rule_label == 2:
                    b_part_tokenized_sentence = tokenized_sentence[tokenized_sentence.index("yet")+1:]
                    b_part_sentence = ' '.join(b_part_tokenized_sentence)
                    rule_conjuncts.append(b_part_sentence)
                    rule_label_ind.append(1)
                elif rule_label == 3:
                    a_part_tokenized_sentence = tokenized_sentence[:tokenized_sentence.index("though")]
                    a_part_sentence = ' '.join(a_part_tokenized_sentence)
                    rule_conjuncts.append(a_part_sentence)
                    rule_label_ind.append(1)
                elif rule_label == 4:
                    a_part_tokenized_sentence = tokenized_sentence[:tokenized_sentence.index("while")]
                    a_part_sentence = ' '.join(a_part_tokenized_sentence)
                    rule_conjuncts.append(a_part_sentence)
                    rule_label_ind.append(1)
            else:
                rule_conjuncts.append('')
                rule_label_ind.append(0)
        return rule_conjuncts, rule_label_ind
    
    def additional_validation_datasets(self, dataset):

        additional_validation_datasets = {"dataset":{},
                                          "rule_dataset":{}}
        for key, value in dataset.items():
            
            sentences = dataset[key]["sentence"]
            sentiment_labels = dataset[key]["sentiment_label"]
            
            sentences_but_features, sentences_but_features_ind = self.rule_conjunct_extraction(dataset[key], rule=1)

            sentences = self.remove_extra_samples(sentences)
            sentiment_labels = self.remove_extra_samples(sentiment_labels)
            sentences_but_features = self.remove_extra_samples(sentences_but_features)
            sentences_but_features_ind = self.remove_extra_samples(sentences_but_features_ind)

            sentences, sentiment_labels = self.vectorize(sentences, sentiment_labels)
            sentences_but_features, sentences_but_features_ind = self.vectorize(sentences_but_features, sentences_but_features_ind)

            dataset_obj = DatasetSST2(sentences, sentiment_labels, transform = None)
            dataset_rule_features_obj = DatasetSST2(sentences_but_features, sentences_but_features_ind, transform = None)

            if self.config["distillation"] == True:
                additional_validation_datasets["dataset"][key] = dataset_obj
                additional_validation_datasets["rule_dataset"][key] = dataset_rule_features_obj
            else:
                additional_validation_datasets["dataset"][key] = dataset_obj

        return additional_validation_datasets
    
    def train_model(self, train_dataset, val_dataset, test_dataset):

        # Model to train
        print(self.config["model_name"]) 
        model = eval(self.config["model_name"]+"(self.config)") # {ELMo} × {Static, Fine-tuning} × {no distillation, distillation}
        model = model.to(self.config["device"])

        # Get train and val dataset
        train_sentences = train_dataset["sentence"]
        train_sentiment_labels = train_dataset["sentiment_label"]
        val_sentences = val_dataset["sentence"]
        val_sentiment_labels = val_dataset["sentiment_label"]

        # Create train and val rule features
        train_sentences_but_features, train_sentences_but_features_ind = self.rule_conjunct_extraction(train_dataset, rule=1)
        val_sentences_but_features, val_sentences_but_features_ind = self.rule_conjunct_extraction(val_dataset, rule=1)
        
        # Remove extra samples (making them multiples of batch_size)
        train_sentences = self.remove_extra_samples(train_sentences)
        train_sentiment_labels = self.remove_extra_samples(train_sentiment_labels)
        val_sentences = self.remove_extra_samples(val_sentences)
        val_sentiment_labels = self.remove_extra_samples(val_sentiment_labels)
        train_sentences_but_features = self.remove_extra_samples(train_sentences_but_features)
        train_sentences_but_features_ind = self.remove_extra_samples(train_sentences_but_features_ind)
        val_sentences_but_features = self.remove_extra_samples(val_sentences_but_features)
        val_sentences_but_features_ind = self.remove_extra_samples(val_sentences_but_features_ind)

        # Vectorize
        train_sentences, train_sentiment_labels = self.vectorize(train_sentences, train_sentiment_labels)
        val_sentences, val_sentiment_labels = self.vectorize(val_sentences, val_sentiment_labels)
        train_sentences_but_features, train_sentences_but_features_ind = self.vectorize(train_sentences_but_features, train_sentences_but_features_ind)
        val_sentences_but_features, val_sentences_but_features_ind = self.vectorize(val_sentences_but_features, val_sentences_but_features_ind)

        # Train 
        train_dataset = DatasetSST2(train_sentences, train_sentiment_labels, transform = None)
        train_dataset_rule_features = DatasetSST2(train_sentences_but_features, train_sentences_but_features_ind, transform = None)
        val_dataset = DatasetSST2(val_sentences, val_sentiment_labels, transform = None)
        val_dataset_rule_features = DatasetSST2(val_sentences_but_features, val_sentences_but_features_ind, transform = None)

        # Additional validation datasets
        if "additional_val_datasets" in self.config["callbacks"]:
            additional_validation_datasets = self.additional_validation_datasets(test_dataset)
        else:
            additional_validation_datasets = None

        if self.config["optimizer"] == "adadelta":
            optimizer = optim.Adadelta(model.parameters(), lr=self.config["learning_rate"], rho=0.95, eps=1e-6)
        elif self.config["optimizer"] == "adam":
            optimizer = optim.Adam(model.parameters(), lr=self.config["learning_rate"])

        if self.config["distillation"] == False:
            loss = nn.NLLLoss()
            history = Fit(self.config).fit_no_distillation(train_dataset, val_dataset, model, optimizer, loss, additional_validation_datasets=additional_validation_datasets)
        elif self.config["distillation"] == True:
            loss = nn.NLLLoss()
            history = Fit(self.config).fit_distillation(train_dataset, val_dataset, model, optimizer, loss, train_dataset_rule_features, val_dataset_rule_features, additional_validation_datasets=additional_validation_datasets)
        
        # Save the history of the model
        if not os.path.exists("assets/training_log/"):
            os.makedirs("assets/training_log/")
        with open("assets/training_log/"+self.config["asset_name"]+".pickle", "wb") as handle:
            pickle.dump(history, handle)

    def train_model_nested_cv(self, datasets_nested_cv):
        
        # Make paths
        if not os.path.exists("assets/training_log/"):
            os.makedirs("assets/training_log/")
        training_log = {}

        for k_fold in range(1, self.config["k_samples"]+1):
            for l_fold in range(1, self.config["l_samples"]+1):
                
                # Model to train 
                model = eval(self.config["model_name"]+"(self.config)") # {ELMo} × {Static, Fine-tuning} × {no distillation, distillation}
                model = model.to(self.config["device"])

                train_dataset = datasets_nested_cv["train_dataset_"+str(k_fold)+"_"+str(l_fold)]
                val_dataset = datasets_nested_cv["val_dataset_"+str(k_fold)+"_"+str(l_fold)]["test_dataset"]
                additional_val_datasets = datasets_nested_cv["val_dataset_"+str(k_fold)+"_"+str(l_fold)]

                # Get train and val dataset for this fold
                train_sentences = train_dataset["sentence"]
                train_sentiment_labels = train_dataset["sentiment_label"]
                val_sentences = val_dataset["sentence"]
                val_sentiment_labels = val_dataset["sentiment_label"]

                # Create train and val rule features
                train_sentences_but_features, train_sentences_but_features_ind = self.rule_conjunct_extraction(train_dataset, rule=1)
                val_sentences_but_features, val_sentences_but_features_ind = self.rule_conjunct_extraction(val_dataset, rule=1)
                
                # Remove extra samples
                train_sentences = self.remove_extra_samples(train_sentences)
                train_sentiment_labels = self.remove_extra_samples(train_sentiment_labels)
                val_sentences = self.remove_extra_samples(val_sentences)
                val_sentiment_labels = self.remove_extra_samples(val_sentiment_labels)
                train_sentences_but_features = self.remove_extra_samples(train_sentences_but_features)
                train_sentences_but_features_ind = self.remove_extra_samples(train_sentences_but_features_ind)
                val_sentences_but_features = self.remove_extra_samples(val_sentences_but_features)
                val_sentences_but_features_ind = self.remove_extra_samples(val_sentences_but_features_ind)

                # Vectorize
                train_sentences, train_sentiment_labels = self.vectorize(train_sentences, train_sentiment_labels)
                val_sentences, val_sentiment_labels = self.vectorize(val_sentences, val_sentiment_labels)
                train_sentences_but_features, train_sentences_but_features_ind = self.vectorize(train_sentences_but_features, train_sentences_but_features_ind)
                val_sentences_but_features, val_sentences_but_features_ind = self.vectorize(val_sentences_but_features, val_sentences_but_features_ind)

                # Train 
                train_dataset = DatasetSST2(train_sentences, train_sentiment_labels, transform = None)
                train_dataset_rule_features = DatasetSST2(train_sentences_but_features, train_sentences_but_features_ind, transform = None)
                val_dataset = DatasetSST2(val_sentences, val_sentiment_labels, transform = None)
                val_dataset_rule_features = DatasetSST2(val_sentences_but_features, val_sentences_but_features_ind, transform = None)

                if self.config["optimizer"] == "adadelta":
                    optimizer = optim.Adadelta(model.parameters(), lr=self.config["learning_rate"], rho=0.95, eps=1e-6)
                elif self.config["optimizer"] == "adam":
                    optimizer = optim.Adam(model.parameters(), lr=self.config["learning_rate"])

                if self.config["distillation"] == False:
                    loss = nn.NLLLoss()
                    history = Fit(self.config).fit_no_distillation(train_dataset, val_dataset, model, optimizer, loss, k_fold=k_fold, l_fold=l_fold)
                    training_log[str(k_fold)+"_"+str(l_fold)] = history
                elif self.config["distillation"] == True:
                    loss = nn.NLLLoss()
                    history = Fit(self.config).fit_distillation(train_dataset, val_dataset, model, optimizer, loss, train_dataset_rule_features, val_dataset_rule_features, k_fold=k_fold, l_fold=l_fold)
                    training_log[str(k_fold)+"_"+str(l_fold)] = history
        
        # Save the history of the model
        if not os.path.exists("assets/trained_models/"+self.config["asset_name"]):
            os.makedirs("assets/trained_models/"+self.config["asset_name"])
        with open("assets/trained_models/"+self.config["asset_name"]+"/training_log.pickle", "wb") as handle:
            pickle.dump(training_log, handle)