from config import *

class Preprocess_sentiment140_dataset(object):
    def __init__(self, config):
        self.config = config
    
    def preprocess_text(self, text, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' '):
        """
        Preprocess text as per Keras Tokenizer preprocess code. 
        Tokenize by just sentence.split()
        Whole process is similar to Keras Tokenizer
        """
        text = text.lower() # lower case
        maketrans = str.maketrans
        translate_dict = {c: split for c in filters}
        translate_map = maketrans(translate_dict) 
        text = text.translate(translate_map) # remove all punctuations and replace them with whitespace (because puntuations mark as a whitespace between words)
        return text

    def conjunction_analysis(self, dataset, rule_keywords=['but', 'yet', 'though', 'while']):
        """
        Count the sentences labeled with a particular rule like A-but-B in the dataset during dataset creation
        Perform a conjunction analysis for that rule in the sentences
        Check if both counts are equal
        If not equal, remove the datapoints which has the rule label but fails on its conjunction analysis
        """
        rule_keywords = set(rule_keywords)
        rule_labels = []
        contrasts = []
        for index, sentence in enumerate(dataset['sentence']): # Check for any rule structure in no rule sentences and remove any sentence containing a rule structure
            tokenized_sentence = sentence.split()
            if (set(tokenized_sentence).intersection(rule_keywords)==set(["but"]) and tokenized_sentence.index('but') != 0 and tokenized_sentence.index('but') != len(tokenized_sentence)-1 and tokenized_sentence.count('but') == 1):
                rule_labels.append(1)
                contrasts.append(1)
            elif (set(tokenized_sentence).intersection(rule_keywords)==set(["yet"]) and tokenized_sentence.index('yet') != 0 and tokenized_sentence.index('yet') != len(tokenized_sentence)-1 and tokenized_sentence.count('yet') == 1):
                rule_labels.append(2)
                contrasts.append(1)
            elif (set(tokenized_sentence).intersection(rule_keywords)==set(["though"]) and tokenized_sentence.index('though') != 0 and tokenized_sentence.index('though') != len(tokenized_sentence)-1 and tokenized_sentence.count('though') == 1):
                rule_labels.append(3)
                contrasts.append(1)
            elif (set(tokenized_sentence).intersection(rule_keywords)==set(["while"]) and tokenized_sentence.index('while') != 0 and tokenized_sentence.index('while') != len(tokenized_sentence)-1 and tokenized_sentence.count('while') == 1):
                rule_labels.append(4)
                contrasts.append(1)
            else:
                rule_labels.append(0)
                contrasts.append(0)
        dataset["rule_label"] = rule_labels
        dataset["contrast"] = contrasts
        return dataset
    
    def create_rule_masks(self, dataset):
        """
        create rule masks for each sentence in the dataset
        """
        rule_label_masks = []
        for index, sentence in enumerate(list(dataset['sentence'])):
            tokenized_sentence = sentence.split()
            rule_label = dataset['rule_label'][index]
            contrast = dataset['contrast'][index]
            try:
                if rule_label == 1 and contrast == 1:
                    a_part_tokenized_sentence = tokenized_sentence[:tokenized_sentence.index("but")]
                    b_part_tokenized_sentence = tokenized_sentence[tokenized_sentence.index("but")+1:]
                    rule_label_mask = [0]*len(a_part_tokenized_sentence) + [0]*len(["but"]) + [1]*len(b_part_tokenized_sentence)
                    rule_label_masks.append(rule_label_mask)

                elif rule_label == 2 and contrast == 1:
                    a_part_tokenized_sentence = tokenized_sentence[:tokenized_sentence.index("yet")]
                    b_part_tokenized_sentence = tokenized_sentence[tokenized_sentence.index("yet")+1:]
                    rule_label_mask = [0]*len(a_part_tokenized_sentence) + [0]*len(["yet"]) + [1]*len(b_part_tokenized_sentence)
                    rule_label_masks.append(rule_label_mask)

                elif rule_label == 3 and contrast == 1:
                    a_part_tokenized_sentence = tokenized_sentence[:tokenized_sentence.index("though")]
                    b_part_tokenized_sentence = tokenized_sentence[tokenized_sentence.index("though")+1:]
                    rule_label_mask = [1]*len(a_part_tokenized_sentence) + [0]*len(["though"]) + [0]*len(b_part_tokenized_sentence)
                    rule_label_masks.append(rule_label_mask)

                elif rule_label == 4 and contrast == 1:
                    a_part_tokenized_sentence = tokenized_sentence[:tokenized_sentence.index("while")]
                    b_part_tokenized_sentence = tokenized_sentence[tokenized_sentence.index("while")+1:]
                    rule_label_mask = [1]*len(a_part_tokenized_sentence) + [0]*len(["while"]) + [0]*len(b_part_tokenized_sentence)
                    rule_label_masks.append(rule_label_mask)
    
                else:
                    mask_length = len(tokenized_sentence)
                    rule_label_mask = [1]*mask_length
                    rule_label_masks.append(rule_label_mask)
            except:
                mask_length = len(tokenized_sentence)
                rule_label_mask = [1]*mask_length
                rule_label_masks.append(rule_label_mask)
        dataset["rule_mask"] = rule_label_masks
        return dataset

    def preprocess(self, dataset):

        # Select columns
        dataset = dataset[['text', 'target']]

        # Converting sentiment value of 4 to 1 and removing neutral sentences
        dataset['target'].replace({4: 1}, inplace=True)
        dataset = dataset.loc[(dataset["target"]==0)|(dataset["target"]==1)]

        # Renaming the text column to sentence and target to sentiment label
        dataset = dataset.rename(columns={'text': 'sentence'})
        dataset = dataset.rename(columns={'target': 'sentiment_label'})

        # Preprocess sentences
        preprocessed_sentences = [self.preprocess_text(sentence) for sentence in list(dataset['sentence'])]
        dataset["sentence"] = preprocessed_sentences

        # Perform conjunction analysis on sentences and provide rule_label & contrast labels
        dataset = self.conjunction_analysis(dataset)
        
        # Create rule masks
        dataset = self.create_rule_masks(dataset)

        # Balance the dataset between one rule and no rule
        dataset_one_rule = dataset.loc[dataset["rule_label"]!=0]
        dataset_no_rule_pos = dataset.loc[(dataset["rule_label"]==0)&(dataset["sentiment_label"]==1)]
        dataset_no_rule_neg = dataset.loc[(dataset["rule_label"]==0)&(dataset["sentiment_label"]==0)]
        dataset_no_rule_sample_pos = dataset_no_rule_pos.sample(n=50000, random_state=self.config["seed_value"])
        dataset_no_rule_sample_neg = dataset_no_rule_neg.sample(n=50000, random_state=self.config["seed_value"])
        dataset = pd.concat([dataset_one_rule, dataset_no_rule_sample_pos, dataset_no_rule_sample_neg])
        dataset = dataset.sample(frac=1, random_state=self.config["seed_value"]).reset_index(drop=True)

        # Balance the sentiment label distribution between but rule sentences
        dataset_but_rule_neg = dataset.loc[(dataset["rule_label"]==1)&(dataset["sentiment_label"]==0)]
        dataset_every_sentence_except_but_neg = dataset.loc[dataset.index.difference(dataset_but_rule_neg.index), ]
        dataset_but_rule_sample_neg = dataset_but_rule_neg.sample(n=50000, random_state=self.config["seed_value"])
        dataset = pd.concat([dataset_every_sentence_except_but_neg, dataset_but_rule_sample_neg])
        dataset = dataset.sample(frac=1, random_state=self.config["seed_value"]).reset_index(drop=True)

        dataset = dataset[(dataset['rule_label']==0)|(dataset['rule_label']==1)].reset_index(drop=True)
        dataset = dataset.sample(n=50000, random_state=self.config["seed_value"]).reset_index(drop=True)
        
        return dataset
