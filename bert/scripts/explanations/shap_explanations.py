from config import *
from scripts.models.models import *

class Shap_explanations(object):
    """
    Calculate the LIME explanations for one-rule sentences in the test set
    """
    def __init__(self, config):
        self.config = config
        self.model = None
    
    def prediction(self, x):
        return self.model.predict(x)
    
    def batch(self, iterable, batch_size=1):
        l = len(iterable)
        for ndx in range(0, l, batch_size):
            yield iterable[ndx:min(ndx + batch_size, l)]

    def vectorize(self, texts):
        """
        tokenize each preprocessed sentence in dataset using bert tokenizer
        """
        max_len = 76
        input_ids = []
        attention_masks = []
        if type(texts) != list:
            texts = [texts]
        for sentence in texts:
            tokenized_context = self.config["bert_tokenizer"].encode(sentence)
            input_id = tokenized_context.ids
            attention_mask = [1] * len(input_id)
            input_ids.append(input_id)
            attention_masks.append(attention_mask)
        for index, input_id in enumerate(input_ids):
            padding_length = max_len - len(input_ids[index])
            input_ids[index] = input_ids[index] + ([0] * padding_length)
            attention_masks[index] = attention_masks[index] + ([0] * padding_length)
        return np.array(input_ids), np.array(attention_masks)
    
    def create_shap_explanations(self, train_dataset, test_dataset):

        explanations = {"sentence":[],
                        "features":[], 
                        "sentiment_probability_prediction":[],
                        "sentiment_label":[],
                        "rule_label":[],
                        "contrast":[],
                        "SHAP_explanation":[], 
                        "SHAP_explanation_normalised":[]}
        
        # Load trained model
        self.model = eval(self.config["model_name"]+"(self.config)")
        self.model.load_weights("assets/trained_models/"+self.config["asset_name"]+"_ckpt").expect_partial()

        train_sentences = list(train_dataset.loc[(train_dataset["rule_label"]!=0)&(train_dataset["contrast"]==1)]['sentence'])
        test_dataset = test_dataset["test_dataset_a_but_b_rule"]
        test_sentences = list(test_dataset['sentence'])
        sentiment_labels = list(test_dataset['sentiment_label'])
        rule_labels = list(test_dataset['rule_label'])
        contrasts = list(test_dataset['contrast'])

        # Batched (USE TranSHAP to solve the tokenizer issue)
        train_sentences_batched = [input for input in self.batch(train_sentences, self.config["mini_batch_size"]*10)]
        test_sentences_batched = [input for input in self.batch(test_sentences, self.config["mini_batch_size"])]
        sentiment_labels_batched = [input for input in self.batch(sentiment_labels, self.config["mini_batch_size"])]
        rule_labels_batched = [input for input in self.batch(rule_labels, self.config["mini_batch_size"])]
        contrasts_batched = [input for input in self.batch(contrasts, self.config["mini_batch_size"])]

        for index, test_sentence in enumerate(test_sentences_batched):
            train_sentence = train_sentences_batched[index]
            sentiment_labels = sentiment_labels_batched[index]
            rule_labels = rule_labels_batched[index]
            contrasts = contrasts_batched[index]
            train_sentences_vectorize, train_sentences_vectorize_att_masks = self.vectorize(train_sentence)
            test_sentences_vectorize, test_sentences_attention_masks = self.vectorize(test_sentence)
            probabilities = self.model.predict(x=[test_sentences_vectorize, test_sentences_attention_masks])
            exp_explainer = shap.KernelExplainer(self.prediction, shap.sample(train_sentences_vectorize))
            shap_explanations = exp_explainer.shap_values(test_sentences_vectorize, nsamples=64, l1_reg="aic")
            for index, test_datapoint in enumerate(test_sentence):
                probability = [1 - probabilities[index].tolist()[0], probabilities[index].tolist()[0]]
                sentiment_label = sentiment_labels[index]
                rule_label = rule_labels[index]
                contrast = contrasts[index]
                shap_value =  shap_explanations[index].tolist()
                shap_value_normalised = [abs(value) for value in shap_value]
                tokenized_sentence = test_datapoint.split()
                explanations['sentence'].append(test_datapoint)
                explanations['features'].append(tokenized_sentence)
                explanations['sentiment_probability_prediction'].append(probability)
                explanations["sentiment_label"].append(sentiment_label)
                explanations['rule_label'].append(rule_label)
                explanations['contrast'].append(contrast)
                explanations['SHAP_explanation'].append(shap_value)
                explanations['SHAP_explanation_normalised'].append(shap_value_normalised)

        if not os.path.exists("assets/shap_explanations/"):
            os.makedirs("assets/shap_explanations/")
        with open("assets/shap_explanations/"+self.config["asset_name"]+".pickle", "wb") as handle:
            pickle.dump(explanations, handle)
    
    def create_shap_explanations_nested_cv(self, datasets_nested_cv):

        explanations = {"sentence":[], 
                        "sentiment_probability_prediction":[],
                        "rule_label":[],
                        "contrast":[],
                        "base_value":[],
                        "SHAP_explanation":[], 
                        "SHAP_explanation_normalised":[]}

        # self.model = eval(self.config["model_name"]+"(self.config)")
        for k_fold in range(1, self.config["k_samples"]+1):
            for l_fold in range(1, self.config["l_samples"]+1):
                # try:
                #     self.model.load_weights("assets/trained_models/"+self.config["asset_name"]+"/"+self.config["asset_name"]+"_"+str(k_fold)+"_"+str(l_fold)+"_ckpt")
                # except:
                self.model = cnn(self.config)
                self.model.load_weights("assets/trained_models/"+self.config["asset_name"]+"/"+self.config["asset_name"]+"_"+str(k_fold)+"_"+str(l_fold)+".h5")
                train_dataset = datasets_nested_cv["train_dataset_"+str(k_fold)+"_"+str(l_fold)]
                test_dataset = datasets_nested_cv["val_dataset_"+str(k_fold)+"_"+str(l_fold)]
                train_sentences = list(train_dataset.loc[(train_dataset["rule_label"]!=0)&(train_dataset["contrast"]==1)]['sentence'])
                test_sentences = list(test_dataset.loc[(test_dataset["rule_label"]!=0)&(test_dataset["contrast"]==1)]['sentence'])
                test_rule_labels = list(test_dataset.loc[(test_dataset["rule_label"]!=0)&(test_dataset["contrast"]==1)]['rule_label'])
                test_contrasts = list(test_dataset.loc[(test_dataset["rule_label"]!=0)&(test_dataset["contrast"]==1)]['contrast'])
                train_sentences_vectorize, train_sentences_vectorize_att_masks = self.vectorize(train_sentences)
                test_sentences_vectorize, test_sentences_vectorize_att_masks = self.vectorize(test_sentences)
                probabilities = self.model_nested_cv.predict(x=[test_sentences_vectorize, test_sentences_vectorize_att_masks])
                exp_explainer = shap.Explainer(model=self.prediction, algorithm="permutation")
                shap_explanations = exp_explainer(test_sentences)
                base_values = shap_explanations.base_values
                shap_values = shap_explanations.values
                for index, test_datapoint in enumerate(tqdm(test_sentences)):
                    probability = [1 - probabilities[index].tolist()[0], probabilities[index].tolist()[0]]
                    rule_label = test_rule_labels[index]
                    contrast = test_contrasts[index]
                    base_value = base_values[index]
                    shap_value = shap_values[index]
                    shap_value_normalised = [abs(value) for value in shap_value]
                    tokenized_sentence = test_datapoint.split()
                    if rule_label == 1:
                        word_index_value = tokenized_sentence.index('but')
                    explanations['sentence'].append(tokenized_sentence)
                    explanations['sentiment_probability_prediction'].append(probability)
                    explanations['rule_label'].append(rule_label)
                    explanations['contrast'].append(contrast)
                    explanations['base_value'].append(base_value)
                    explanations['SHAP_explanation'].append(shap_value)
                    explanations['SHAP_explanation_normalised'].append(shap_value_normalised)
        if not os.path.exists("assets/shap_explanations/"):
            os.makedirs("assets/shap_explanations/")
        with open("assets/shap_explanations/"+self.config["asset_name"]+".pickle", "wb") as handle:
            pickle.dump(explanations, handle)