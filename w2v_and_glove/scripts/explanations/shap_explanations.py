from config import *
from scripts.models.models import *

class Shap_explanations(object):
    """
    Calculate the LIME explanations for one-rule sentences in the test set
    """
    def __init__(self, config):
        self.config = config
        self.model = None
        self.word_index = None

    def prediction(self, x):
        return self.model.predict(x)
    
    def vectorize(self, texts):
        if tf.executing_eagerly():
            texts = np.array(texts)
            vocab = [key for key in self.word_index.keys()]
            vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(standardize=None, split='whitespace', vocabulary=vocab, pad_to_max_tokens=False)
            encoded_texts = vectorize_layer(texts).numpy()
            padded_encoded_texts = tf.keras.preprocessing.sequence.pad_sequences(encoded_texts, value=0, padding='post', maxlen=53)
            return padded_encoded_texts
        else:
            toknized_texts = [text.split() for text in texts]
            encoded_texts = []
            for text in toknized_texts:
                encoded_text = []
                for word in text:
                    encoded_text.append(self.vocab.index(word)) 
                encoded_texts.append(encoded_text)
            encoded_texts = np.array(encoded_texts)
            padded_encoded_texts = tf.keras.preprocessing.sequence.pad_sequences(encoded_texts, value=0, padding='post', maxlen=53)
            return np.array(padded_encoded_texts)
    
    def create_explanations(self, train_dataset, test_dataset, word_vectors, word_index):

        explanations = {"sentence":[], 
                            "sentiment_probability_prediction":[],
                            "rule_label":[],
                            "contrast":[],
                            "base_value":[],
                            "SHAP_explanation":[], 
                            "SHAP_explanation_normalised":[]}
        
        # Load trained model
        self.model = eval(self.config["model_name"]+"(self.config, word_vectors)") # {Word2vec, Glove} × {Static, Fine-tuning} × {no distillation, distillation}
        self.model.load_weights("assets/trained_models/"+self.config["asset_name"]+".h5")
        self.word_index = word_index

        test_dataset = test_dataset["test_dataset_a_but_b_rule"]
        train_sentences = list(train_dataset.loc[(train_dataset["rule_label"]!=0)&(train_dataset["contrast"]==1)]['sentence'])
        test_sentences = list(test_dataset['sentence'])
        test_rule_labels = list(test_dataset['rule_label'])
        test_contrasts = list(test_dataset['contrast'])
        train_sentences_vectorize = self.vectorize(train_sentences)
        test_sentences_vectorize = self.vectorize(test_sentences)
        probabilities = self.model.predict(x=test_sentences_vectorize)

        exp_explainer = shap.Explainer(model=self.prediction, masker=train_sentences_vectorize[:len(test_sentences_vectorize)*100], algorithm="permutation")
        shap_explanations = exp_explainer(test_sentences_vectorize)
        base_values = shap_explanations.base_values
        shap_values = shap_explanations.values
        # explainer = shap.KernelExplainer(self.prediction, train_sentences_vectorize[:len(test_sentences_vectorize)*100])
        # shap_values = explainer.shap_values(test_sentences_vectorize, nsamples=1000)

        for index, test_datapoint in enumerate(tqdm(test_sentences)):
            probability = [1 - probabilities[index].tolist()[0], probabilities[index].tolist()[0]]
            rule_label = test_rule_labels[index]
            contrast = test_contrasts[index]
            base_value = base_values[index]
            shap_value = shap_values[index]
            shap_value_normalised = [abs(value) for value in shap_value]
            tokenized_sentence = test_datapoint.split()
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

    def create_explanations_nested_cv(self, datasets_nested_cv, word_vectors, word_index):

            explanations = {"sentence":[], 
                            "sentiment_probability_prediction":[],
                            "rule_label":[],
                            "contrast":[],
                            "base_value":[],
                            "SHAP_explanation":[], 
                            "SHAP_explanation_normalised":[]}

            for k_fold in range(1, self.config["k_samples"]+1):
                for l_fold in range(1, self.config["l_samples"]+1):

                    # Load trained model
                    self.model = eval(self.config["model_name"]+"(self.config, word_vectors)") # {Word2vec, Glove} × {Static, Fine-tuning} × {no distillation, distillation}
                    self.model.load_weights("assets/trained_models/"+self.config["asset_name"]+"/"+self.config["asset_name"]+"_"+str(k_fold)+"_"+str(l_fold)+".h5")
                    self.word_index = word_index

                    train_dataset = datasets_nested_cv["train_dataset_"+str(k_fold)+"_"+str(l_fold)]
                    test_dataset = datasets_nested_cv["val_dataset_"+str(k_fold)+"_"+str(l_fold)]["test_dataset_a_but_b_rule"]
                    train_sentences = list(train_dataset.loc[(train_dataset["rule_label"]!=0)&(train_dataset["contrast"]==1)]['sentence'])
                    test_sentences = list(test_dataset['sentence'])
                    test_rule_labels = list(test_dataset['rule_label'])
                    test_contrasts = list(test_dataset['contrast'])
                    train_sentences_vectorize = self.vectorize(train_sentences)
                    test_sentences_vectorize = self.vectorize(test_sentences)

                    probabilities = self.model.predict(x=test_sentences_vectorize)
                    exp_explainer = shap.Explainer(model=self.prediction, masker=train_sentences_vectorize[:len(test_sentences_vectorize)*10], algorithm="permutation")
                    shap_explanations = exp_explainer(test_sentences_vectorize)
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