from config import *
from scripts.models.models import *

class Anchors_explanations(object):
    def __init__(self, config):
        self.config = config
        self.model = None
        self.word_index = None

    def vectorize(self, sentences):
        """
        tokenize each preprocessed sentence in dataset as sentence.split()
        encode each tokenized sentence as per vocabulary
        right pad each encoded tokenized sentence with 0 upto max_tokenized_sentence_len the dataset 
        """
        if tf.executing_eagerly():
            texts = np.array(sentences)
            vocab = [key for key in self.word_index.keys()]
            vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(standardize=None, split='whitespace', vocabulary=vocab, pad_to_max_tokens=False)
            encoded_texts = vectorize_layer(texts).numpy()
            padded_encoded_texts = tf.keras.preprocessing.sequence.pad_sequences(encoded_texts, value=0, padding='post', maxlen=53)
            return padded_encoded_texts
        else:
            toknized_texts = [text.split() for text in sentences]
            vocab = [key for key in self.word_index.keys()]
            encoded_texts = []
            for text in toknized_texts:
                encoded_text = []
                for word in text:
                    encoded_text.append(vocab.index(word)) 
                encoded_texts.append(encoded_text)
            encoded_texts = np.array(encoded_texts)
            padded_encoded_texts = tf.keras.preprocessing.sequence.pad_sequences(encoded_texts, value=0, padding='post', maxlen=53)
            return np.array(padded_encoded_texts)

    def prediction(self, text):
        x = self.vectorize(text)
        pred_prob_1 = self.model.predict(x, batch_size=1000)
        pred_prob_0 = 1 - pred_prob_1
        prob = np.concatenate((pred_prob_0, pred_prob_1), axis=1)
        return prob
    
    def create_explanations_nested_cv(self, datasets_nested_cv, word_vectors, word_index):

        with open("assets/results/"+self.config["asset_name"]+".pickle", 'rb') as handle:
            results = pickle.load(handle)
        results = pd.DataFrame(results)

        explanations = {"sentence":[],
                        "features":[], 
                        "sentiment_probability_prediction":[],
                        "sentiment_label":[],
                        "rule_label":[],
                        "contrast":[],
                        "anchor_explanation_object":[]}

        for k_fold in range(1, self.config["k_samples"]+1):
            for l_fold in range(1, self.config["l_samples"]+1):

                # Load trained model
                if self.config["model_name"] == "cnn":
                    self.model = cnn(self.config, word_vectors) # {Word2vec, Glove} × {Static, Fine-tuning} × {no distillation, distillation}
                    self.model.load_weights("assets/trained_models/"+self.config["asset_name"]+"/"+self.config["asset_name"]+"_"+str(k_fold)+"_"+str(l_fold)+".h5")
                self.word_index = word_index

                test_dataset = datasets_nested_cv["val_dataset_"+str(k_fold)+"_"+str(l_fold)]
                test_sentences = list(test_dataset.loc[(test_dataset["rule_label"]!=0)&(test_dataset["contrast"]==1)]['sentence'])
                sentiment_labels = list(test_dataset.loc[(test_dataset["rule_label"]!=0)&(test_dataset["contrast"]==1)]['sentiment_label'])
                rule_labels = list(test_dataset.loc[(test_dataset["rule_label"]!=0)&(test_dataset["contrast"]==1)]['rule_label'])
                contrasts = list(test_dataset.loc[(test_dataset["rule_label"]!=0)&(test_dataset["contrast"]==1)]['contrast'])

                test_sentences_vectorize = self.vectorize(test_sentences)
                probabilities = self.prediction(test_sentences)

                # Anchors
                model = 'en_core_web_md'
                spacy_model(model=model)
                nlp = spacy.load(model)
                explainer = AnchorText(predictor=self.prediction,
                                        sampling_strategy='unknown',       
                                        nlp=nlp,                           
                                        sample_proba=0.5)

                for index, test_datapoint in enumerate(test_sentences):
                    probability = [1 - probabilities[index].tolist()[0], probabilities[index].tolist()[0]]
                    sentiment_label = sentiment_labels[index]
                    rule_label = rule_labels[index]
                    contrast = contrasts[index]
                    tokenized_sentence = test_datapoint.split()
                    try:
                        explanation = explainer.explain(test_datapoint, threshold=0.95, batch_size=30)
                    except:
                        explanations['sentence'].append(test_datapoint)
                        explanations['features'].append(test_datapoint.split())
                        explanations['sentiment_label'].append(sentiment_label)
                        explanations['sentiment_probability_prediction'].append(probability)
                        explanations['rule_label'].append(rule_label)
                        explanations['contrast'].append(contrast)
                        explanations['anchor_explanation_object'].append("could_not_process")

        if not os.path.exists("assets/anchor_explanations/"):
            os.makedirs("assets/anchor_explanations/")
        with open("assets/anchor_explanations/"+self.config["asset_name"]+".pickle", "wb") as handle:
            pickle.dump(explanations, handle)