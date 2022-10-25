from config import *
from scripts.models.models import *

class Lime_explanations(object):
    """
    Calculate the LIME explanations for one-rule sentences in the test set
    """
    def __init__(self, config):
        self.config = config
        self.model = None
        self.word_index = None
    
    def handler(self, signum, frame):
        print("Forever is over!")
        raise Exception("end of time")
        
    def vectorize(self, sentences):
        """
        tokenize each preprocessed sentence in dataset as sentence.split()
        encode each tokenized sentence as per vocabulary
        right pad each encoded tokenized sentence with 0 upto max_tokenized_sentence_len the dataset 
        """
        vocab = [key for key in self.word_index.keys()]
        vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(standardize=None, split='whitespace', vocabulary=vocab)
        return vectorize_layer(np.array(sentences)).numpy()

    def prediction(self, text):
        x = self.vectorize(text)
        pred_prob_1 = self.model.predict(x, batch_size=50)
        pred_prob_0 = 1 - pred_prob_1
        prob = np.concatenate((pred_prob_0, pred_prob_1), axis=1)
        return prob
    
    def create_explanations(self, test_dataset, word_vectors, word_index):

        explanations = {"sentence":[],
                        "features":[], 
                        "sentiment_probability_prediction":[],
                        "sentiment_label":[],
                        "rule_label":[],
                        "contrast":[],
                        "LIME_explanation":[], 
                        "LIME_explanation_normalised":[]}
        
        # Load trained model
        self.model = eval(self.config["model_name"]+"(self.config, word_vectors)") # {Word2vec, Glove} × {Static, Fine-tuning} × {no distillation, distillation}
        self.model.load_weights("assets/trained_models/"+self.config["asset_name"]+".h5")
        self.word_index = word_index
        
        test_dataset = test_dataset["test_dataset_a_but_b_rule"]
        test_sentences = list(test_dataset['sentence'])
        sentiment_labels = list(test_dataset['sentiment_label'])
        rule_labels = list(test_dataset['rule_label'])
        contrasts = list(test_dataset['contrast'])

        test_sentences_vectorize = self.vectorize(test_sentences)
        probabilities = self.model.predict(x=test_sentences_vectorize)

        explainer = lime_text.LimeTextExplainer(class_names=["negative_sentiment", "positive_sentiment"], split_expression=" ", random_state=self.config["seed_value"])

        signal.signal(signal.SIGALRM, self.handler)

        for index, test_datapoint in enumerate(tqdm(test_sentences)):
            signal.alarm(100)
            probability = [1 - probabilities[index].tolist()[0], probabilities[index].tolist()[0]]
            sentiment_label = sentiment_labels[index]
            rule_label = rule_labels[index]
            contrast = contrasts[index]
            tokenized_sentence = test_datapoint.split()
            try:
                exp = explainer.explain_instance(test_datapoint, self.prediction, num_features = self.config["lime_num_of_features"], num_samples=self.config["lime_no_of_samples"])
            except Exception as exc:
                text = test_datapoint
                explanation = "couldn't process"
                explanations['sentence'].append(text)
                explanations['features'].append(text.split())
                explanations['sentiment_probability_prediction'].append(probability)
                explanations['sentiment_label'].append(sentiment_label)
                explanations['rule_label'].append(rule_label)
                explanations['contrast'].append(contrast)
                explanations['LIME_explanation'].append(explanation)
                explanations['LIME_explanation_normalised'].append(explanation)
                continue
            signal.alarm(0)
            text = []
            explanation = []
            explanation_normalised = []
            for word in test_datapoint.split():
                for weight in exp.as_list():
                    weight = list(weight)
                    if weight[0]==word:
                        text.append(word)
                        if weight[1] < 0:
                            weight_normalised_negative_class = abs(weight[1])*probability[0]
                            explanation_normalised.append(weight_normalised_negative_class)
                        elif weight[1] > 0:
                            weight_normalised_positive_class = abs(weight[1])*probability[1]
                            explanation_normalised.append(weight_normalised_positive_class)
                        explanation.append(weight[1])
            explanations['sentence'].append(test_datapoint)
            explanations['features'].append(text)
            explanations['sentiment_label'].append(sentiment_label)
            explanations['sentiment_probability_prediction'].append(probability)
            explanations['rule_label'].append(rule_label)
            explanations['contrast'].append(contrast)
            explanations['LIME_explanation'].append(explanation)
            explanations['LIME_explanation_normalised'].append(explanation_normalised)
        if not os.path.exists("assets/lime_explanations/"):
            os.makedirs("assets/lime_explanations/")
        with open("assets/lime_explanations/"+self.config["asset_name"]+".pickle", "wb") as handle:
            pickle.dump(explanations, handle)
    
    def create_explanations_nested_cv(self, datasets_nested_cv, word_vectors, word_index):

        explanations = {"sentence":[], 
                        "sentiment_probability_prediction":[],
                        "sentiment_label":[],
                        "rule_label":[],
                        "contrast":[],
                        "LIME_explanation":[], 
                        "LIME_explanation_normalised":[]}

        for k_fold in range(1, self.config["k_samples"]+1):
            for l_fold in range(1, self.config["l_samples"]+1):

                # Load trained model
                self.model = eval(self.config["model_name"]+"(self.config, word_vectors)") # {Word2vec, Glove} × {Static, Fine-tuning} × {no distillation, distillation}
                self.model.load_weights("assets/trained_models/"+self.config["asset_name"]+"/"+self.config["asset_name"]+"_"+str(k_fold)+"_"+str(l_fold)+".h5")
                self.word_index = word_index

                test_dataset = datasets_nested_cv["val_dataset_"+str(k_fold)+"_"+str(l_fold)]["test_dataset_a_but_b_rule"]
                test_sentences = list(test_dataset['sentence'])
                sentiment_labels = list(test_dataset['sentiment_label'])
                rule_labels = list(test_dataset['rule_label'])
                contrasts = list(test_dataset['contrast'])

                test_sentences_vectorize = self.vectorize(test_sentences)
                probabilities = self.model.predict(x=test_sentences_vectorize)

                explainer = lime_text.LimeTextExplainer(class_names=["negative_sentiment", "positive_sentiment"], split_expression=" ", random_state=self.config["seed_value"])

                for index, test_datapoint in enumerate(tqdm(test_sentences)):
                    probability = [1 - probabilities[index].tolist()[0], probabilities[index].tolist()[0]]
                    sentiment_label = sentiment_labels[index]
                    rule_label = rule_labels[index]
                    contrast = contrasts[index]
                    tokenized_sentence = test_datapoint.split()
                    try:
                        exp = explainer.explain_instance(test_datapoint, self.prediction, num_features = len(tokenized_sentence), num_samples=self.config["lime_no_of_samples"])
                    except:
                        text = test_datapoint
                        explanation = "couldn't process"
                        explanations['sentence'].append(text)
                        explanations['sentiment_probability_prediction'].append(probability)
                        explanations['sentiment_label'].append(sentiment_label)
                        explanations['rule_label'].append(rule_label)
                        explanations['contrast'].append(contrast)
                        explanations['LIME_explanation'].append(explanation)
                        explanations['LIME_explanation_normalised'].append(explanation)
                        continue
                    text = []
                    explanation = []
                    explanation_normalised = []
                    for word in test_datapoint.split():
                        for weight in exp.as_list():
                            weight = list(weight)
                            if weight[0]==word:
                                text.append(word)
                                if weight[1] < 0:
                                    weight_normalised_negative_class = abs(weight[1])*probability[0]
                                    explanation_normalised.append(weight_normalised_negative_class)
                                elif weight[1] > 0:
                                    weight_normalised_positive_class = abs(weight[1])*probability[1]
                                    explanation_normalised.append(weight_normalised_positive_class)
                                explanation.append(weight[1])
                    explanations['sentence'].append(text)
                    explanations['sentiment_label'].append(sentiment_label)
                    explanations['sentiment_probability_prediction'].append(probability)
                    explanations['rule_label'].append(rule_label)
                    explanations['contrast'].append(contrast)
                    explanations['LIME_explanation'].append(explanation)
                    explanations['LIME_explanation_normalised'].append(explanation_normalised)

        if not os.path.exists("assets/lime_explanations/"):
            os.makedirs("assets/lime_explanations/")
        with open("assets/lime_explanations/"+self.config["asset_name"]+".pickle", "wb") as handle:
            pickle.dump(explanations, handle)
