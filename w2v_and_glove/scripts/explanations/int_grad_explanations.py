from config import *
from scripts.models.models import *

class Int_grad_explanations(object):
    """
    Calculate the Int-grad explanations for one-rule sentences in the test set
    """
    def __init__(self, config):
        self.config = config
        self.model = None
        self.word_index = None
    
    def batch(self, iterable, batch_size=10):
        l = len(iterable)
        for ndx in range(0, l, batch_size):
            yield iterable[ndx:min(ndx + batch_size, l)]

    def vectorize(self, sentences):
        """
        tokenize each preprocessed sentence in dataset as sentence.split()
        encode each tokenized sentence as per vocabulary
        right pad each encoded tokenized sentence with 0 upto max_tokenized_sentence_len the dataset 
        """
        if type(sentences) != list:
            sentences = [sentences]
        vocab = [key for key in self.word_index.keys()]
        vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(standardize=None, split='whitespace', vocabulary=vocab)
        vectorized_sentences = vectorize_layer(np.array(sentences)).numpy()
        return vectorized_sentences

    def prediction(self, text):
        pred_prob_1 = self.model.predict(text, batch_size=1000)
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
                        "INT_GRAD_explanation":[], 
                        "INT_GRAD_explanation_normalised":[]}

        # Load trained model
        self.model = eval(self.config["model_name"]+"(self.config, word_vectors)") # {Word2vec, Glove} × {Static, Fine-tuning} × {no distillation, distillation}
        self.model.load_weights("assets/trained_models/"+self.config["asset_name"]+".h5")
        self.word_index = word_index

        test_dataset = test_dataset["test_dataset_a_but_b_rule"]
        test_sentences = list(test_dataset['sentence'])
        sentiment_labels = list(test_dataset['sentiment_label'])
        rule_labels = list(test_dataset['rule_label'])
        contrasts = list(test_dataset['contrast'])

        # embed_layer = self.model.get_layer(index=1)
        # new_model = tf.keras.Sequential()
        # for layer in self.model.layers[2:]:
        #     new_model.add(layer)

        integrated_grad_cam = IntegratedGradients(self.model, 
                                                layer = self.model.layers[1],
                                                method="gausslegendre", 
                                                n_steps=200, 
                                                internal_batch_size=32)

        # # Single instance
        # for index, sentence in enumerate(tqdm(test_sentences)):
        #     test_sentences_vectorize = self.vectorize(sentence)
        #     probabilities = self.model.predict(x=test_sentences_vectorize)
        #     try:
        #         exp = integrated_grad_cam.explain(test_sentences_vectorize, target=probabilities)
        #     except:
        #         explanations["sentence"].append(sentence)
        #         explanations["features"].append(sentence.split())
        #         explanations["sentiment_probability_prediction"].append(probabilities[index])
        #         explanations["sentiment_label"].append(sentiment_labels[index])
        #         explanations["rule_label"].append(rule_labels[index])
        #         explanations["contrast"].append(contrasts[index])
        #         explanations["INT_GRAD_explanation"].append("could_not_process")
        #         explanations["INT_GRAD_explanation_normalised"].append("could_not_process")
        #     attributions = exp.attributions
        #     attributions = [att.sum(axis=2) for att in attributions]
        #     attribution = attributions[0][0]
        #     normalised_attribution = []
        #     probability_0_1 = [1 - probabilities[0].tolist()[0], probabilities[0].tolist()[0]]
        #     for att_value in attribution:
        #         if att_value < 0:
        #             weight_normalised_negative_class = abs(att_value)*probability_0_1[0]
        #             normalised_attribution.append(weight_normalised_negative_class)
        #         elif att_value > 0:
        #             weight_normalised_positive_class = abs(att_value)*probability_0_1[1]
        #             normalised_attribution.append(weight_normalised_positive_class)
        #     explanations["sentence"].append(sentence)
        #     explanations["features"].append(sentence.split())
        #     explanations["sentiment_probability_prediction"].append(probabilities[0])
        #     explanations["sentiment_label"].append(sentiment_labels[index])
        #     explanations["rule_label"].append(rule_labels[index])
        #     explanations["contrast"].append(contrasts[index])
        #     explanations["INT_GRAD_explanation"].append(list(attribution))
        #     explanations["INT_GRAD_explanation_normalised"].append(normalised_attribution)
        
        # Batched
        test_sentences_batched = [input for input in self.batch(test_sentences, self.config["mini_batch_size"])]
        sentiment_labels_batched = [input for input in self.batch(sentiment_labels, self.config["mini_batch_size"])]
        rule_labels_batched = [input for input in self.batch(rule_labels, self.config["mini_batch_size"])]
        contrasts_batched = [input for input in self.batch(contrasts, self.config["mini_batch_size"])]
        
        for index, sentence in enumerate(tqdm(test_sentences_batched)):
            sentiment_labels = sentiment_labels_batched[index]
            rule_labels = rule_labels_batched[index]
            contrasts = contrasts_batched[index]
            test_sentences_vectorize = self.vectorize(sentence)
            probabilities = self.model.predict(x=test_sentences_vectorize)
            exp = integrated_grad_cam.explain(test_sentences_vectorize, target=probabilities)
            attributions = exp.attributions
            attributions = [att.sum(axis=2) for att in attributions][0]
            for index, attribution in enumerate(attributions):
                probability = [1 - probabilities[index].tolist()[0], probabilities[index].tolist()[0]]
                normalised_attribution = []
                for att_value in attribution:
                    if att_value < 0:
                        weight_normalised_negative_class = abs(att_value)*probability[0]
                        normalised_attribution.append(weight_normalised_negative_class)
                    elif att_value > 0:
                        weight_normalised_positive_class = abs(att_value)*probability[1]
                        normalised_attribution.append(weight_normalised_positive_class)
                explanations["sentence"].append(sentence[index])
                explanations["features"].append(sentence[index].split())
                explanations["sentiment_probability_prediction"].append(probability)
                explanations["sentiment_label"].append(sentiment_labels[index])
                explanations["rule_label"].append(rule_labels[index])
                explanations["contrast"].append(contrasts[index])
                explanations["INT_GRAD_explanation"].append(list(attribution))
                explanations["INT_GRAD_explanation_normalised"].append(normalised_attribution)

        # Save the explanations
        if not os.path.exists("assets/int_grad_explanations/"):
            os.makedirs("assets/int_grad_explanations/")
        with open("assets/int_grad_explanations/"+self.config["asset_name"]+".pickle", "wb") as handle:
            pickle.dump(explanations, handle)
    
    def create_explanations_nested_cv(self, datasets_nested_cv, word_vectors, word_index):

        explanations = {"sentence":[],
                        "features":[], 
                        "sentiment_probability_prediction":[],
                        "sentiment_label":[],
                        "rule_label":[],
                        "contrast":[],
                        "INT_GRAD_explanation":[], 
                        "INT_GRAD_explanation_normalised":[]}

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

                integrated_grad_cam = IntegratedGradients(self.model, 
                                                            layer = self.model.layers[1],
                                                            method="gausslegendre", 
                                                            n_steps=100, 
                                                            internal_batch_size=32)

                # Batched
                test_sentences_batched = [input for input in self.batch(test_sentences)]
                sentiment_labels_batched = [input for input in self.batch(sentiment_labels)]
                rule_labels_batched = [input for input in self.batch(rule_labels)]
                contrasts_batched = [input for input in self.batch(contrasts)]

                for index, sentence in enumerate(tqdm(test_sentences_batched)):
                    sentiment_labels = sentiment_labels_batched[index]
                    rule_labels = rule_labels_batched[index]
                    contrasts = contrasts_batched[index]
                    test_sentences_vectorize = self.vectorize(sentence)
                    probabilities = self.model.predict(test_sentences_vectorize)
                    exp = integrated_grad_cam.explain(test_sentences_vectorize, target=probabilities)
                    attributions = exp.attributions
                    attributions = [att.sum(axis=2) for att in attributions][0]
                    for index, attribution in enumerate(attributions):
                        probability = [1 - probabilities[index].tolist()[0], probabilities[index].tolist()[0]]
                        normalised_attribution = []
                        for att_value in attribution:
                            if att_value < 0:
                                weight_normalised_negative_class = abs(att_value)*probability[0]
                                normalised_attribution.append(weight_normalised_negative_class)
                            elif att_value > 0:
                                weight_normalised_positive_class = abs(att_value)*probability[1]
                                normalised_attribution.append(weight_normalised_positive_class)
                        explanations["sentence"].append(sentence[index])
                        explanations["features"].append(sentence[index].split())
                        explanations["sentiment_probability_prediction"].append(probability)
                        explanations["sentiment_label"].append(sentiment_labels[index])
                        explanations["rule_label"].append(rule_labels[index])
                        explanations["contrast"].append(contrasts[index])
                        explanations["INT_GRAD_explanation"].append(list(attribution))
                        explanations["INT_GRAD_explanation_normalised"].append(normalised_attribution)
                
                # exp = integrated_grad_cam.explain(test_sentences_vectorize, target=probabilities)
                # attributions = exp.attributions
                # attributions = [att.sum(axis=2) for att in attributions]

                # # Normalize attributions
                # normalised_attributions = []
                # for index, attribution in enumerate(attributions[0]):
                #     probability = [1 - probabilities[index].tolist()[0], probabilities[index].tolist()[0]]
                #     normalised_attribution = []
                #     for att_value in attribution:
                #         if att_value < 0:
                #             weight_normalised_negative_class = abs(att_value)*probability[0]
                #             normalised_attribution.append(weight_normalised_negative_class)
                #         elif att_value > 0:
                #             weight_normalised_positive_class = abs(att_value)*probability[1]
                #             normalised_attribution.append(weight_normalised_positive_class)
                #     normalised_attributions.append(normalised_attribution)

                # # Append to explanations
                # for index, sentence in enumerate(tqdm(test_sentences)):
                #     explanations["sentence"].append(sentence)
                #     explanations["features"].append(sentence.split())
                #     explanations["sentiment_probability_prediction"].append(probabilities[index])
                #     explanations["sentiment_label"].append(sentiment_labels[index])
                #     explanations["rule_label"].append(rule_labels[index])
                #     explanations["contrast"].append(contrasts[index])
                #     explanations["INT_GRAD_explanation"].append(list(attributions[0][index]))
                #     explanations["INT_GRAD_explanation_normalised"].append(normalised_attributions[index])

        # Save the explanations
        if not os.path.exists("assets/int_grad_explanations/"):
            os.makedirs("assets/int_grad_explanations/")
        with open("assets/int_grad_explanations/"+self.config["asset_name"]+".pickle", "wb") as handle:
            pickle.dump(explanations, handle)
