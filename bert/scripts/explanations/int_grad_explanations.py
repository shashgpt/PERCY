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
        tokenize each preprocessed sentence in dataset using bert tokenizer
        """
        max_len = 0
        input_ids = []
        attention_masks = []
        if type(sentences) != list:
            sentences = [sentences]
        for sentence in sentences:
            tokenized_context = self.config["bert_tokenizer"].encode(sentence)
            input_id = tokenized_context.ids
            attention_mask = [1] * len(input_id)
            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            if len(input_id) > max_len:
                max_len = len(input_id)
        for index, input_id in enumerate(input_ids):
            padding_length = max_len - len(input_ids[index])
            input_ids[index] = input_ids[index] + ([0] * padding_length)
            attention_masks[index] = attention_masks[index] + ([0] * padding_length)
        return np.array(input_ids), np.array(attention_masks)

    def prediction(self, text):
        x, att_masks = self.vectorize(text)
        pred_prob_1 = self.model.predict([x, att_masks], batch_size=1000)
        pred_prob_0 = 1 - pred_prob_1
        prob = np.concatenate((pred_prob_0, pred_prob_1), axis=1)
        return prob
    
    def create_explanations(self, test_dataset):

        explanations = {"sentence":[],
                        "features":[], 
                        "sentiment_probability_prediction":[],
                        "sentiment_label":[],
                        "rule_label":[],
                        "contrast":[],
                        "INT_GRAD_explanation":[], 
                        "INT_GRAD_explanation_normalised":[]}

        # Load trained model
        self.model = eval(self.config["model_name"]+"(self.config)")
        # self.model.summary(line_length = 150) 
        self.model.load_weights("assets/trained_models/"+self.config["asset_name"]+"_ckpt")

        test_dataset = test_dataset["test_dataset_a_but_b_rule"]
        test_sentences = list(test_dataset['sentence'])
        sentiment_labels = list(test_dataset['sentiment_label'])
        rule_labels = list(test_dataset['rule_label'])
        contrasts = list(test_dataset['contrast'])

        integrated_grad_cam = IntegratedGradients(self.model, 
                                                layer = self.model.layers[0].layers[0].embeddings,
                                                method="gausslegendre", 
                                                n_steps=100, 
                                                internal_batch_size=32)

        # # Single instance
        # for index, sentence in enumerate(tqdm(test_sentences)):
        #     test_sentences_vectorize, test_sentences_attention_masks = self.vectorize(sentence)
        #     probabilities = self.model.predict(x=[test_sentences_vectorize, test_sentences_attention_masks])
        #     # try:
        #     exp = integrated_grad_cam.explain(test_sentences_vectorize, target=probabilities)
        #     # except:
        #     #     explanations["sentence"].append(sentence)
        #     #     explanations["features"].append(sentence.split())
        #     #     explanations["sentiment_probability_prediction"].append(probabilities[index])
        #     #     explanations["sentiment_label"].append(sentiment_labels[index])
        #     #     explanations["rule_label"].append(rule_labels[index])
        #     #     explanations["contrast"].append(contrasts[index])
        #     #     explanations["INT_GRAD_explanation"].append("could_not_process")
        #     #     explanations["INT_GRAD_explanation_normalised"].append("could_not_process")
    
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
        test_sentences_batched = [input for input in self.batch(test_sentences)]
        sentiment_labels_batched = [input for input in self.batch(sentiment_labels)]
        rule_labels_batched = [input for input in self.batch(rule_labels)]
        contrasts_batched = [input for input in self.batch(contrasts)]
        
        for index, sentence in enumerate(tqdm(test_sentences_batched)):
            sentiment_labels = sentiment_labels_batched[index]
            rule_labels = rule_labels_batched[index]
            contrasts = contrasts_batched[index]

            test_sentences_vectorize, test_sentences_attention_masks = self.vectorize(sentence)
            probabilities = self.model.predict(x=[test_sentences_vectorize, test_sentences_attention_masks])

            try:
                exp = integrated_grad_cam.explain(test_sentences_vectorize, target=probabilities)
            except:
                for index_sent, sent in enumerate(sentence):
                    explanations["sentence"].append(sent)
                    explanations["features"].append(sent.split())
                    explanations["sentiment_probability_prediction"].append(probabilities[index_sent])
                    explanations["sentiment_label"].append(sentiment_labels[index_sent])
                    explanations["rule_label"].append(rule_labels[index_sent])
                    explanations["contrast"].append(contrasts[index_sent])
                    explanations["INT_GRAD_explanation"].append("could_not_process")
                    explanations["INT_GRAD_explanation_normalised"].append("could_not_process")
                    continue
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
    
    def create_explanations_nested_cv(self, datasets_nested_cv):

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
                self.model = eval(self.config["model_name"]+"(self.config)")
                # self.model.summary(line_length = 150) 
                self.model.load_weights("assets/trained_models/"+self.config["asset_name"]+"/"+self.config["asset_name"]+"_"+str(k_fold)+"_"+str(l_fold)+"_ckpt")

                test_dataset = datasets_nested_cv["val_dataset_"+str(k_fold)+"_"+str(l_fold)]["test_dataset_a_but_b_rule"]
                test_sentences = list(test_dataset['sentence'])
                sentiment_labels = list(test_dataset['sentiment_label'])
                rule_labels = list(test_dataset['rule_label'])
                contrasts = list(test_dataset['contrast'])

                test_sentences_vectorize, test_sentences_attention_masks = self.vectorize(test_sentences)
                probabilities = self.model.predict(x=[test_sentences_vectorize, test_sentences_attention_masks])

                integrated_grad_cam = IntegratedGradients(self.model, 
                                                            layer = self.model.layers[0].layers[0].embeddings,
                                                            method="gausslegendre", 
                                                            n_steps=100, 
                                                            internal_batch_size=32)
                                                            
                exp = integrated_grad_cam.explain(test_sentences_vectorize, target=probabilities)
                attributions = exp.attributions
                attributions = [att.sum(axis=2) for att in attributions]

                # Normalize attributions
                normalised_attributions = []
                for index, attribution in enumerate(attributions[0]):
                    probability = [1 - probabilities[index].tolist()[0], probabilities[index].tolist()[0]]
                    normalised_attribution = []
                    for att_value in attribution:
                        if att_value < 0:
                            weight_normalised_negative_class = abs(att_value)*probability[0]
                            normalised_attribution.append(weight_normalised_negative_class)
                        elif att_value > 0:
                            weight_normalised_positive_class = abs(att_value)*probability[1]
                            normalised_attribution.append(weight_normalised_positive_class)
                    normalised_attributions.append(normalised_attribution)

                # Append to explanations
                for index, sentence in enumerate(tqdm(test_sentences)):
                    explanations["sentence"].append(sentence)
                    explanations["features"].append(sentence.split())
                    explanations["sentiment_probability_prediction"].append(probabilities[index])
                    explanations["sentiment_label"].append(sentiment_labels[index])
                    explanations["rule_label"].append(rule_labels[index])
                    explanations["contrast"].append(contrasts[index])
                    explanations["INT_GRAD_explanation"].append(list(attributions[0][index]))
                    explanations["INT_GRAD_explanation_normalised"].append(normalised_attributions[index])

        # Save the explanations
        if not os.path.exists("assets/int_grad_explanations/"):
            os.makedirs("assets/int_grad_explanations/")
        with open("assets/int_grad_explanations/"+self.config["asset_name"]+".pickle", "wb") as handle:
            pickle.dump(explanations, handle)
