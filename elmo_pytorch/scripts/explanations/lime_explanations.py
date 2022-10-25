from config import *
from scripts.models.models import *

class Lime_explanations(object):
    """
    Calculate the LIME explanations for one-rule sentences in the test set
    """
    def __init__(self, config):
        self.config = config
        self.model = None
    
    def batch(self, iterable, batch_size=1):
        l = len(iterable)
        for ndx in range(0, l, batch_size):
            yield iterable[ndx:min(ndx + batch_size, l)]

    def vectorize(self, sentences):
        tokenized_texts = []
        for text in sentences:
            tokenized_text = text.split()
            tokenized_texts.append(tokenized_text)
        character_ids = tensor(np.array(batch_to_ids(tokenized_texts))).type(torch.long).to(self.config["device"])
        return character_ids
    
    def prediction(self, texts):
        """
        Takes a raw text as input and returns model prediction for it 
        """
        probabilities = []
        self.model.eval()
        with torch.no_grad():
            for input in self.batch(texts, self.config["mini_batch_size"]):
                character_ids = self.vectorize(input)
                softmax = nn.Softmax(dim=1)
                model_output = softmax(self.model(character_ids)).cpu().detach().numpy().tolist()
                for probability in model_output:
                    probabilities.append(probability)
        return np.array(probabilities)
        
    def create_lime_explanations(self, test_dataset):

        explanations = {"sentence":[],
                        "features":[], 
                        "sentiment_probability_prediction":[],
                        "sentiment_label":[],
                        "rule_label":[],
                        "contrast":[],
                        "LIME_explanation":[], 
                        "LIME_explanation_normalised":[]}
        
        # Load trained model
        self.model = eval(self.config["model_name"]+"(self.config)") # {ELMo} × {Static, Fine-tuning} × {no distillation, distillation}
        self.model = self.model.to(self.config["device"])
        self.model.load_state_dict(torch.load("assets/trained_models/"+self.config["asset_name"]+".pt", map_location=self.config["device"]))

        test_dataset = test_dataset["test_dataset_a_but_b_rule"]
        test_sentences = list(test_dataset['sentence'])
        sentiment_labels = list(test_dataset['sentiment_label'])
        rule_labels = list(test_dataset['rule_label'])
        contrasts = list(test_dataset['contrast'])

        probabilities = self.prediction(test_sentences)

        explainer = lime_text.LimeTextExplainer(class_names=["negative_sentiment", "positive_sentiment"], split_expression=" ", random_state=self.config["seed_value"])

        for index, test_datapoint in enumerate(tqdm(test_sentences)):
            probability = probabilities[index]
            sentiment_label = sentiment_labels[index]
            rule_label = rule_labels[index]
            contrast = contrasts[index]
            tokenized_sentence = test_datapoint.split()
            try:
                exp = explainer.explain_instance(test_datapoint, self.prediction, num_features = len(tokenized_sentence), num_samples=self.config["lime_no_of_samples"])
            except:
                explanations['sentence'].append(test_datapoint)
                explanations['features'].append(tokenized_sentence)
                explanations['sentiment_probability_prediction'].append(probability)
                explanations['sentiment_label'].append(sentiment_label)
                explanations['rule_label'].append(rule_label)
                explanations['contrast'].append(contrast)
                explanations['LIME_explanation'].append("couldn't process")
                explanations['LIME_explanation_normalised'].append("couldn't process")
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

    def create_lime_explanations_nested_cv(self, datasets_nested_cv):

        explanations = {"sentence":[],
                        "features":[], 
                        "sentiment_probability_prediction":[],
                        "sentiment_label":[],
                        "rule_label":[],
                        "contrast":[],
                        "LIME_explanation":[], 
                        "LIME_explanation_normalised":[]}

        for k_fold in range(1, self.config["k_samples"]+1):
            for l_fold in range(1, self.config["l_samples"]+1):

                self.model = eval(self.config["model_name"]+"(self.config)") # {ELMo} × {Static, Fine-tuning} × {no distillation, distillation}
                self.model = self.model.to(self.config["device"])
                self.model.load_state_dict(torch.load("assets/trained_models/"+self.config["asset_name"]+"/"+self.config["asset_name"]+"_"+str(k_fold)+"_"+str(l_fold)+".pt", map_location=self.config["device"]))

                test_dataset = datasets_nested_cv["val_dataset_"+str(k_fold)+"_"+str(l_fold)]["test_dataset_a_but_b_rule"]
                test_sentences = list(test_dataset.loc[(test_dataset["rule_label"]!=0)&(test_dataset["contrast"]==1)]['sentence'])
                sentiment_labels = list(test_dataset.loc[(test_dataset["rule_label"]!=0)&(test_dataset["contrast"]==1)]['sentiment_label'])
                rule_labels = list(test_dataset.loc[(test_dataset["rule_label"]!=0)&(test_dataset["contrast"]==1)]['rule_label'])
                contrasts = list(test_dataset.loc[(test_dataset["rule_label"]!=0)&(test_dataset["contrast"]==1)]['contrast'])

                probabilities = self.prediction(test_sentences)

                explainer = lime_text.LimeTextExplainer(class_names=["negative_sentiment", "positive_sentiment"], split_expression=" ", random_state=self.config["seed_value"])

                for index, test_datapoint in enumerate(tqdm(test_sentences)):
                    probability = probabilities[index]
                    sentiment_label = sentiment_labels[index]
                    rule_label = rule_labels[index]
                    contrast = contrasts[index]
                    tokenized_sentence = test_datapoint.split()
                    try:
                        exp = explainer.explain_instance(test_datapoint, self.prediction, num_features = len(tokenized_sentence), num_samples=self.config["lime_no_of_samples"])
                    except:
                        explanations['sentence'].append(test_datapoint)
                        explanations['features'].append(tokenized_sentence)
                        explanations['sentiment_probability_prediction'].append(probability)
                        explanations['sentiment_label'].append(sentiment_label)
                        explanations['rule_label'].append(rule_label)
                        explanations['contrast'].append(contrast)
                        explanations['LIME_explanation'].append("couldn't process")
                        explanations['LIME_explanation_normalised'].append("couldn't process")
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