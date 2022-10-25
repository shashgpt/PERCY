from config import *
from scripts.models.models import *

class Shap_explanations(object):
    """
    Calculate the SHAP explanations for one-rule sentences in the test set
    """
    def __init__(self, config):
        self.config = config
        self.model = None
    
    def batch(self, iterable, batch_size=1):
        l = len(iterable)
        for ndx in range(0, l, batch_size):
            yield iterable[ndx:min(ndx + batch_size, l)]
    
    def vectorize(self, texts):
        """
        Tokenize, encode and PAD a batch of texts as ELMO vocabulary
        """
        tokenized_sentences = []
        max_len = 53
        for sentence in texts:
            tokenized_sentence = sentence.split()
            tokenized_sentences.append(tokenized_sentence)
        padded_tokenized_sentences = []
        for sentence in tokenized_sentences:
            padding_length = max_len - len(sentence)
            for pad in range(padding_length):
                sentence.append('@@PADDING@@')
            padded_tokenized_sentences.append(sentence)
        padded_tokenized_sentences = np.array(padded_tokenized_sentences)
        return padded_tokenized_sentences
    
    def elmo_encodings(self, padded_tokenized_sentences):
        instances = []
        indexer = ELMoTokenCharactersIndexer()
        for sentence in padded_tokenized_sentences:
            tokens = [Token(token) for token in sentence]
            field = TextField(tokens, {"character_ids": indexer})
            instance = Instance({"elmo": field})
            instances.append(instance)
        dataset = Batch(instances)
        vocab = Vocabulary()
        dataset.index_instances(vocab)
        encodings = dataset.as_tensor_dict()["elmo"]["character_ids"]["elmo_tokens"].numpy()
        for index, sentence in enumerate(encodings):
            pad_encoding = sentence[len(sentence)-1]
            pad_indices = []
            for token_index, token in enumerate(padded_tokenized_sentences[index]):
                if token == '@@PADDING@@':
                    pad_indices.append(token_index)
            for pad_index in pad_indices:
                sentence[pad_index] = np.array([0]*len(pad_encoding))
            encodings[index] = sentence
        return encodings

    def prediction(self, padded_tokenized_sentences):
        """
        Takes a raw text as input and returns model prediction for it 
        """
        probabilities = []
        self.model.eval()
        with torch.no_grad():
            for input in self.batch(padded_tokenized_sentences, self.config["mini_batch_size"]):
                character_ids = self.elmo_encodings(input)
                character_ids = tensor(character_ids).to(self.config["device"])
                softmax = nn.Softmax(dim=1)
                model_output = softmax(self.model(character_ids)).cpu().detach().numpy().tolist()
                for probability in model_output:
                    probabilities.append(probability)
        return np.array(probabilities)

    def create_explanations(self, train_dataset, test_dataset):

        explanations = {"sentence":[], 
                        "sentiment_probability_prediction":[],
                        "rule_label":[],
                        "contrast":[],
                        "base_value":[],
                        "SHAP_explanation":[], 
                        "SHAP_explanation_normalised":[]}
        
        # Load trained model
        self.model = eval(self.config["model_name"]+"(self.config)") # {ELMo} × {Static, Fine-tuning} × {no distillation, distillation}
        self.model = self.model.to(self.config["device"])
        self.model.load_state_dict(torch.load("assets/trained_models/"+self.config["asset_name"]+".pt", map_location=self.config["device"]))

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
            train_sentences_padded_tokenized = self.vectorize(train_sentence)
            test_sentences_padded_tokenized = self.vectorize(test_sentence)
            probabilities = self.prediction(test_sentences_padded_tokenized)
            exp_explainer = shap.KernelExplainer(self.prediction, shap.sample(train_sentences_padded_tokenized))
            shap_explanations = exp_explainer.shap_values(test_sentences_padded_tokenized, nsamples=64, l1_reg="aic")
    
    def create_shap_explanations_nested_cv(self, datasets_nested_cv):

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
                self.model = eval(self.config["model_name"]+"(self.config)") # {ELMo} × {Static, Fine-tuning} × {no distillation, distillation}
                self.model = self.model.to(self.config["device"])
                self.model.load_state_dict(torch.load("assets/trained_models/"+self.config["asset_name"]+".pt", map_location=self.config["device"]))

                train_dataset = datasets_nested_cv["train_dataset_"+str(k_fold)+"_"+str(l_fold)]
                test_dataset = datasets_nested_cv["val_dataset_"+str(k_fold)+"_"+str(l_fold)]
                train_sentences = list(train_dataset.loc[(train_dataset["rule_label"]!=0)&(train_dataset["contrast"]==1)]['sentence'])
                test_sentences = list(test_dataset.loc[(test_dataset["rule_label"]!=0)&(test_dataset["contrast"]==1)]['sentence'])
                test_rule_labels = list(test_dataset.loc[(test_dataset["rule_label"]!=0)&(test_dataset["contrast"]==1)]['rule_label'])
                test_contrasts = list(test_dataset.loc[(test_dataset["rule_label"]!=0)&(test_dataset["contrast"]==1)]['contrast'])

                train_sentences_padded_tokenized = self.vectorize(train_sentences)
                test_sentences_padded_tokenized = self.vectorize(test_sentences)
    
                probabilities = self.prediction(test_sentences_padded_tokenized)

                exp_explainer = shap.Explainer(model=self.prediction, masker=train_sentences_padded_tokenized[:len(train_sentences_padded_tokenized)*10], algorithm="permutation")
                shap_explanations = exp_explainer(test_sentences_padded_tokenized)

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