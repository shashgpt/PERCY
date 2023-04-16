from config import *
from scripts.models.models import *

class Evaluation(object):
    def __init__(self, config):
        self.config = config
    
    def vectorize(self, sentences):
        """
        tokenize each preprocessed sentence in dataset using bert tokenizer
        """
        max_len = 0
        input_ids = []
        attention_masks = []
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
    
    def evaluate_model(self, test_dataset):
        results = {'sentence':[], 
                    'sentiment_label':[],
                    'rule_label':[],
                    'contrast':[],  
                    'sentiment_probability_output':[], 
                    'sentiment_prediction_output':[]}
        
        # Load trained model
        model = eval(self.config["model_name"]+"(self.config)")
        model.load_weights("assets/trained_models/"+self.config["asset_name"]+"_ckpt")

        test_dataset = test_dataset["test_dataset"]
        test_sentences = test_dataset["sentence"]
        test_sentiment_labels = test_dataset["sentiment_label"]

        test_sentences, test_attention_masks = self.vectorize(test_sentences)
        test_sentiment_labels = np.array(test_sentiment_labels)
        dataset = ([test_sentences, test_attention_masks], test_sentiment_labels)
        predictions = model.predict(x=dataset[0])

        for index, sentence in enumerate(test_dataset["sentence"]):
            results['sentence'].append(list(test_dataset['sentence'])[index])
            results['sentiment_label'].append(list(test_dataset['sentiment_label'])[index])
            results['rule_label'].append(list(test_dataset['rule_label'])[index])
            results['contrast'].append(list(test_dataset['contrast'])[index])
        for prediction in predictions:
            results['sentiment_probability_output'].append(prediction)
            prediction = np.rint(prediction)
            results['sentiment_prediction_output'].append(prediction[0])

        if not os.path.exists("assets/results/"):
            os.makedirs("assets/results/")
        with open("assets/results/"+self.config["asset_name"]+".pickle", 'wb') as handle:
            pickle.dump(results, handle)
    
    def evaluate_model_nested_cv(self, datasets_nested_cv):
        results = {'sentence':[], 
                    'sentiment_label':[],
                    'rule_label':[],
                    'contrast':[],  
                    'sentiment_probability_output':[], 
                    'sentiment_prediction_output':[]}
        for k_fold in range(1, self.config["k_samples"]+1):
            for l_fold in range(1, self.config["l_samples"]+1):
                
                # Load trained model
                try:
                    model = eval(self.config["model_name"]+"(self.config)")
                    model.load_weights("assets/trained_models/"+self.config["asset_name"]+"/"+self.config["asset_name"]+"_"+str(k_fold)+"_"+str(l_fold)+"_ckpt")
                except:
                    model = cnn(config)
                    model.load_weights("assets/trained_models/"+self.config["asset_name"]+"/"+self.config["asset_name"]+"_"+str(k_fold)+"_"+str(l_fold)+".h5")

                test_dataset = datasets_nested_cv["val_dataset_"+str(k_fold)+"_"+str(l_fold)]["test_dataset"]
                test_sentences = test_dataset["sentence"]
                test_sentiment_labels = test_dataset["sentiment_label"]
                test_sentences, test_attention_masks = self.vectorize(test_sentences)
                test_sentiment_labels = np.array(test_sentiment_labels).astype(np.float32)
                dataset = ([test_sentences, test_attention_masks], test_sentiment_labels)
                predictions = model.predict(x=dataset[0])
                
                for index, sentence in enumerate(test_dataset["sentence"]):
                    results['sentence'].append(list(test_dataset['sentence'])[index])
                    results['sentiment_label'].append(list(test_dataset['sentiment_label'])[index])
                    results['rule_label'].append(list(test_dataset['rule_label'])[index])
                    results['contrast'].append(list(test_dataset['contrast'])[index])
                for prediction in predictions:
                    results['sentiment_probability_output'].append(prediction)
                    prediction = np.rint(prediction)
                    results['sentiment_prediction_output'].append(prediction[0])

        if not os.path.exists("assets/results/"):
            os.makedirs("assets/results/")
        with open("assets/results/"+self.config["asset_name"]+".pickle", 'wb') as handle:
            pickle.dump(results, handle)