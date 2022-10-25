from config import *
from scripts.training.utils.datasetSST2 import *
from scripts.models.models import *

class Evaluation(object):
    def __init__(self, config):
        self.config = config
    
    def fit_eval(self, test_dataset, model):
        probabilities = []
        test_loader = DataLoader(test_dataset, batch_size = self.config["mini_batch_size"], shuffle=False)
        model.eval()
        with torch.no_grad():
            for batch_idx, input in enumerate(tqdm(test_loader)):
                input_data = input[0]
                input_data = input_data.to(self.config["device"])
                softmax = nn.Softmax(dim=1)
                model_output = softmax(model(input_data)).cpu().detach().numpy().tolist()
                for probability in model_output:
                    probabilities.append(probability)
        return probabilities
    
    def vectorize(self, sentences, sentiment_labels):
        tokenized_texts = []
        for text in sentences:
            tokenized_text = text.split()
            tokenized_texts.append(tokenized_text)
        character_ids = tensor(np.array(batch_to_ids(tokenized_texts))).type(torch.long)
        sentiment_labels = tensor(np.array(sentiment_labels)).type(torch.long)
        return character_ids, sentiment_labels

    def evaluate_model(self, test_dataset):
        results = {'sentence':[], 
                    'sentiment_label':[],
                    'rule_label':[],
                    'contrast':[],  
                    'sentiment_probability_output':[], 
                    'sentiment_prediction_output':[]}
        
        # Load trained model
        model = eval(self.config["model_name"]+"(self.config)") # {ELMo} × {Static, Fine-tuning} × {no distillation, distillation}
        model = model.to(self.config["device"])
        model.load_state_dict(torch.load("assets/trained_models/"+self.config["asset_name"]+".pt", map_location=self.config["device"]))

        # Get test dataset for this fold
        test_dataset = test_dataset["test_dataset"]
        test_sentences = test_dataset["sentence"]
        test_sentiment_labels = test_dataset["sentiment_label"]

        # Vectorize
        test_sentences, test_sentiment_labels = self.vectorize(test_sentences, test_sentiment_labels)

        # Calcualte predictions
        test_dataset_obj = DatasetSST2(test_sentences, test_sentiment_labels, transform = None)
        probabilities = self.fit_eval(test_dataset_obj, model)

        for index, sentence in enumerate(test_dataset["sentence"]):
            results['sentence'].append(list(test_dataset['sentence'])[index])
            results['sentiment_label'].append(list(test_dataset['sentiment_label'])[index])
            results['rule_label'].append(list(test_dataset['rule_label'])[index])
            results['contrast'].append(list(test_dataset['contrast'])[index])
        for probability in probabilities:
            results['sentiment_probability_output'].append(probability)
            prediction = probability.index(max(probability))
            results['sentiment_prediction_output'].append(prediction)

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
                model = eval(self.config["model_name"]+"(self.config)") # {ELMo} × {Static, Fine-tuning} × {no distillation, distillation}
                model = model.to(self.config["device"])
                model.load_state_dict(torch.load("assets/trained_models/"+self.config["asset_name"]+"/"+self.config["asset_name"]+"_"+str(k_fold)+"_"+str(l_fold)+".pt", map_location=self.config["device"]))

                # Get test dataset for this fold
                test_dataset_df = datasets_nested_cv["val_dataset_"+str(k_fold)+"_"+str(l_fold)]["test_dataset"]
                test_sentences = test_dataset_df["sentence"]
                test_sentiment_labels = test_dataset_df["sentiment_label"]

                # Vectorize
                test_sentences, test_sentiment_labels = self.vectorize(test_sentences, test_sentiment_labels)

                # Create dataset
                test_dataset = DatasetSST2(test_sentences, test_sentiment_labels, transform = None)

                # Calcualte predictions
                probabilities = self.fit_eval(test_dataset, model)

                for index, sentence in enumerate(test_dataset_df["sentence"]):
                    results['sentence'].append(list(test_dataset_df['sentence'])[index])
                    results['sentiment_label'].append(list(test_dataset_df['sentiment_label'])[index])
                    results['rule_label'].append(list(test_dataset_df['rule_label'])[index])
                    results['contrast'].append(list(test_dataset_df['contrast'])[index])
                for probability in probabilities:
                    results['sentiment_probability_output'].append(probability)
                    prediction = probability.index(max(probability))
                    results['sentiment_prediction_output'].append(prediction)

        if not os.path.exists("assets/results/"):
            os.makedirs("assets/results/")
        with open("assets/results/"+self.config["asset_name"]+".pickle", 'wb') as handle:
            pickle.dump(results, handle)