from config import *
from scripts.models.models import *

class Evaluation(object):
    def __init__(self, config):
        self.config = config
    
    def vectorize(self, sentences, word_index):
        """
        tokenize each preprocessed sentence in dataset as sentence.split()
        encode each tokenized sentence as per vocabulary
        right pad each encoded tokenized sentence with 0 upto max_tokenized_sentence_len the dataset 
        """
        vocab = [key for key in word_index.keys()]
        vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(standardize=None, split='whitespace', vocabulary=vocab)
        return vectorize_layer(np.array(sentences)).numpy()
    
    def evaluate_model(self, test_dataset, word_vectors, word_index):
        results = {'sentence':[], 
                    'sentiment_label':[],
                    'rule_label':[],
                    'contrast':[],  
                    'sentiment_probability_output':[], 
                    'sentiment_prediction_output':[]}
        
        # Load trained model
        model = eval(self.config["model_name"]+"(self.config, word_vectors)") # {Word2vec, Glove} × {Static, Fine-tuning} × {no distillation, distillation}
        model.load_weights("assets/trained_models/"+self.config["asset_name"]+".h5")
        
        test_dataset = test_dataset["test_dataset"]
        test_sentences = test_dataset["sentence"]
        test_sentiment_labels = test_dataset["sentiment_label"]

        test_sentences_vectorized = self.vectorize(test_sentences, word_index)
        test_sentiment_labels = np.array(test_sentiment_labels)
        dataset = (test_sentences_vectorized, test_sentiment_labels)

        predictions = model.predict(x=dataset[0])

        for index, sentence in enumerate(tqdm(test_sentences)):
            results['sentence'].append(sentence)
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
        
    def evaluate_model_nested_cv(self, datasets_nested_cv, word_vectors, word_index):
        results = {'sentence':[], 
                    'sentiment_label':[],
                    'rule_label':[],
                    'contrast':[],  
                    'sentiment_probability_output':[], 
                    'sentiment_prediction_output':[]}
        for k_fold in range(1, self.config["k_samples"]+1):
            for l_fold in range(1, self.config["l_samples"]+1):

                # Load trained model
                model = eval(self.config["model_name"]+"(self.config, word_vectors)") # {Word2vec, Glove} × {Static, Fine-tuning} × {no distillation, distillation}
                model.load_weights("assets/trained_models/"+self.config["asset_name"]+"/"+self.config["asset_name"]+"_"+str(k_fold)+"_"+str(l_fold)+".h5")

                test_dataset = datasets_nested_cv["val_dataset_"+str(k_fold)+"_"+str(l_fold)]["test_dataset"]
                test_sentences = test_dataset["sentence"]
                test_sentiment_labels = test_dataset["sentiment_label"]

                test_sentences_vectorized = self.vectorize(test_sentences, word_index)
                test_sentiment_labels = np.array(test_sentiment_labels)
                dataset = (test_sentences_vectorized, test_sentiment_labels)

                predictions = model.predict(x=dataset[0])

                for index, sentence in enumerate(tqdm(test_sentences)):
                    results['sentence'].append(sentence)
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