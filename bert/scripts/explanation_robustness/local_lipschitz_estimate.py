from config import *

class Local_lipschitz_estimate(object):
    def __init__(self, config):
        """
        Calculate the lipschitz value for each explanation in LIME and SHAP
        """
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

    def topk_argmax(self, A, k):
        # Use argpartition to avoid sorting whole matrix
        argmaxs = np.argpartition(A,-k)[:,-k:]
        vals = np.array([A[k][idxs] for k,idxs in enumerate(argmaxs)])
        # We now have topk, but they're not sorted. Now sort these (fast since only k)
        argmaxs = np.array([argmaxs[i,np.argsort(-v)] for i,v in enumerate(vals)])
        vals = np.array([A[k][idxs] for k,idxs in enumerate(argmaxs)])
        return vals, argmaxs

    def estimate_discrete_dataset_lipschitz(self, dataset, explanations, probabilities, eps = None, top_k = 1, metric = 'euclidean', same_class = False):
        """
            For every point in dataset, find pair point y in dataset that maximizes
            Lipschitz: || f(x) - f(y) ||/||x - y||

            Args:
                - dataset: a tds obkect
                - top_k : how many to return
                - max_distance: maximum distance between points to consider (radius)
                - same_class: ignore cases where argmax g(x) != g(y), where g is the prediction model
        """
        Xs  = dataset
        n, d = Xs.shape
        Fs = explanations
        Preds_prob = probabilities
        # Preds_class = Preds_prob.argmax(axis=1)
        Preds_class = [np.rint(Pred_prob) for Pred_prob in Preds_prob]
        num_dists = pairwise_distances(Fs)#, metric = 'euclidean')
        den_dists = pairwise_distances(Xs, metric = metric) # Or chebyshev?
        if eps is not None:
            nonzero = np.sum((den_dists > eps))
            total   = den_dists.size
            print('Number of zero denom distances: {} ({:4.2f}%)'.format(
                total - nonzero, 100*(total-nonzero)/total))
            den_dists[den_dists > eps] = -1.0 #float('inf')
        den_dists[den_dists==0] = -1 #float('inf')
        if same_class:
            for i in range(n):
                for j in range(n):
                    if Preds_class[i] != Preds_class[j]:
                        den_dists[i,j] = -1
        ratios = (num_dists/den_dists)
        argmaxes = {k: [] for k in range(n)}
        vals, inds = self.topk_argmax(ratios, top_k)
        argmaxes = {i:  [(j,v) for (j,v) in zip(inds[i,:],vals[i,:])] for i in range(n)}
        return vals.squeeze(), argmaxes

    def lime_explanations_estimates(self):

        # Load results
        with open("assets/results/"+self.config["asset_name"]+".pickle", 'rb') as handle:
            results = pickle.load(handle)
            results = pd.DataFrame(results)
            results = results.loc[results["rule_label"]==1].reset_index(drop=True)

        # Load lime explanations
        with open("assets/lime_explanations/"+self.config["asset_name"]+".pickle", "rb") as handle:
            explanations = pickle.load(handle)
            explanations = pd.DataFrame(explanations)
        
        # Filter out sentences for which LIME explanations couldn't be calculated
        indices = []
        couldn_process_indices = []
        for index, explanation in enumerate(explanations["LIME_explanation"]):
            if explanation != "couldn't process":
                indices.append(index)
            elif explanation == "couldn't process":
                couldn_process_indices.append(index)

        # Dataset
        test_sentences = list(results.iloc[indices]["sentence"])
        probabilities = np.array(list(results.iloc[indices]["sentiment_probability_output"]))
        test_sentences_vectorized, _ = self.vectorize(test_sentences)
        rule_labels = list(results.iloc[indices]["rule_label"])
        
        # LIME explanations
        lime_explanations = list(explanations.iloc[indices]["LIME_explanation"])
        lime_explanations_padded = tf.keras.preprocessing.sequence.pad_sequences(lime_explanations, dtype='float32', padding='post', value=0.0)

        # Lipschitz estimates
        lime_vals, lime_argmaxes = self.estimate_discrete_dataset_lipschitz(test_sentences_vectorized, lime_explanations_padded, probabilities, top_k = 3, metric = "chebyshev", same_class = True)
        max_lip = lime_vals.max()
        imax, _ = np.unravel_index(np.argmax(lime_vals), lime_vals.shape)
        jmax = lime_argmaxes[imax][0][0]
        print('Max Lip value: {}, attained for pair ({}, {})'.format(max_lip, imax, jmax))

        # Append the values
        lime_lip_vals = [value[0] for value in lime_vals]
        for index in couldn_process_indices:
            lime_lip_vals.insert(index, "couldn't process")
        explanations["LIME_lipschtiz_value"] = lime_lip_vals
        
        # Save the values
        if not os.path.exists("assets/lime_explanations/"):
            os.makedirs("assets/lime_explanations/")
        with open("assets/lime_explanations/"+self.config["asset_name"]+".pickle", "wb") as handle:
            explanations.to_dict()
            pickle.dump(explanations, handle)
        
    def shap_explanations_estimates(self):

        # Load results
        with open("assets/results/"+self.config["asset_name"]+".pickle", 'rb') as handle:
            results = pickle.load(handle)
            results = pd.DataFrame(results)
            results = results.loc[results["rule_label"]==1].reset_index(drop=True)
        
        # Load shap explanations
        with open("assets/shap_explanations/"+self.config["asset_name"]+".pickle", "rb") as handle:
            explanations = pickle.load(handle)
            explanations = pd.DataFrame(explanations)
        
        # Dataset
        test_sentences = list(results["sentence"])
        probabilities = np.array(list(results["sentiment_probability_output"]))
        test_sentences_vectorized = self.vectorize(test_sentences)
        rule_labels = list(results["rule_label"])
    
        # SHAP explanations
        shap_explanations = list(explanations["SHAP_explanation"])
        shap_explanations_tokens = []
        for index, sentence in enumerate(test_sentences):
            tokenized_sentence = sentence.split()
            if rule_labels[index] == 1:
                shap_explanations_tokens.append(shap_explanations[index][:len(tokenized_sentence)])
        shap_explanations_padded = tf.keras.preprocessing.sequence.pad_sequences(shap_explanations_tokens, dtype='float32', padding='post', value=0.0)        
            
        shap_vals, shap_argmaxes = self.estimate_discrete_dataset_lipschitz(test_sentences_vectorized, shap_explanations_padded, probabilities, top_k = 3, metric = "chebyshev", same_class = True)
        max_lip = shap_vals.max()
        imax, _ = np.unravel_index(np.argmax(shap_vals), shap_vals.shape)
        jmax = shap_argmaxes[imax][0][0]
        print('Max Lip value: {}, attained for pair ({}, {})'.format(max_lip, imax, jmax))

        # Append the values
        shap_lip_vals = [value[0] for value in shap_vals]
        explanations["SHAP_lipschtiz_value"] = shap_lip_vals
        
        # Save the values
        if not os.path.exists("assets/shap_explanations/"):
            os.makedirs("assets/shap_explanations/")
        with open("assets/shap_explanations/"+self.config["asset_name"]+".pickle", "wb") as handle:
            explanations.to_dict()
            pickle.dump(explanations, handle)
    
    def int_grad_explanations_estimates(self):

        # Load results
        with open("assets/results/"+self.config["asset_name"]+".pickle", 'rb') as handle:
            results = pickle.load(handle)
            results = pd.DataFrame(results)
            results = results.loc[results["rule_label"]==1].reset_index(drop=True)
        
        # Load shap explanations
        with open("assets/int_grad_explanations/"+self.config["asset_name"]+".pickle", "rb") as handle:
            explanations = pickle.load(handle)
            explanations = pd.DataFrame(explanations)

        # Dataset
        test_sentences = list(results["sentence"])
        probabilities = np.array(list(results["sentiment_probability_output"]))
        test_sentences_vectorized, _ = self.vectorize(test_sentences)
        rule_labels = list(results["rule_label"])
    
        # INT-GRAD explanations
        int_grad_explanations = list(explanations["INT_GRAD_explanation"])
        explanations_tokens = []
        for index, sentence in enumerate(test_sentences):
            tokenized_sentence = sentence.split()
            if rule_labels[index] == 1:
                explanations_tokens.append(int_grad_explanations[index][:len(tokenized_sentence)])
        int_grad_explanations_padded = tf.keras.preprocessing.sequence.pad_sequences(explanations_tokens, dtype='float32', padding='post', value=0.0)
            
        vals, argmaxes = self.estimate_discrete_dataset_lipschitz(test_sentences_vectorized, int_grad_explanations_padded, probabilities, top_k = 3, metric = "chebyshev", same_class = True)
        max_lip = vals.max()
        imax, _ = np.unravel_index(np.argmax(vals), vals.shape)
        jmax = argmaxes[imax][0][0]
        print('Max Lip value: {}, attained for pair ({}, {})'.format(max_lip, imax, jmax))

        # Append the values
        lip_vals = [value[0] for value in vals]
        explanations["Int_grad_lipschtiz_value"] = lip_vals
        
        # Save the values
        if not os.path.exists("assets/int_grad_explanations/"):
            os.makedirs("assets/int_grad_explanations/")
        with open("assets/int_grad_explanations/"+self.config["asset_name"]+".pickle", "wb") as handle:
            explanations.to_dict()
            pickle.dump(explanations, handle)