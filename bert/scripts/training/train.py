from config import *

from scripts.models.models import *
from scripts.training.additional_validation_sets import AdditionalValidationSets

class Train(object):
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

    # def vectorize(self, sentences):
    #     """
    #     tokenize each preprocessed sentence in dataset using bert tokenizer
    #     """
    #     max_len = 0
    #     input_ids = []
    #     for sentence in sentences:
    #         tokenized_context = self.config["bert_tokenizer"].encode(sentence)
    #         input_id = tokenized_context.ids
    #         input_ids.append(input_id)
    #         if len(input_id) > max_len:
    #             max_len = len(input_id)
    #     for index, input_id in enumerate(input_ids):
    #         padding_length = max_len - len(input_ids[index])
    #         input_ids[index] = input_ids[index] + ([0] * padding_length)
    #     return np.array(input_ids)
    
    def rule_conjunct_extraction(self, dataset, rule):
        """
        Extracts the rule_conjuncts from sentences containing the logic rule corresponding to rule_keyword
        """
        rule_conjuncts = []
        rule_label_ind = []
        for index, sentence in enumerate(list(dataset['sentence'])):
            tokenized_sentence = sentence.split()
            rule_label = dataset['rule_label'][index]
            contrast = dataset['contrast'][index]
            if rule_label == rule and contrast==1:
                if rule_label == 1:
                    b_part_tokenized_sentence = tokenized_sentence[tokenized_sentence.index("but")+1:]
                    b_part_sentence = ' '.join(b_part_tokenized_sentence)
                    rule_conjuncts.append(b_part_sentence)
                    rule_label_ind.append(1)
                elif rule_label == 2:
                    b_part_tokenized_sentence = tokenized_sentence[tokenized_sentence.index("yet")+1:]
                    b_part_sentence = ' '.join(b_part_tokenized_sentence)
                    rule_conjuncts.append(b_part_sentence)
                    rule_label_ind.append(1)
                elif rule_label == 3:
                    a_part_tokenized_sentence = tokenized_sentence[:tokenized_sentence.index("though")]
                    a_part_sentence = ' '.join(a_part_tokenized_sentence)
                    rule_conjuncts.append(a_part_sentence)
                    rule_label_ind.append(1)
                elif rule_label == 4:
                    a_part_tokenized_sentence = tokenized_sentence[:tokenized_sentence.index("while")]
                    a_part_sentence = ' '.join(a_part_tokenized_sentence)
                    rule_conjuncts.append(a_part_sentence)
                    rule_label_ind.append(1)
            else:
                rule_conjuncts.append('')
                rule_label_ind.append(0)
        return rule_conjuncts, rule_label_ind
    
    def remove_extra_samples(self, sample):
        sample = sample[:(sample.shape[0]-sample.shape[0]%self.config["mini_batch_size"])]
        return sample
    
    def additional_validation_datasets(self, dataset):

        additional_validation_datasets = []
        for key, value in dataset.items():
            sentences, attention_masks = self.vectorize(dataset[key]["sentence"])
            sentiment_labels = np.array(dataset[key]["sentiment_label"]).astype(np.float32)

            sentences_but_features, sentences_but_features_ind = self.rule_conjunct_extraction(dataset[key], rule=1)
            sentences_but_features, sentences_but_features_attention_masks  = self.vectorize(sentences_but_features)
            sentences_but_features_ind = np.array(sentences_but_features_ind).astype(np.float32)
            sentences_but_features_ind = sentences_but_features_ind.reshape(sentences_but_features_ind.shape[0], 1)

            sentences = self.remove_extra_samples(sentences)
            attention_masks = self.remove_extra_samples(attention_masks)
            sentiment_labels = self.remove_extra_samples(sentiment_labels)
            sentences_but_features = self.remove_extra_samples(sentences_but_features)
            sentences_but_features_attention_masks = self.remove_extra_samples(sentences_but_features_attention_masks)
            sentences_but_features_ind = self.remove_extra_samples(sentences_but_features_ind)

            if self.config["distillation"] == True:
                key_dataset = ([[sentences, attention_masks], [sentences_but_features, sentences_but_features_attention_masks]], 
                                [sentiment_labels, [sentences_but_features_ind]], key)
                additional_validation_datasets.append(key_dataset)
            else:
                key_dataset = ([sentences, attention_masks], sentiment_labels, key)
                additional_validation_datasets.append(key_dataset)
        return additional_validation_datasets
    
    def train_model(self, train_dataset, val_dataset, test_dataset):

        # Create model
        # model = eval(self.config["model_name"]+"(self.config)") # {BERT} × {Static, Fine-tuning} × {no distillation, distillation}
        if self.config["model_name"] == "lstm":
            model = lstm(self.config)
        elif self.config["model_name"] == "cnn":
            model = cnn(self.config)
        model.summary(line_length = 150)
        if not os.path.exists("assets/computation_graphs"):
            os.makedirs("assets/computation_graphs")
        # plot_model(model, show_shapes = True, to_file = "assets/computation_graphs/"+self.config["asset_name"]+".png")
        
        # Create train dataset
        train_sentences = train_dataset["sentence"]
        train_sentiment_labels = train_dataset["sentiment_label"]
        train_sentences, train_attention_masks = self.vectorize(train_sentences)
        train_sentiment_labels = np.array(train_sentiment_labels).astype(np.float32)

        # Create validation dataset
        val_sentences = val_dataset["sentence"]
        val_sentiment_labels = val_dataset["sentiment_label"]
        val_sentences, val_attention_masks = self.vectorize(val_sentences)
        val_sentiment_labels = np.array(val_sentiment_labels).astype(np.float32)

        # Create train rule features
        train_sentences_but_features, train_sentences_but_features_ind = self.rule_conjunct_extraction(train_dataset, rule=1)
        train_sentences_but_features, train_sentences_but_features_attention_masks = self.vectorize(train_sentences_but_features)
        train_sentences_but_features_ind = np.array(train_sentences_but_features_ind).astype(np.float32)
        train_sentences_but_features_ind = train_sentences_but_features_ind.reshape(train_sentences_but_features_ind.shape[0], 1)

        # Create val rule features
        val_sentences_but_features, val_sentences_but_features_ind = self.rule_conjunct_extraction(val_dataset, rule=1)
        val_sentences_but_features, val_sentences_but_features_attention_masks = self.vectorize(val_sentences_but_features)
        val_sentences_but_features_ind = np.array(val_sentences_but_features_ind).astype(np.float32)
        val_sentences_but_features_ind = val_sentences_but_features_ind.reshape(val_sentences_but_features_ind.shape[0], 1)

        # Remove extra samples
        train_sentences = self.remove_extra_samples(train_sentences)
        train_attention_masks = self.remove_extra_samples(train_attention_masks)
        train_sentiment_labels = self.remove_extra_samples(train_sentiment_labels)
        train_sentences_but_features = self.remove_extra_samples(train_sentences_but_features)
        train_sentences_but_features_attention_masks = self.remove_extra_samples(train_sentences_but_features_attention_masks)
        train_sentences_but_features_ind = self.remove_extra_samples(train_sentences_but_features_ind)
        val_sentences = self.remove_extra_samples(val_sentences)
        val_attention_masks = self.remove_extra_samples(val_attention_masks)
        val_sentiment_labels = self.remove_extra_samples(val_sentiment_labels)
        val_sentences_but_features = self.remove_extra_samples(val_sentences_but_features)
        val_sentences_but_features_attention_masks = self.remove_extra_samples(val_sentences_but_features_attention_masks)
        val_sentences_but_features_ind = self.remove_extra_samples(val_sentences_but_features_ind)

        additional_validation_datasets = self.additional_validation_datasets(test_dataset)

        if self.config["distillation"] == True:

            # Create train and val datasets
            train_dataset = ([[train_sentences, train_attention_masks], [train_sentences_but_features, train_sentences_but_features_attention_masks]], 
                            [train_sentiment_labels, [train_sentences_but_features_ind]])
            val_dataset = ([[val_sentences, val_attention_masks], [val_sentences_but_features, val_sentences_but_features_attention_masks]], 
                            [val_sentiment_labels, [val_sentences_but_features_ind]])

            # Train
            my_callbacks = []
            if "early_stopping" in self.config["callbacks"]:
                metric = self.config["metric"]
                patience = self.config["patience"]
                early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor=metric,              # 1. Calculate val_loss_1 
                                                                            min_delta = 0,                  # 2. Check val_losses for next 10 epochs 
                                                                            patience=patience,                    # 3. Stop training if none of the val_losses are lower than val_loss_1
                                                                            verbose=0,                      # 4. Get the trained weights corresponding to val_loss_1
                                                                            mode="auto",
                                                                            baseline=None, 
                                                                            restore_best_weights=True)
                my_callbacks.append(early_stopping_callback)
            if "additional_val_datasets" in self.config["callbacks"]: 
                my_callbacks.append(AdditionalValidationSets(additional_validation_datasets, self.config))

            model.fit(x=train_dataset[0], 
                    y=train_dataset[1], 
                    epochs=self.config["train_epochs"], 
                    batch_size=self.config["mini_batch_size"], 
                    validation_data=val_dataset, 
                    callbacks=my_callbacks,
                    shuffle=False)

            # Save trained weights of the model
            if not os.path.exists("assets/trained_models/"+self.config["asset_name"]):
                os.makedirs("assets/trained_models/"+self.config["asset_name"])
            model.save_weights("assets/trained_models/"+self.config["asset_name"]+"_ckpt")

        else:
            
            # Create train and val datasets
            train_dataset = ([train_sentences, train_attention_masks], train_sentiment_labels)
            val_dataset = ([val_sentences, val_attention_masks], val_sentiment_labels)

            # Train
            my_callbacks = []
            if "early_stopping" in self.config["callbacks"]:
                metric = self.config["metric"]
                patience = self.config["patience"]
                early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor = metric,              # 1. Calculate val_loss_1 
                                                                            min_delta = 0,                # 2. Check val_losses for next 10 epochs 
                                                                            patience = patience,          # 3. Stop training if none of the val_losses are lower than val_loss_1
                                                                            verbose = 0,                  # 4. Get the trained weights corresponding to val_loss_1
                                                                            mode = "auto",
                                                                            baseline = None, 
                                                                            restore_best_weights = True)
                my_callbacks.append(early_stopping_callback)
            if "additional_val_datasets" in self.config["callbacks"]: 
                my_callbacks.append(AdditionalValidationSets(additional_validation_datasets, self.config))
                
            model.fit(x=train_dataset[0], 
                    y=train_dataset[1], 
                    epochs=self.config["train_epochs"], 
                    batch_size=self.config["mini_batch_size"], 
                    validation_data=val_dataset, 
                    callbacks=my_callbacks,
                    shuffle=False)

            # Save trained weights of the model
            if not os.path.exists("assets/trained_models/"+self.config["asset_name"]):
                os.makedirs("assets/trained_models/"+self.config["asset_name"])
            model.save_weights("assets/trained_models/"+self.config["asset_name"]+"_ckpt")

    def train_model_nested_cv(self, datasets_nested_cv):

        # Make paths
        if not os.path.exists("assets/training_log/"):
            os.makedirs("assets/training_log/")
        training_log = {}

        # nested CV
        for k_fold in range(1, self.config["k_samples"]+1):
            for l_fold in range(1, self.config["l_samples"]+1):
                
                # Create model
                # model = eval(self.config["model_name"]+"(self.config)") # {BERT} × {Static, Fine-tuning} × {no distillation, distillation}
                if self.config["model_name"] == "lstm":
                    model = lstm(self.config)
                elif self.config["model_name"] == "cnn":
                    model = lstm(self.config)
                # model.summary(line_length = 150)
                if not os.path.exists("assets/computation_graphs"):
                    os.makedirs("assets/computation_graphs")
                # plot_model(model, show_shapes = True, to_file = "assets/computation_graphs/"+self.config["asset_name"]+".png")

                # Create train dataset
                train_dataset = datasets_nested_cv["train_dataset_"+str(k_fold)+"_"+str(l_fold)]
                val_dataset = datasets_nested_cv["val_dataset_"+str(k_fold)+"_"+str(l_fold)]["test_dataset"]
                additional_val_datasets = datasets_nested_cv["val_dataset_"+str(k_fold)+"_"+str(l_fold)]

                train_sentences = train_dataset["sentence"]
                train_sentiment_labels = train_dataset["sentiment_label"]
                train_sentences, train_attention_masks = self.vectorize(train_sentences)
                train_sentiment_labels = np.array(train_sentiment_labels).astype(np.float32)

                # Create validation dataset
                val_sentences = val_dataset["sentence"]
                val_sentiment_labels = val_dataset["sentiment_label"]
                val_sentences, val_attention_masks = self.vectorize(val_sentences)
                val_sentiment_labels = np.array(val_sentiment_labels).astype(np.float32)

                # Create train rule features
                train_sentences_but_features, train_sentences_but_features_ind = self.rule_conjunct_extraction(train_dataset, rule=1)
                train_sentences_but_features, train_sentences_but_features_attention_masks = self.vectorize(train_sentences_but_features)
                train_sentences_but_features_ind = np.array(train_sentences_but_features_ind).astype(np.float32)
                train_sentences_but_features_ind = train_sentences_but_features_ind.reshape(train_sentences_but_features_ind.shape[0], 1)

                # Create val rule features
                val_sentences_but_features, val_sentences_but_features_ind = self.rule_conjunct_extraction(val_dataset, rule=1)
                val_sentences_but_features, val_sentences_but_features_attention_masks = self.vectorize(val_sentences_but_features)
                val_sentences_but_features_ind = np.array(val_sentences_but_features_ind).astype(np.float32)
                val_sentences_but_features_ind = val_sentences_but_features_ind.reshape(val_sentences_but_features_ind.shape[0], 1)

                # Remove extra samples
                train_sentences = self.remove_extra_samples(train_sentences)
                train_attention_masks = self.remove_extra_samples(train_attention_masks)
                train_sentiment_labels = self.remove_extra_samples(train_sentiment_labels)
                train_sentences_but_features = self.remove_extra_samples(train_sentences_but_features)
                train_sentences_but_features_attention_masks = self.remove_extra_samples(train_sentences_but_features_attention_masks)
                train_sentences_but_features_ind = self.remove_extra_samples(train_sentences_but_features_ind)
                val_sentences = self.remove_extra_samples(val_sentences)
                val_attention_masks = self.remove_extra_samples(val_attention_masks)
                val_sentiment_labels = self.remove_extra_samples(val_sentiment_labels)
                val_sentences_but_features = self.remove_extra_samples(val_sentences_but_features)
                val_sentences_but_features_attention_masks = self.remove_extra_samples(val_sentences_but_features_attention_masks)
                val_sentences_but_features_ind = self.remove_extra_samples(val_sentences_but_features_ind)

                if self.config["distillation"] == True:

                    # Create train and val datasets
                    train_dataset = ([[train_sentences, train_attention_masks], [train_sentences_but_features, train_sentences_but_features_attention_masks]], 
                                    [train_sentiment_labels, [train_sentences_but_features_ind]])
                    val_dataset = ([[val_sentences, val_attention_masks], [val_sentences_but_features, val_sentences_but_features_attention_masks]], 
                                    [val_sentiment_labels, [val_sentences_but_features_ind]])

                    # Train
                    my_callbacks = []
                    if "early_stopping" in self.config["callbacks"]:
                        metric = self.config["metric"]
                        patience = self.config["patience"]
                        early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor=metric,              # 1. Calculate val_loss_1 
                                                                                    min_delta = 0,                  # 2. Check val_losses for next 10 epochs 
                                                                                    patience=patience,                    # 3. Stop training if none of the val_losses are lower than val_loss_1
                                                                                    verbose=0,                      # 4. Get the trained weights corresponding to val_loss_1
                                                                                    mode="auto",
                                                                                    baseline=None, 
                                                                                    restore_best_weights=True)
                        my_callbacks.append(early_stopping_callback)
                    # additional_validation_set_callback = AdditionalValidationSets(additional_validation_datasets, self.config)
                    history = model.fit(x=train_dataset[0], 
                                        y=train_dataset[1], 
                                        epochs=self.config["train_epochs"], 
                                        batch_size=self.config["mini_batch_size"], 
                                        validation_data=val_dataset, 
                                        callbacks=my_callbacks,
                                        shuffle=False)
                    # Save trained weights of the model
                    if not os.path.exists("assets/trained_models/"+self.config["asset_name"]):
                        os.makedirs("assets/trained_models/"+self.config["asset_name"])
                    model.save_weights("assets/trained_models/"+self.config["asset_name"]+"/"+self.config["asset_name"]+"_"+str(k_fold)+"_"+str(l_fold)+"_ckpt")
                    training_log[str(k_fold)+"_"+str(l_fold)] = history.history

                else:
                    
                    # Create train and val datasets
                    train_dataset = ([train_sentences, train_attention_masks], train_sentiment_labels)
                    val_dataset = ([val_sentences, val_attention_masks], val_sentiment_labels)

                    # Train
                    if "early_stopping" in self.config["callbacks"]:
                        metric = self.config["metric"]
                        patience = self.config["patience"]
                        early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor = metric,              # 1. Calculate val_loss_1 
                                                                                    min_delta = 0,                # 2. Check val_losses for next 10 epochs 
                                                                                    patience = patience,          # 3. Stop training if none of the val_losses are lower than val_loss_1
                                                                                    verbose = 0,                  # 4. Get the trained weights corresponding to val_loss_1
                                                                                    mode = "auto",
                                                                                    baseline = None, 
                                                                                    restore_best_weights = True)

                    my_callbacks = [early_stopping_callback]
                    history = model.fit(x=train_dataset[0], 
                                        y=train_dataset[1], 
                                        epochs=self.config["train_epochs"], 
                                        batch_size=self.config["mini_batch_size"], 
                                        validation_data=val_dataset, 
                                        callbacks=my_callbacks,
                                        shuffle=False)

                    # Save trained weights of the model
                    if not os.path.exists("assets/trained_models/"+self.config["asset_name"]):
                        os.makedirs("assets/trained_models/"+self.config["asset_name"])
                    model.save_weights("assets/trained_models/"+self.config["asset_name"]+"/"+self.config["asset_name"]+"_"+str(k_fold)+"_"+str(l_fold)+"_ckpt")
                    training_log[str(k_fold)+"_"+str(l_fold)] = history.history

        # Save the history of the model
        if not os.path.exists("assets/trained_models/"+self.config["asset_name"]):
            os.makedirs("assets/trained_models/"+self.config["asset_name"])
        with open("assets/trained_models/"+self.config["asset_name"]+"/training_log.pickle", "wb") as handle:
            pickle.dump(training_log, handle)

