from config import *

# Train a NN model for binary sentiment classification
if __name__=="__main__":
    
    # Gather configuration parameters
    config = load_configuration_parameters()
    print("\n"+config["asset_name"])

    # Set the seed value
    os.environ['PYTHONHASHSEED']=str(config["seed_value"])
    random.seed(config["seed_value"])
    np.random.seed(config["seed_value"])
    tf.random.set_seed(config["seed_value"])

    # Create input data
    if config["create_input_data"] == True:
        print("\nCreating input data")
        preprocessed_dataset = Preprocess_dataset(config).preprocess()
        word_vectors, word_index = Word_vectors(config).create_word_vectors(preprocessed_dataset)
        if config["validation_method"]=="early_stopping":
            train_dataset, val_dataset, test_dataset = Dataset_division(config).train_val_test_split(preprocessed_dataset, divide_into_rule_sections=True)
        elif config["validation_method"]=="nested_cv":
            datasets_nested_cv = Dataset_division(config).nested_cv_split(preprocessed_dataset, divide_into_rule_sections=True)
    else:
        preprocessed_dataset = pickle.load(open("datasets/"+config["dataset_name"]+"/preprocessed_dataset/"+config["dataset_name"]+".pickle", "rb"))
        word_vectors = np.load(open("datasets/pre_trained_word_vectors/"+config["word_embeddings"]+"/"+config["dataset_name"]+".npy", "rb"))
        word_index = pickle.load(open("datasets/pre_trained_word_vectors/"+config["word_embeddings"]+"/"+config["dataset_name"]+".pickle", "rb"))
        if config["validation_method"]=="early_stopping":
            train_dataset, val_dataset, test_dataset = Dataset_division(config).train_val_test_split(preprocessed_dataset, divide_into_rule_sections=True)
        elif config["validation_method"]=="nested_cv":
            datasets_nested_cv = Dataset_division(config).nested_cv_split(preprocessed_dataset, divide_into_rule_sections=True)

    # Train model (asset 1: generates trained_models and training_log)
    if config["train_model"] == True:
        print("\nTraining")
        if config["validation_method"]=="early_stopping":
            Train(config).train_model(train_dataset, val_dataset, test_dataset, word_vectors, word_index)
        elif config["validation_method"]=="nested_cv":
            Train(config).train_model_nested_cv(datasets_nested_cv, word_vectors, word_index)

    # Evaluate model (asset 2: generates results)
    if config["evaluate_model"] == True:
        print("\nEvaluation")
        if config["validation_method"] == "early_stopping":
            Evaluation(config).evaluate_model(test_dataset, word_vectors, word_index)
        elif config["validation_method"] == "nested_cv":
            Evaluation(config).evaluate_model_nested_cv(datasets_nested_cv, word_vectors, word_index)

    # LIME explanations (asset 3: generates lime_explanations for results) ### TAKING A LOT OF TIME FOR SENTIMENT140 DATASET (REDUCE IT'S SIZE MAYBE)
    if config["generate_lime_explanations"] == True:
        print("\nLIME explanations")
        if config["validation_method"] == "early_stopping":
            Lime_explanations(config).create_explanations(test_dataset, word_vectors, word_index)
        elif config["validation_method"] == "nested_cv":
            Lime_explanations(config).create_explanations_nested_cv(datasets_nested_cv, word_vectors, word_index)
    
    # SHAP explanations (asset 4: generates shap_explanations for results)
    if config["generate_shap_explanations"] == True:
        print("\nSHAP explanations")
        if config["validation_method"] == "early_stopping":
            Shap_explanations(config).create_explanations(train_dataset, test_dataset, word_vectors, word_index)
        elif config["validation_method"] == "nested_cv":
            Shap_explanations(config).create_explanations_nested_cv(datasets_nested_cv, word_vectors, word_index)
    
    # Inegrated Gradients explanations (asset 5: generates int_grad_explanations for results)
    if config["generate_int_grad_explanations"] == True:
        print("\nINT-GRAD explanations")
        if config["validation_method"] == "early_stopping":
            Int_grad_explanations(config).create_explanations(test_dataset, word_vectors, word_index)
        elif config["validation_method"] == "nested_cv":
            Int_grad_explanations(config).create_explanations_nested_cv(datasets_nested_cv, word_vectors, word_index)
        
    # Robustness metric (asset 6: generates lipschitz_estimates in lime explanations)
    if config["generate_lipschitz_scores_lime"] == True:
        print("\nLocal Lipschitz Estimate for LIME")
        Local_lipschitz_estimate(config).lime_explanations_estimates(word_index)

    # Robustness metric (asset 7: generates lipschitz_estimates in shap explanations)
    if config["generate_lipschitz_scores_shap"] == True:
        print("\nLocal Lipschitz Estimate for SHAP")
        Local_lipschitz_estimate(config).shap_explanations_estimates(word_index)
    
    # Robustness metric (asset 8: generates lipschitz_estimates in int_grad explanations)
    if config["generate_lipschitz_scores_int_grad"] == True:
        print("\nLocal Lipschitz Estimate for Int-grad")
        Local_lipschitz_estimate(config).int_grad_explanations_estimates(word_index)

    # Save the configuration parameters (asset 9: generates configurations) (marks creation of an result to be presented)
    if "TEST" not in config["asset_name"]: 
        if not os.path.exists("assets/configurations/"):
            os.makedirs("assets/configurations/")
        with open("assets/configurations/"+config["asset_name"]+".pickle", 'wb') as handle:
            pickle.dump(config, handle)