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

    from scripts.models.models import cnn
    model = cnn(config)
    model.summary(line_length = 150)

    # Create input data
    if config["create_input_data"] == True:
        print("\nCreating input data")
        preprocessed_dataset = Preprocess_dataset(config).preprocess()
        if config["validation_method"]=="early_stopping":
            train_dataset, val_dataset, test_dataset = Dataset_division(config).train_val_test_split(preprocessed_dataset, divide_into_rule_sections=True)
        elif config["validation_method"]=="nested_cv":
            datasets_nested_cv = Dataset_division(config).nested_cv_split(preprocessed_dataset, divide_into_rule_sections=True)
    else:
        preprocessed_dataset = pickle.load(open("datasets/"+config["dataset_name"]+"/preprocessed_dataset/"+config["dataset_name"]+".pickle", "rb"))
        if config["validation_method"]=="early_stopping":
            train_dataset, val_dataset, test_dataset = Dataset_division(config).train_val_test_split(preprocessed_dataset, divide_into_rule_sections=True)
        elif config["validation_method"]=="nested_cv":
            datasets_nested_cv = Dataset_division(config).nested_cv_split(preprocessed_dataset, divide_into_rule_sections=True)

    # Train model (asset 1: create trained_models and training_log)
    if config["train_model"] == True:
        print("\nTraining")
        if config["validation_method"]=="early_stopping":
            Train(config).train_model(train_dataset, val_dataset, test_dataset)
        elif config["validation_method"]=="nested_cv":
            Train(config).train_model_nested_cv(datasets_nested_cv)

    # Evaluate trained models
    if config["evaluate_model"] == True:
        print("\nEvaluation")
        if config["validation_method"] == "early_stopping":
            Evaluation(config).evaluate_model(test_dataset)
        elif config["validation_method"] == "nested_cv":
            Evaluation(config).evaluate_model_nested_cv(datasets_nested_cv)
    
    # LIME explanations
    if config["generate_lime_explanations"] == True:
        print("\nLIME explanations")
        if config["validation_method"] == "early_stopping":
            Lime_explanations(config).create_lime_explanations(test_dataset)
        elif config["validation_method"] == "nested_cv":
            Lime_explanations(config).create_lime_explanations_nested_cv(datasets_nested_cv)

    # SHAP explanations
    if config["generate_shap_explanations"] == True:
        print("\nSHAP explanations")
        if config["validation_method"] == "early_stopping":
            Shap_explanations(config).create_shap_explanations(train_dataset, test_dataset)
        elif config["validation_method"] == "nested_cv":
            Shap_explanations(config).create_shap_explanations_nested_cv(datasets_nested_cv)
    
    # Inegrated Gradients explanations (asset 5: generates int_grad_explanations for results)
    if config["generate_int_grad_explanations"] == True:
        print("\nINT-GRAD explanations")
        if config["validation_method"] == "early_stopping":
            Int_grad_explanations(config).create_explanations(test_dataset)
        elif config["validation_method"] == "nested_cv":
            Int_grad_explanations(config).create_explanations_nested_cv(datasets_nested_cv)

    # Robustness metric (asset 6: generates lipschitz_estimates in lime explanations)
    if config["generate_lipschitz_scores_lime"] == True:
        print("\nLocal Lipschitz Estimate for LIME")
        Local_lipschitz_estimate(config).lime_explanations_estimates()

    # Robustness metric (asset 7: generates lipschitz_estimates in shap explanations)
    if config["generate_lipschitz_scores_shap"] == True:
        print("\nLocal Lipschitz Estimate for SHAP")
        Local_lipschitz_estimate(config).shap_explanations_estimates()
    
    # Robustness metric (asset 8: generates lipschitz_estimates in int_grad explanations)
    if config["generate_lipschitz_scores_int_grad"] == True:
        print("\nLocal Lipschitz Estimate for Int-grad")
        Local_lipschitz_estimate(config).int_grad_explanations_estimates()