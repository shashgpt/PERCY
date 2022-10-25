import numpy as np

def pearson_corr_consistency(configurations, distributions, metric_1, metric_2, consistency_value):
    for config in configurations:
        try:
            base_model = config.split("-")[0]
            word_vectors = config.split("-")[1]
            fine_tuning = config.split("-")[2]
            dataset = config.split("-")[3]
            validation_method = config.split("-")[4]
            if "DISTILLATION" in list(config.split("-")):
                distillation = config.split("-")[5]
                EA_values_lime = distributions[base_model][word_vectors][fine_tuning][dataset][validation_method][distillation][metric_1]
                EA_values_shap = distributions[base_model][word_vectors][fine_tuning][dataset][validation_method][distillation][metric_2]
                my_rho = np.corrcoef(EA_values_lime, EA_values_shap)[0][1]
                distributions[base_model][word_vectors][fine_tuning][dataset][validation_method][distillation][consistency_value] = my_rho
            else:
                EA_values_lime = distributions[base_model][word_vectors][fine_tuning][dataset][validation_method][metric_1]
                EA_values_shap = distributions[base_model][word_vectors][fine_tuning][dataset][validation_method][metric_2]
                my_rho = np.corrcoef(EA_values_lime, EA_values_shap)[0][1]
                distributions[base_model][word_vectors][fine_tuning][dataset][validation_method][consistency_value] = my_rho
        except:
            base_model = config.split("-")[0]
            word_vectors = config.split("-")[1]
            fine_tuning = config.split("-")[2]
            dataset = "SST2"
            validation_method = config.split("-")[3]
            if "DISTILLATION" in list(config.split("-")):
                distillation = config.split("-")[4]
                EA_values_lime = distributions[base_model][word_vectors][fine_tuning][validation_method][distillation][metric_1]
                EA_values_shap = distributions[base_model][word_vectors][fine_tuning][validation_method][distillation][metric_2]
                my_rho = np.corrcoef(EA_values_lime, EA_values_shap)[0][1]
                distributions[base_model][word_vectors][fine_tuning][validation_method][distillation][consistency_value] = my_rho
            else:
                EA_values_lime = distributions[base_model][word_vectors][fine_tuning][validation_method][metric_1]
                EA_values_shap = distributions[base_model][word_vectors][fine_tuning][validation_method][metric_2]
                my_rho = np.corrcoef(EA_values_lime, EA_values_shap)[0][1]
                distributions[base_model][word_vectors][fine_tuning][validation_method][consistency_value] = my_rho
    return distributions