# Runtime parameters
SEED_VALUE = 3435
ASSET_NAME = "cnn_model-WORD2VEC-NON_STATIC-EARLY_STOPPING-TEST" # (change manually)

# Model
MODEL_NAME = "cnn" # cnn, rnn, lstm, bilstm, gru, bigru, attention based seq. models
N_FILTERS = 100
FILTER_SIZES = [3,4,5]
HIDDEN_UNITS_SEQ_LAYER = 128

# Dataset parameters
DATASET_NAME = "SST2" # SST2, MR, CR, Covid-19_tweets
WORD_EMBEDDINGS = "word2vec" # word2vec, glove, fasttext, elmo, bert
EMBEDDING_DIM = 300
FINE_TUNE_WORD_EMBEDDINGS = True

# Training parameters
DROPOUT = 0.5
OPTIMIZER = "adadelta" # adam, adadelta (change manually)
LEARNING_RATE = 1.0 # 1e-5, 3e-5, 5e-5, 10e-5
MINI_BATCH_SIZE = 50 # 30, 50
TRAIN_EPOCHS = 20
VALIDATION_METHOD = "early_stopping" # early_stopping, nested_cv
K_SAMPLES = 5
L_SAMPLES = 3
SAMPLING = "stratified"

# IKD parameters
DISTILLATION = True
RULES_LAMBDA = [1]
TEACHER_REGULARIZER = 6.0

# Lime parameters
LIME_NO_OF_SAMPLES = 1000

def load_configuration_parameters():
    config = {"asset_name":ASSET_NAME,
                "model_name":MODEL_NAME,
                "n_filters":N_FILTERS,
                "filter_sizes":FILTER_SIZES,
                "seed_value":SEED_VALUE,
                "dataset_name":DATASET_NAME,
                "word_embeddings":WORD_EMBEDDINGS,
                "embedding_dim":EMBEDDING_DIM,
                "fine_tune_word_embeddings":FINE_TUNE_WORD_EMBEDDINGS,
                "validation_method":VALIDATION_METHOD,
                "optimizer":OPTIMIZER,
                "learning_rate":LEARNING_RATE, 
                "mini_batch_size":MINI_BATCH_SIZE, 
                "train_epochs":TRAIN_EPOCHS,
                "dropout":DROPOUT,
                "lime_no_of_samples":LIME_NO_OF_SAMPLES,
                "sampling":SAMPLING,
                "k_samples":K_SAMPLES,
                "l_samples":L_SAMPLES,
                "distillation":DISTILLATION}
    return config
