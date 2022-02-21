# Runtime parameters
SEED_VALUE = 3435
ASSET_NAME = "CNN-WORD2VEC-STATIC-EARLY_STOPPING-TEST" # (change manually)

# Model
if "CNN" in ASSET_NAME.split("-"):
    MODEL_NAME = "cnn" # cnn, rnn, lstm, bilstm, gru, bigru, attention based seq. models (change manually)
    N_FILTERS = 100
    FILTER_SIZES = [3,4,5]
HIDDEN_UNITS_SEQ_LAYER = 128

# Dataset parameters
DATASET_NAME = "SST2" # SST2, MR, CR, Covid-19_tweets (change manually)
if "WORD2VEC" in ASSET_NAME.split("-"):
    WORD_EMBEDDINGS = "word2vec" # word2vec, glove, fasttext, elmo, bert 
    EMBEDDING_DIM = 300 
if "STATIC" in ASSET_NAME.split("-"):
    FINE_TUNE_WORD_EMBEDDINGS = False # True, False
elif "NON_STATIC" in ASSET_NAME.split("-"):
    FINE_TUNE_WORD_EMBEDDINGS = True

# Training parameters
DROPOUT = 0.5
OPTIMIZER = "adadelta" # adam, adadelta
if OPTIMIZER == "adadelta":
    LEARNING_RATE = 1.0 # 1e-5, 3e-5, 5e-5, 10e-5
elif OPTIMIZER == "adam":
    LEARNING_RATE = 3e-5
MINI_BATCH_SIZE = 50 # 30, 50
TRAIN_EPOCHS = 20
if "EARLY_STOPPING" in ASSET_NAME.split("-"):
    VALIDATION_METHOD = "early_stopping" # early_stopping, nested_cv
elif "NESTED_CV" in ASSET_NAME.split("-"):
    VALIDATION_METHOD = "nested_cv"
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
    config = {k.lower(): v for k, v in globals().items()}
    return config
