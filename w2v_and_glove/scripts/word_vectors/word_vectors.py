from config import *

class Word_vectors(object):
    def __init__(self, config):
        self.config = config

    def load_pre_trained_word_vectors(self):
        """
        Load pre-trained word vectors
        """
        pre_trained_word_vectors = {}
        if self.config["word_embeddings"] == "word2vec":
            word_vectors_path = "datasets/pre_trained_word_vectors/word2vec/GoogleNews-vectors-negative300.bin"
            with open(word_vectors_path, "rb") as f:
                header = f.readline()
                vocab_size, layer1_size = map(int, header.split()) # 3000000, 300
                binary_len = np.dtype('float32').itemsize * layer1_size
                for line in range(vocab_size):
                    word = []
                    while True:
                        ch = f.read(1)
                        if ch == b' ':
                            word = ''.join([x.decode('latin-1') for x in word])
                            break
                        if ch != b'\n':
                            word.append(ch)
                    pre_trained_word_vectors[word] = np.frombuffer(f.read(binary_len), dtype='float32')

        elif self.config["word_embeddings"] == "glove":
            word_vectors_path = "datasets/pre_trained_word_vectors/glove/glove.840B.300d.txt"
            with open(word_vectors_path, 'rb') as f:
                glove_vocab_size, layer1_size = 2200000, 300
                for l in f:
                    line = l.decode('latin-1').split()
                    if len(line) == 301:
                        word = line[0]
                        vect = np.array(line[1:]).astype(np.float)
                        pre_trained_word_vectors[word] = vect
        return pre_trained_word_vectors
    
    def create_vocabulary(self, dataset):
        """
        Tokenize each sentence in dataset by sentence.split()
        assign each token in every sentence a unique int value (unique in the entire dataset)
        return a dictionary word_index[word] = unique int value
        """
        if tf.executing_eagerly():
            # vectorize_layer = tf.keras.layers.TextVectorization(standardize=None, split='whitespace')
            vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(standardize=None, split='whitespace')
            vectorize_layer.adapt(np.array(dataset["sentence"]))
            vocab = vectorize_layer.get_vocabulary()
            word_index = dict(zip(vocab, range(len(vocab))))
            return word_index
        else:
            tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', split=' ')
            tokenizer.fit_on_texts(dataset["sentence"])
            word_index = tokenizer.word_index
            vocab = [key for key in word_index.keys()]
            vocab.insert(0, '[UNK]')
            vocab.insert(0, '')
            word_index = dict(zip(vocab, range(len(vocab))))
            return word_index

    def create_word_vectors(self, dataset):
        pre_trained_word_vectors = self.load_pre_trained_word_vectors()
        word_index = self.create_vocabulary(dataset)
        num_tokens = len(word_index) + 2
        word_vectors = np.zeros((num_tokens, self.config["embedding_dim"]))
        hits = 0
        misses = 0  
        for word, i in word_index.items():
            embedding_vector = pre_trained_word_vectors.get(word)
            if embedding_vector is not None:
                # Words not found in embedding index will be all-zeros.
                # This includes the representation for "padding" and "OOV"
                word_vectors[i] = embedding_vector
                hits += 1
            else:
                word_vectors[i] = np.random.uniform(-0.25, 0.25, self.config["embedding_dim"])
                misses += 1
        print("\nWord vectors created")
        print("\nConverted %d words (%d misses)" % (hits, misses))
        
        if not os.path.exists("datasets/pre_trained_word_vectors/"+self.config["word_embeddings"]+"/"):
            os.makedirs("datasets/pre_trained_word_vectors/"+self.config["word_embeddings"]+"/")
        with open("datasets/pre_trained_word_vectors/"+self.config["word_embeddings"]+"/"+self.config["dataset_name"]+".npy", "wb") as handle:
            np.save(handle, word_vectors)
        
        if not os.path.exists("datasets/pre_trained_word_vectors/"+self.config["word_embeddings"]+"/"):
            os.makedirs("datasets/pre_trained_word_vectors/"+self.config["word_embeddings"]+"/")
        with open("datasets/pre_trained_word_vectors/"+self.config["word_embeddings"]+"/"+self.config["dataset_name"]+".pickle", "wb") as handle:
            pickle.dump(word_index, handle)
            
        return word_vectors, word_index

