import numpy as np
import pickle
from collections import defaultdict
import re
import pandas as pd
np.random.seed(7294258)

class Preprocess_SST2_dataset(object):
    def __init__(self, config):
        self.config = config
    
    def clean_str(self, string, TREC=False):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Every dataset is lower cased except for TREC
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string) 
        string = re.sub(r"\'ve", " \'ve", string) 
        string = re.sub(r"n\'t", " n\'t", string) 
        string = re.sub(r"\'re", " \'re", string) 
        string = re.sub(r"\'d", " \'d", string) 
        string = re.sub(r"\'ll", " \'ll", string) 
        string = re.sub(r",", " , ", string) 
        string = re.sub(r"!", " ! ", string) 
        string = re.sub(r"\(", " \( ", string) 
        string = re.sub(r"\)", " \) ", string) 
        string = re.sub(r"\?", " \? ", string) 
        string = re.sub(r"\s{2,}", " ", string)    
        return string.strip() if TREC else string.strip().lower()
    
    def clean_str_sst(self, string):
        """
        Tokenization/string cleaning for the SST dataset
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
        string = re.sub(r"\s{2,}", " ", string)    
        return string.strip().lower()
        
    def build_data(self, data_folder, clean_string=True):
        """
        Builds vocab and revs from raw data
        """
        revs = []
        [train_file,dev_file,test_file] = data_folder
        vocab = defaultdict(float)

        # Train revs
        for line in train_file:
            line = line.strip()
            y = int(line[0])
            rev = []
            rev.append(line[2:].strip())
            if clean_string:
                orig_rev = self.clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":y,
                    "text": orig_rev,                             
                    "num_words": len(orig_rev.split()),
                    "split": 0} # 0-train, 1-dev, 2-test
            revs.append(datum)

        # Dev revs
        for line in dev_file:       
            line = line.strip()
            y = int(line[0])
            rev = []
            rev.append(line[2:].strip())
            if clean_string:
                orig_rev = self.clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":y, 
                    "text": orig_rev,                             
                    "num_words": len(orig_rev.split()),
                    "split": 1}
            revs.append(datum)
            
        # Test revs
        for line in test_file:       
            line = line.strip()
            y = int(line[0])
            rev = []
            rev.append(line[2:].strip())
            if clean_string:
                orig_rev = self.clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":y, 
                    "text": orig_rev,                             
                    "num_words": len(orig_rev.split()),
                    "split": 2}
            revs.append(datum)
        return revs, vocab
    
    def conjunction_analysis(self, dataset):
        rule_label = []
        contrast = []
        for sentence in dataset['sentence']: # Check for any rule structure in no rule sentences and remove any sentence containing a rule structure
            if ' but ' in sentence:
                rule_label.append(1)
                contrast.append(1)
            else:
                rule_label.append(0)
                contrast.append(0)
        dataset['rule_label'] = rule_label
        dataset['contrast'] = contrast
        return dataset
    
    def load_bin_vec(self, fname, vocab):
        """
        Loads 300x1 word vecs from Google (Mikolov) word2vec
        """
        word_vecs = {}
        header = fname.readline()
        vocab_size, layer1_size = map(int, header.split()) # 3000000, 300
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = fname.read(1)
                if ch == b' ':
                    word = ''.join([x.decode('latin-1') for x in word])
                    break
                if ch != b'\n':
                    word.append(ch)

            if word in vocab:
                word_vecs[word] = np.fromstring(fname.read(binary_len), dtype='float32')
            else:
                fname.read(binary_len)
        return word_vecs
    
    def load_glove_vectors(self, fname, vocab):
        """
        Loads 300x1 word vecs from stanford nlp glove
        """
        word_vecs = {}
        with open(fname, 'rb') as f:
            glove_vocab_size, layer1_size = 2200000, 300
            for l in f:
                line = l.decode('latin-1').split()
                if len(line) == 301:
                    word = line[0]
                    if word in vocab:
                        vect = np.array(line[1:]).astype(np.float)
                        word_vecs[word] = vect
            
        return word_vecs

    def add_unknown_words(self, word_vecs, vocab, min_df=1, k=300):
        """
        For words that occur in at least min_df documents, create a separate word vector.    
        0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
        """
        for word in vocab:
            if word not in word_vecs and vocab[word] >= min_df:
                word_vecs[word] = np.random.uniform(-0.25,0.25,k)

    def get_W(self, word_vecs, k=300):
        """
        Get word matrix. W[i] is the vector for word indexed by i
        """
        vocab_size = len(word_vecs)
        word_idx_map = dict()
        W = np.zeros(shape=(vocab_size+1, k), dtype='float32')            
        W[0] = np.zeros(k, dtype='float32')
        i = 1
        for word in word_vecs:
            W[i] = word_vecs[word]
            word_idx_map[word] = i
            i += 1
        return W, word_idx_map

    def preprocess_SST2_sentences(self, train_data_file, dev_data_file, test_data_file, pre_trained_word_vectors_file):
        """
        Main function for this class
        """
        # Preprocess sentences 
        data_folder = [train_data_file, dev_data_file, test_data_file]       
        revs, vocab = self.build_data(data_folder, clean_string=True)

        w2v = self.load_bin_vec(pre_trained_word_vectors_file, vocab)
        self.add_unknown_words(w2v, vocab)
        W, word_idx_map = self.get_W(w2v)
        rand_vecs = {}
        self.add_unknown_words(rand_vecs, vocab)
        W2, _ = self.get_W(rand_vecs)

        # # Add rule label
        # dataset = self.conjunction_analysis(dataset)

        return revs, W, W2, word_idx_map, vocab

class Preprocess_raw_data(object):
    def __init__(self, config):
        self.config = config
    
    def preprocess(self):
        stsa_path = "datasets/"+self.config["dataset_name"]+"/"+"raw_dataset"
        train_data_file = open("%s/stsa.binary.train" % stsa_path, "r")
        dev_data_file = open("%s/stsa.binary.dev" % stsa_path, "r")
        test_data_file = open("%s/stsa.binary.test" % stsa_path, "r")
        pre_trained_word_vectors_file = open("datasets/pre_trained_word_vectors/"+self.config["word_embeddings"]+"/GoogleNews-vectors-negative300.bin", "rb")
        revs, word_vectors, random_word_vectors, word_idx_map, vocab  = Preprocess_SST2_dataset(self.config).preprocess_SST2_sentences(train_data_file, dev_data_file, test_data_file, pre_trained_word_vectors_file)
        return revs, word_vectors, random_word_vectors, word_idx_map, vocab