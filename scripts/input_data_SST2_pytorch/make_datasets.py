import numpy as np

class Make_datasets(object):
    def __init__(self, revs, word_vectors, random_word_vectors, word_idx_map, vocab):
        self.revs = revs
        self.word_vectors = word_vectors
        self.random_word_vectors = random_word_vectors
        self.word_idx_map = word_idx_map
        self.vocab = vocab

    def get_idx_from_sent(self, sent, word_idx_map, max_l=51, k=300, filter_h=5):
        """
        Transforms sentence into a list of indices. Pad with zeroes.
        """
        x = []
        pad = filter_h - 1
        for i in range(pad):
            x.append(0)
        words = sent.split()
        for word in words:
            if word in word_idx_map:
                x.append(word_idx_map[word])
        while len(x) < max_l+2*pad:
            x.append(0)
        return x

    def make_idx_data(self, revs, word_idx_map, max_l, k, filter_h):
        """
        Transforms sentences into a 2-d matrix.
        """
        train, dev, test = [], [], []
        train_text, dev_text, test_text = [], [], []
        for i,rev in enumerate(revs):
            sent = self.get_idx_from_sent(rev["text"], word_idx_map, max_l, k, filter_h)
            sent.append(rev["y"])
            if rev["split"]==0:
                train.append(sent)
                train_text.append(rev["text"])
            elif rev["split"]==1:
                dev.append(sent)
                dev_text.append(rev["text"])
            else:  
                test.append(sent)
                test_text.append(rev["text"])
        train = np.array(train,dtype="int")
        dev = np.array(dev,dtype="int")
        test = np.array(test,dtype="int")
        train_text = np.array(train_text)
        dev_text = np.array(dev_text)
        test_text = np.array(test_text)
        return [train, dev, test, train_text, dev_text, test_text]

    def datasets(self):

        """
        Main function of this class

        """
        """
        Description
            
            datasets: A list of 9 objects.

                    datasets[0]: A 2-d matrix of train texts where each row is a text in 1x62 vector form and each value represents index value of a word in 
                                word_idx_map and the last value represents the sentiment value. For eg, datasets[0][42] will output a 1x62 vector 
                                corresponding to revs[42]['text'] text value in which say the first value - 501 - will represent index value of first word 
                                'singer' in corresponding text in word_idx_map. 76961x62 in shape.

                    datasets[1]: A 2-d matrix of dev texts where each row is a text in 1x62 vector form and each value represents 
                                index value of a word in word_idx_map and the last value represents the sentiment value. For eg, datasets[1][42] will output 
                                a 1x62 vector corresponding to revs[76961 + 42]['text'] text value in which say 13465 will represent index value of first 
                                word('despite') in text value in word_idx_map. 872x62 in shape.

                    datasets[2]: A 2-d matrix of dev texts where each row is a text in 1x62 vector form and each value represents index value of a word in 
                                word_idx_map and the last value represents the sentiment value. 872x62 in shape. For eg, datasets[2][42] will output a 1x62 
                                vector corresponding to revs[76961 + 872 + 42]['text'] text value in which say 5084 will represent index value of first word 
                                ('the') in text value in word_idx_map. 1821x62 in shape.

                    datasets[3]: A dictionary containing 3 keys in which datasets[3]['but_text']: Outputs a list of but-features of train texts in revs. 76961x1 in shape. 
                                datasets[3]['but_ind']: Outputs a list of but-indices of train texts in revs. 76961x1 in shape.
                                datasets[3]['but']: Ouputs a list of vectorized forms of corresponding but-features in datasets[3]['but_text']. 76961x61 in shape.

                    datasets[4]: A dictionary containing 3 keys in which 
                                datasets[4]['but_text']: Outputs a list of but-features of dev texts in revs. 872x1 in shape.
                                datasets[4]['but_ind']: Outputs a list of but-indices of dev texts in revs. 872x1 in shape.
                                datasets[4]['but']: Ouputs a list of vectorized forms of corresponding but-features in 
                                                    datasets[4]['but_text']. 872x61 in shape.

                    datasets[5]: A dictionary containing 3 keys in which 
                                datasets[5]['but_text']: Outputs a list of but-features of test texts in revs. 1821x1 in shape.
                                datasets[5]['but_ind']: Outputs a list of but-indices of test texts in revs. 1821x1 in shape.
                                datasets[5]['but']: Ouputs a list of vectorized forms of corresponding but-features in 
                                                    datasets[5]['but_text']. 1821x61 in shape.

                    datasets[6]: A list of train texts. 76961x1 in shape.

                    datasets[7]: A list of dev texts. 872x1 in shape.

                    datasets[8]: A list of test texts. 1821x1 in shape.
        """
        datasets = self.make_idx_data(revs = self.revs, word_idx_map = self.word_idx_map, max_l = 53,k = 300, filter_h = 5)
        return datasets