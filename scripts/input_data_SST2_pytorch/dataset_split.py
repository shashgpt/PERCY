import copy
import numpy as np
from sklearn.model_selection import StratifiedKFold

class Dataset_split(object):
    def __init__(self, config):
        np.random.seed(3435)
        self.config = config

    def train_split(self, datasets_in_study):
        datasets_in_study = datasets_in_study
        img_h = len(datasets_in_study[0][0])-1
        batch_size = self.config["mini_batch_size"]
        if datasets_in_study[0].shape[0] % batch_size > 0:
            extra_data_num = batch_size - datasets_in_study[0].shape[0] % batch_size
            permutation_order = np.random.permutation(datasets_in_study[0].shape[0])
            train_set = datasets_in_study[0][permutation_order]
            extra_data = train_set[:extra_data_num]
            new_data = np.append(datasets_in_study[0],extra_data,axis=0)
        else:
            new_data = datasets_in_study[0]
        permutation_order = np.random.permutation(new_data.shape[0]) #shuffle both training data and features
        new_data = new_data[permutation_order]
        n_batches = new_data.shape[0]/batch_size
        n_train_batches = n_batches
        train_set = new_data
        train_set_x, train_set_y = train_set[:,:img_h],train_set[:,-1]
        return train_set_x, train_set_y

    def val_split(self, datasets_in_study):
        datasets_in_study = datasets_in_study
        img_h = len(datasets_in_study[0][0])-1
        batch_size = self.config["mini_batch_size"]
        if datasets_in_study[1].shape[0] % batch_size > 0:
            extra_data_num = batch_size - datasets_in_study[1].shape[0] % batch_size
            permutation_order = np.random.permutation(datasets_in_study[1].shape[0])
            val_set = datasets_in_study[1][permutation_order]
            extra_data = val_set[:extra_data_num]
            new_data=np.append(datasets_in_study[1],extra_data,axis=0)
        else:
            new_data = datasets_in_study[1]
        permutation_order = np.random.permutation(new_data.shape[0]) #shuffle both training data and features
        new_data = new_data[permutation_order]
        n_batches = new_data.shape[0]/batch_size
        n_train_batches = n_batches
        val_set = new_data
        val_set_x, val_set_y = val_set[:,:img_h],val_set[:,-1]
        return val_set_x, val_set_y

    def test_split(self, datasets_in_study):
        datasets_in_study = datasets_in_study
        img_h = len(datasets_in_study[0][0])-1
        batch_size = self.config["mini_batch_size"]
        test_set_x = datasets_in_study[2][:,:img_h] 
        test_set_y = np.asarray(datasets_in_study[2][:,-1],"int32")
        return test_set_x, test_set_y
    
    def test_split_cv(self, datasets_in_study):
        datasets_in_study = datasets_in_study
        img_h = len(datasets_in_study[0][0])-1
        batch_size = self.config["mini_batch_size"]
        test_set_x = datasets_in_study[0][:,:img_h] 
        test_set_y = np.asarray(datasets_in_study[0][:,-1],"int32")
        return test_set_x, test_set_y
    
    def nested_kl_cv(self, datasets_in_study):
        
        k = self.config["k_samples"]
        l = self.config["l_samples"]
            
        dataset_train = np.array(datasets_in_study[0], dtype = int)
        dataset_val = np.array(datasets_in_study[1], dtype = int)
        dataset_test = np.array(datasets_in_study[2], dtype = int)
        dataset_total = np.concatenate((dataset_train, dataset_val, dataset_test), axis = 0)

        X = dataset_total[:,:61] # sentences in vector forms
        Y = dataset_total[:,61] # labels

        datasets_kl_cv = {}
        skf = StratifiedKFold(n_splits = k)
        skf.get_n_splits(X, Y)
        k_fold=1

        for train_index_k, test_index_k in skf.split(X, Y):
            X_train_k, Y_train_k = X[train_index_k], Y[train_index_k]
            slf = StratifiedKFold(n_splits = l)
            slf.get_n_splits(X_train_k, Y_train_k)
            l_fold=1

            for train_index_l, val_index_l in slf.split(X_train_k, Y_train_k):
                X_train_l, Y_train_l = X_train_k[train_index_l], Y_train_k[train_index_l]
                datasets_train_l = []
                datasets_train_l.append(np.concatenate((X_train_l, Y_train_l.reshape(len(Y_train_l), 1)), axis=1))
                X_train_l, Y_train_l = self.train_split(datasets_in_study = datasets_train_l)
                X_val_l, Y_val_l = X_train_k[val_index_l], Y_train_k[val_index_l]
                datasets_val_l = []
                datasets_val_l.append(np.concatenate((X_val_l, Y_val_l.reshape(len(Y_val_l), 1)), axis=1))
                X_val_l_full_set, Y_val_l_full_set = self.test_split_cv(datasets_val_l)
                datasets_kl_cv["train_"+str(k_fold)+"_"+str(l_fold)] = [X_train_l, Y_train_l]
                datasets_kl_cv["val_"+str(k_fold)+"_"+str(l_fold)] = [X_val_l_full_set, Y_val_l_full_set]
                l_fold=l_fold+1
            k_fold=k_fold+1

        return datasets_kl_cv