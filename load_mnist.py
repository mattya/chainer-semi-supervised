import sys
import cPickle as pickle
import datetime, math, sys, time

from sklearn.datasets import fetch_mldata
import numpy as np

from chainer import cuda


class Data:
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def get(self, n, balance=True):
        ind = np.random.permutation(self.data.shape[0])
        if not balance:
            return cuda.to_gpu(self.data[ind[:n],:].astype(np.float32)), cuda.to_gpu(self.label[ind[:n]].astype(np.int32))
        else:
            cnt = [0]*10
            m = 0
            ret_data = np.zeros((n, self.data.shape[1])).astype(np.float32)
            ret_label = np.zeros(n).astype(np.int32)
            for i in range(self.data.shape[0]):
                if cnt[self.label[ind[i]]] < n/10:
                    ret_data[m,:] = self.data[ind[i]]
                    ret_label[m] = self.label[ind[i]]

                    cnt[self.label[ind[i]]] += 1
                    m += 1
                    if m==n:
                        break
            return cuda.to_gpu(ret_data), cuda.to_gpu(ret_label)

    def put(self, data, label):
        if self.data is None:
            self.data = cuda.to_cpu(data)
            self.label = cuda.to_cpu(label)
        else:
            self.data = np.vstack([self.data, cuda.to_cpu(data)])
            self.label = np.hstack([self.label, cuda.to_cpu(label)]).reshape((self.data.shape[0]))

def load_mnist(scale, shift, N_train_labeled, N_train_unlabeled, N_test):
    print 'fetch MNIST dataset'
    mnist = fetch_mldata('MNIST original', data_home='.')
    #mnist.data = mnist.data.astype(np.float32)/128.0 - 1.0
    mnist.data = mnist.data.astype(np.float32)*scale + shift
    mnist.target = mnist.target.astype(np.int32)

    perm = np.random.permutation(60000)

    # equal number of data in each category
    cnt_l = [0] * 10
    cnt_ul = [0] * 10
    cnt_test = [0] * 10
    ind_l = []
    ind_ul = []
    ind_test = range(60000,70000)

    for i in range(60000):
        l = mnist.target[perm[i]]
        if cnt_l[l] < N_train_labeled/10:
            ind_l.append(perm[i])
            ind_ul.append(perm[i])
            cnt_l[l] += 1
            cnt_ul[l] += 1
        else:
            ind_ul.append(perm[i])
            cnt_ul[l] += 1

    #print cnt_l, cnt_ul, cnt_test
    x_train_l = mnist.data[ind_l]
    x_train_ul = mnist.data[ind_ul]
    x_test = mnist.data[ind_test]
    y_train_l = mnist.target[ind_l]
    y_train_ul = mnist.target[ind_ul]
    y_test = mnist.target[ind_test]

    train_l = Data(x_train_l, y_train_l)
    train_ul = Data(x_train_ul, y_train_ul)
    test_set = Data(x_test, y_test)


    print "load mnist done", train_l.data.shape, train_ul.data.shape, test_set.data.shape
    return train_l, train_ul, test_set
