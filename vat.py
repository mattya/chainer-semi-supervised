# coding: utf-8

# In[1]:

import sys
import cPickle as pickle
import datetime, math, sys, time

from sklearn.datasets import fetch_mldata
import numpy as np

import chainer
from chainer import serializers
import chainer.functions as F
from chainer import FunctionSet, Variable, optimizers, cuda

print 'fetch MNIST dataset'
mnist = fetch_mldata('MNIST original', data_home='.')
mnist.data = mnist.data.astype(np.float32)/128.0 - 1.0
#mnist.data = mnist.data.astype(np.float32)/256.0
mnist.target = mnist.target.astype(np.int32)


cuda.get_device(0).use()
xp = cuda.cupy

class KL_multinomial(chainer.function.Function):

    """ KL divergence between multinomial distributions """

    def __init__(self):
        pass

    def forward_gpu(self, inputs):
        p, q = inputs
        loss = cuda.cupy.ReductionKernel(
            'T p, T q',
            'T loss',
            'p*(log(p)-log(q))',
            'a + b',
            'loss = a',
            '0',
            'kl'
        )(p,q)
        return loss/np.float32(p.shape[0]),

    # backward only q-side
    def backward_gpu(self, inputs, grads):
        p, q = inputs
        return cuda.cupy.zeros_like(p), -np.float32(1.0)*p/(np.float32(1e-8)+q)/np.float32(p.shape[0])

def kl(p,q):
    return KL_multinomial()(p,q)

def distance(y0, y1):
    return kl(F.softmax(y0), F.softmax(y1))
# In[2]:
def vat(forward, distance, x, xi=10, eps=1.4, Ip=1):
    y = forward(Variable(x))
    y.unchain_backward()

    # calc adversarial direction
    d = xp.random.normal(size=x.shape, dtype=np.float32)
    d = d/xp.sqrt(xp.sum(d**2, axis=1)).reshape((x.shape[0],1))
    d_var = Variable(d.astype(np.float32))
    for ip in range(Ip):
        y2 = forward(x+xi*d_var)
        kl_loss = distance(y, y2)
        kl_loss.backward()
        d = d_var.grad
        d_var = Variable(d.astype(np.float32))
    d = d/xp.sqrt(xp.sum(d**2, axis=1)).reshape((x.shape[0],1))
    d_var = Variable(d.astype(np.float32))

    # calc regularization
    y2 = forward(x+eps*d_var)
    return distance(y, y2)

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

# In[3]:

N_train_labeled = 60000
N_train_unlabeled = 60000
N_test = 10000
perm = np.random.permutation(70000)

# equal number of data in each category
cnt_l = [0] * 10
cnt_ul = [0] * 10
cnt_test = [0] * 10
ind_l = []
ind_ul = []
ind_test = []
for i in range(70000):
    l = mnist.target[perm[i]]
    if cnt_l[l] < N_train_labeled/10:
        ind_l.append(perm[i])
        ind_ul.append(perm[i])
        cnt_l[l] += 1
        cnt_ul[l] += 1
    elif cnt_test[l] < N_test/10:
        ind_test.append(perm[i])
        cnt_test[l] += 1
    elif cnt_ul[l] < N_train_unlabeled/10:
        ind_ul.append(perm[i])
        cnt_ul[l] += 1

print cnt_l, cnt_ul, cnt_test
x_train_l = mnist.data[ind_l]
x_train_ul = mnist.data[ind_ul]
x_test = mnist.data[ind_test]
y_train_l = mnist.target[ind_l]
y_train_ul = mnist.target[ind_ul]
y_test = mnist.target[ind_test]

train_l = Data(x_train_l, y_train_l)
train_ul = Data(x_train_ul, y_train_ul)
test_set = Data(x_test, y_test)

#train_ul.put(train_l.data, train_l.label)

print train_l.data.shape, train_ul.data.shape, test_set.data.shape
'''
class Encoder(chainer.Chain):
    def __init__(self):
        n_units = 1200
        super(Encoder, self).__init__(
            l1 = F.Linear(784, n_units, wscale=0.1),
            l2 = F.Linear(n_units, n_units, wscale=0.1),
            l3 = F.Linear(n_units, 10, wscale=0.0001),
            bn1 = F.BatchNormalization(n_units),
            bn2 = F.BatchNormalization(n_units)
        )
    def __call__(self, x, test=False):
        h = F.relu(self.bn1(self.l1(x), test=test))
        h = F.relu(self.bn2(self.l2(h), test=test))
        y = self.l3(h)

        return y
'''
class Encoder(chainer.Chain):
    def __init__(self):

        super(Encoder, self).__init__(
            l1 = F.Linear(784, 1000, wscale=0.1),
            l2 = F.Linear(1000, 500, wscale=0.1),
            l3 = F.Linear(500, 250, wscale=0.1),
            l4 = F.Linear(250, 250, wscale=0.1),
            l5 = F.Linear(250, 250, wscale=0.1),
            l6 = F.Linear(250, 10, wscale=0.0001),
            bn1 = F.BatchNormalization(1000),
            bn2 = F.BatchNormalization(500),
            bn3 = F.BatchNormalization(250),
            bn4 = F.BatchNormalization(250),
            bn5 = F.BatchNormalization(250)
        )
    def __call__(self, x, test=False):
        h = F.relu(self.bn1(self.l1(x), test=test))
        h = F.relu(self.bn2(self.l2(h), test=test))
        h = F.relu(self.bn3(self.l3(h), test=test))
        h = F.relu(self.bn4(self.l4(h), test=test))
        h = F.relu(self.bn5(self.l5(h), test=test))
        y = self.l6(h)

        return y


def loss_labeled(x, t):
    y = enc(x, test=False)
    #print "train",y.data[0],t.data[0]
    L = F.softmax_cross_entropy(y, t)
    return L

def loss_unlabeled(x):
    L = vat(enc, distance, x.data)
    return L


def loss_test(x, t):
    y = enc(x, test=True)
    #print "test",y.data[0],t.data[0]
    L, acc = F.softmax_cross_entropy(y, t), F.accuracy(y, t)
    return L, acc


enc = Encoder()
enc.to_gpu()

o_enc = optimizers.Adam(alpha=0.002, beta1=0.9)
alpha_plan = [0.002] * 100
for i in range(1,100):
    alpha_plan[i] = alpha_plan[i-1]*0.9

o_enc.setup(enc)

#o_enc.add_hook(chainer.optimizer.WeightDecay(0.0001), 'hook_dis1')

batchsize_l = 100
batchsize_ul = 250


for epoch in range(len(alpha_plan)):
    print epoch

    sum_loss_l = 0
    sum_loss_ul = 0
    for it in range(60000/100):
        x,t = train_l.get(batchsize_l)
        loss_l = loss_labeled(Variable(x), Variable(t))

        o_enc.zero_grads()
        loss_l.backward()
        #o_enc.clip_grads(1.0)
        o_enc.update()

        x,_ = train_ul.get(batchsize_ul)
        loss_ul = loss_unlabeled(Variable(x))

        o_enc.zero_grads()
        loss_ul.backward()
        #o_enc.clip_grads(1.0)
        o_enc.update()

        sum_loss_l += loss_l.data
        sum_loss_ul += loss_ul.data

        loss_l.unchain_backward()
        loss_ul.unchain_backward()

    print "classification loss, reconstruction loss: ", sum_loss_l/600, sum_loss_ul/600
    o_enc.alpha = alpha_plan[epoch]

    x,t = test_set.get(10000, balance=False)
    L, acc = loss_test(Variable(x), Variable(t))
    #L.unchain_backward()
    #acc.unchain_backward()
    x,t = train_l.get(100)
    L_, acc_ = loss_test(Variable(x), Variable(t))
    L_.unchain_backward()
    acc_.unchain_backward()
    print "test error, test acc, train error, train acc: ", L.data, acc.data, L_.data, acc_.data
    sys.stdout.flush()
    #if (epoch+1)%10==0:
    #    serializers.save_hdf5("enc4_%d.h5"%epoch, enc)
    #    serializers.save_hdf5("dec4_%d.h5"%epoch, dec)
