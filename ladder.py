
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
#mnist.data = mnist.data.astype(np.float32)/128.0 - 1.0
mnist.data = mnist.data.astype(np.float32)/256.0
mnist.target = mnist.target.astype(np.int32)


cuda.get_device(1).use()
xp = cuda.cupy


# In[2]:

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

N_train_labeled = 100
N_train_unlabeled = 60000
N_test = 10000
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


class AMLP(chainer.Chain):
    def __init__(self):
        super(AMLP, self).__init__(
            l0 = F.Linear(3,2, wscale=0.01),
            l1 = F.Linear(2,2, wscale=0.01),
            l2 = F.Linear(2,2, wscale=0.01),
            l3 = F.Linear(2,1, wscale=0.01)
        )
        
    def __call__(self, u, z, test=False):
        batchsize = u.data.shape[0]
        dim = u.data.shape[1]
        
        uz = F.concat((u*z, u, z), axis=1)
        uz = F.reshape(F.swapaxes(F.reshape(uz, (batchsize, 3, dim)), 1, 2), (batchsize*dim, 3))
        h = F.leaky_relu(self.l0(uz), slope=0.1)
        h = F.leaky_relu(self.l1(h), slope=0.1)
        h = F.leaky_relu(self.l2(h), slope=0.1)
        y = F.reshape(self.l3(h), (batchsize, dim))
        return y

class VanillaComb(chainer.Chain):
    def __init__(self, n):
        super(VanillaComb, self).__init__(
            b0 = F.Parameter(xp.zeros((1, n)).astype(np.float32)),
            w0z = F.Parameter(xp.ones((1, n)).astype(np.float32)),
            w0u = F.Parameter(xp.zeros((1, n)).astype(np.float32)),
            w0zu = F.Parameter(xp.zeros((1, n)).astype(np.float32)),
            ws = F.Parameter(xp.ones((1, n)).astype(np.float32)),
            b1 = F.Parameter(xp.zeros((1, n)).astype(np.float32)),
            w1z = F.Parameter(xp.ones((1, n)).astype(np.float32)),
            w1u = F.Parameter(xp.zeros((1, n)).astype(np.float32)),
            w1zu = F.Parameter(xp.zeros((1, n)).astype(np.float32)),
            )
    def __call__(self, u, z):
        batchsize = u.data.shape[0]
        dim = u.data.shape[1]
        
        b0 = F.broadcast(self.b0.W, u)[0]
        w0z = F.broadcast(self.w0z.W, u)[0]
        w0u = F.broadcast(self.w0u.W, u)[0]
        w0zu = F.broadcast(self.w0zu.W, u)[0]
        ws = F.broadcast(self.ws.W, u)[0]
        b1 = F.broadcast(self.b1.W, u)[0]
        w1z = F.broadcast(self.w1z.W, u)[0]
        w1u = F.broadcast(self.w1u.W, u)[0]
        w1zu = F.broadcast(self.w1zu.W, u)[0]
        
        return b0 + w0z*z + w0u*u + w0zu*z*u + ws*F.sigmoid(b1 + w1z*z + w1u*u + w1zu*z*u)
        


class Ladder_backward(chainer.Chain):
    def __init__(self, n_in, n_out, wscale, nolin=False):
        super(Ladder_backward, self).__init__(
            lin = F.Linear(n_in, n_out, wscale=wscale, nobias=True),
            comb = VanillaComb(n_out)
        )
        self.nolin = nolin
        
    def __call__(self, x, z, test=False):
        if self.nolin:
            h = x
        else:
            h = self.lin(x)
        mu = F.sum(h, axis=0)/h.data.shape[0]
        vr = (F.sum(h*h, axis=0)/h.data.shape[0])**0.5
        self.mu = F.broadcast(F.reshape(mu, (1,h.data.shape[1])),h)[0]
        self.vr = F.broadcast(F.reshape(vr, (1,h.data.shape[1])),h)[0]
        bnh = (h-self.mu)/(self.vr+1e-10)
        return self.comb(bnh, z)


class Ladder_forward(chainer.Chain):
    def __init__(self, n_in, n_out, wscale=1.0, act=F.relu):
        super(Ladder_forward, self).__init__(
            lin = F.Linear(n_in, n_out, wscale=wscale, nobias=True),
            gamma = F.Parameter(xp.ones((1,n_out)).astype(np.float32)),
            beta = F.Parameter(xp.zeros((1,n_out)).astype(np.float32)),
        )
        self.n_in, self.n_out = n_in, n_out
        self.act = act
        
    def __call__(self, x, eta, test=False):
        h = self.lin(x)
        mu = F.sum(h, axis=0)/h.data.shape[0]
        vr = (F.sum(h*h, axis=0)/h.data.shape[0])**0.5
        self.mu = F.broadcast(F.reshape(mu, (1,h.data.shape[1])),h)[0]
        self.vr = F.broadcast(F.reshape(vr, (1,h.data.shape[1])),h)[0]
        bnh = (h-self.mu)/(self.vr+1e-10)
        z = bnh + xp.random.randn(x.data.shape[0], self.n_out)*eta
        if self.act is None:
            return z, F.broadcast(self.gamma.W, z)[0]*(z + F.broadcast(self.beta.W, z)[0])
        else:
            return z, self.act(F.broadcast(self.gamma.W, z)[0]*(z + F.broadcast(self.beta.W, z)[0]))


class Encoder(chainer.Chain):
    def __init__(self):
        #[784,1000,500,250,250,250,10]
        super(Encoder, self).__init__(
                            bn1=Ladder_forward(784, 1000, wscale=1),
                            bn2=Ladder_forward(1000, 500, wscale=1),
                            bn3=Ladder_forward(500, 250, wscale=1),
                            bn4=Ladder_forward(250, 250, wscale=1),
                            bn5=Ladder_forward(250, 250, wscale=1),
                            bn6=Ladder_forward(250, 10, wscale=1, act=None),
        )
    def __call__(self, x, eta, test=False):
        x_ = x + xp.random.randn(x.data.shape[0], x.data.shape[1])*eta
        
        z1, h1 = self.bn1(x_, eta, test=test)
        z2, h2 = self.bn2(h1, eta, test=test)
        z3, h3 = self.bn3(h2, eta, test=test)
        z4, h4 = self.bn4(h3, eta, test=test)
        z5, h5 = self.bn5(h4, eta, test=test)
        z6, y = self.bn6(h5, eta, test=test)
        
        return y, [x_,z1,z2,z3,z4,z5,z6]


class Decoder(chainer.Chain):
    def __init__(self):
        #[784,1000,500,250,250,250,10]
        super(Decoder, self).__init__(
            bntop = Ladder_backward(10,10, wscale=1, nolin=True),
            bn0 = Ladder_backward(10,250, wscale=1),
            bn1 = Ladder_backward(250,250, wscale=1),
            bn2 = Ladder_backward(250,250, wscale=1),
            bn3 = Ladder_backward(250,500, wscale=1),
            bn4 = Ladder_backward(500,1000, wscale=1),
            bn5 = Ladder_backward(1000,784, wscale=1),
        )
    def __call__(self, y, zs, test=False):
        z_6 = self.bntop(y, zs[6], test=test)
        z_5 = self.bn0(z_6, zs[5], test=test)
        z_4 = self.bn1(z_5, zs[4], test=test)
        z_3 = self.bn2(z_4, zs[3], test=test)
        z_2 = self.bn3(z_3, zs[2], test=test)
        z_1 = self.bn4(z_2, zs[1], test=test)
        z_0 = self.bn5(z_1, zs[0], test=test)
        return [z_0,z_1,z_2,z_3,z_4,z_5,z_6]


def loss_labeled(x, t):
    y, zs = enc(x, eta=0.3, test=False)
    L = F.softmax_cross_entropy(y, t)
    '''
    lam = [1000, 10, 0.2, 0.2, 0.2, 0.2, 0.2]
    zs2 = dec(F.softmax(y), zs, test=False)
    y3, zs3 = enc(x, eta=0.0, test=False)
    mus = [enc.bn1.mu, enc.bn2.mu, enc.bn3.mu, enc.bn4.mu, enc.bn5.mu, enc.bn6.mu]
    vrs = [enc.bn1.vr, enc.bn2.vr, enc.bn3.vr, enc.bn4.vr, enc.bn5.vr, enc.bn6.vr]
    
    for i in range(len(zs2)):
        if i==0:
            L += lam[i] * F.mean_squared_error(zs2[i], zs3[i])
        else:
            L += lam[i] * F.mean_squared_error((zs2[i]-mus[i-1])/(vrs[i-1]), zs3[i])
    '''
    return L

def loss_unlabeled(x):
    lam = [1000, 10, 0.1, 0.1, 0.1, 0.1, 0.1]
    y, zs = enc(x, eta=0.3, test=False)
    zs2 = dec(F.softmax(y), zs, test=False)
    y3, zs3 = enc(x, eta=0.0, test=False)
    mus = [enc.bn1.mu, enc.bn2.mu, enc.bn3.mu, enc.bn4.mu, enc.bn5.mu, enc.bn6.mu]
    vrs = [enc.bn1.vr, enc.bn2.vr, enc.bn3.vr, enc.bn4.vr, enc.bn5.vr, enc.bn6.vr]
    
    L = 0
    for i in range(len(zs2)):
        if i==0:
            L += lam[i] * F.mean_squared_error(zs2[i], zs3[i])
        else:
            L += lam[i] * F.mean_squared_error((zs2[i]-mus[i-1])/(vrs[i-1]+1e-10), zs3[i])
    return L



def loss_test(x, t):
    y, zs = enc(x, eta=0.0, test=True)
    L, acc = F.softmax_cross_entropy(y, t), F.accuracy(y, t)
    return L, acc


enc = Encoder()
dec = Decoder()
enc.to_gpu()
dec.to_gpu()

o_enc = optimizers.Adam(alpha=0.002, beta1=0.9)
o_dec = optimizers.Adam(alpha=0.002, beta1=0.9)
alpha_plan = [0.002] * 100
for i in range(50):
    alpha_plan.append(0.002 * (50-i) / 50.0)

o_enc.setup(enc)
o_dec.setup(dec)

#o_enc.add_hook(chainer.optimizer.WeightDecay(0.00001), 'hook_dis1')
#o_dec.add_hook(chainer.optimizer.WeightDecay(0.00001), 'hook_gen')

batchsize_l = 100
batchsize_ul = 100


for epoch in range(len(alpha_plan)):
    print epoch

    sum_loss_l = 0
    sum_loss_ul = 0
    for it in range(60000/100):
        x,t = train_l.get(batchsize_l)
        loss_l = loss_labeled(Variable(x), Variable(t))

        o_enc.zero_grads()
        o_dec.zero_grads()
        loss_l.backward()
        o_enc.update()
        o_dec.update()

        x,_ = train_ul.get(batchsize_ul)
        loss_ul = loss_unlabeled(Variable(x))
        
        o_enc.zero_grads()
        o_dec.zero_grads()
        loss_ul.backward()
        o_enc.update()
        o_dec.update()

        sum_loss_l += loss_l.data
        sum_loss_ul += loss_ul.data
        
        loss_l.unchain_backward()
        loss_ul.unchain_backward()

    print "classification loss, reconstruction loss: ", sum_loss_l/600, sum_loss_ul/600
    o_enc.alpha = alpha_plan[epoch]
    o_dec.alpha = alpha_plan[epoch]
    
    x,t = test_set.get(10000, balance=False)
    L, acc = loss_test(Variable(x, volatile='on'), Variable(t, volatile='on'))
    #L.unchain_backward()
    #acc.unchain_backward()
    x,t = train_l.get(100)
    L_, acc_ = loss_test(Variable(x), Variable(t))
    L_.unchain_backward()
    acc_.unchain_backward()
    print "test error, test acc, train error, train acc: ", L.data, acc.data, L_.data, acc_.data
    sys.stdout.flush()
    if (epoch+1)%10==0:
        serializers.save_hdf5("enc3_%d.h5"%epoch, enc)
        serializers.save_hdf5("dec3_%d.h5"%epoch, dec)


