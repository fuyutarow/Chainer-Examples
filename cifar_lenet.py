
# coding: utf-8

# In[1]:


from __future__ import print_function
import argparse

import chainer
from chainer.dataset import convert
import chainer.links as L
import chainer.functions as F
from chainer import serializers

from chainer.datasets import get_cifar10
from chainer.datasets import get_cifar100

import models.VGG


# In[2]:


dataset = 'cifar10'
bs = 64
lr = 0.05
epochs = 10
gpu = -1
out = 'result'
resume = ''
n_label = 10


# In[3]:


class Fire(chainer.Chain):
    def __init__(self, n_in=None, n_out=32):
        super(Fire, self).__init__()
        with self.init_scope():
            self.bn = L.BatchNormalization(None)
            self.conv1 = L.Convolution2D(n_in, 32, 3)
            self.conv2 = L.Convolution2D(None, n_out, 3)
    
    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        return x    


# In[4]:


class LeNet(chainer.Chain):
    def __init__(self, n_out=10):
        super(LeNet, self).__init__()
        with self.init_scope():
            self.fire1 = Fire(None, 32)
            self.fire2 = Fire(None, 32)
            self.conv1 = L.Convolution2D(None, 32, 3)
            self.conv2 = L.Convolution2D(None, 10, 3)
            self.fc = L.Linear(None, n_out)
    
    def __call__(self, x):
        x = self.conv1(x)
        x = F.max_pooling_2d(x, ksize=2, stride=2)
        #x = self.fire1(x)
        #x = self.fire2(x)
        #x = F.max_pooling_2d(x, ksize=2, stride=2)
        x = self.conv2(x)
        x = F.average_pooling_2d(x, ksize=2, stride=2)
        x = self.fc(x)
        return x    


# In[5]:


class compute_loss(chainer.Chain):
    def __init__(self, model, lossfun=F.softmax_cross_entropy):
        super(LeNet, self).__init__()
        with self.init_scope():
            self.model = model
            self.lossfun = lossfun
    
    def __call__(self, x, t):
        y = self.forward(x)
        loss = self.lossfun(y, t)
        self.accuracy = F.accuracy(y, t)
        chainer.report({'loss': loss, 'accuracy': self.accuracy}, self)
        return loss


# In[6]:


model = L.Classifier(LeNet(n_label),
                    lossfun=F.softmax_cross_entropy,
                    accfun=F.accuracy)
optimizer = chainer.optimizers.MomentumSGD(lr)
optimizer.setup(model)


# In[7]:


train, test = get_cifar10()

train_iter = chainer.iterators.SerialIterator(train, bs)
test_iter = chainer.iterators.SerialIterator(test, bs, repeat=False, shuffle=False)


# In[8]:


batch = train_iter.next()
xs, ts = chainer.dataset.convert.concat_examples(batch, gpu)


# chainer.trainingを使わず，訓練ループをかく
# chainer.trainingでは，自前のデータのイテレータを使うことができないため．
# Augmentorを使いたい

# In[9]:


total_loss = 0
total_acc = 0
cnt = 0
for _ in range(bs):

    batch = train_iter.next()
    xs, ts = chainer.dataset.convert.concat_examples(batch, gpu)
    x = chainer.Variable(xs)
    t = chainer.Variable(ts)
    optimizer.update(model, x, t)
    loss = model(x,t)
    cnt += len(t.data)
    total_loss += float(loss.data) * len(t.data)
    total_acc += float(model.accuracy.data) * len(t.data)
    
print('loss: {}\t acc: {}'.format(total_loss/cnt, total_acc/cnt))


# In[10]:


total_loss = 0
total_acc = 0
cnt = 0
for _ in range(bs):

    batch = test_iter.next()
    xs, ts = chainer.dataset.convert.concat_examples(batch, gpu)
    x = chainer.Variable(xs)
    t = chainer.Variable(ts)
    optimizer.update(model, x, t)
    loss = model(x,t)
    cnt += len(t.data)
    total_loss += float(loss.data) * len(t.data)
    total_acc += float(model.accuracy.data) * len(t.data)
    
print('val_loss: {}\t val_acc: {}'.format(total_loss/cnt, total_acc/cnt))

