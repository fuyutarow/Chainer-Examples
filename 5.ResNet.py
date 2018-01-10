
# coding: utf-8

# In[4]:


from __future__ import print_function
import os

import pandas as pd
from PIL import Image
import numpy as np

import chainer
from chainer.dataset import convert
import chainer.links as L
import chainer.functions as F
import chainer.initializers as I
from chainer import serializers

import utils


# ### ハイパーパラメータ

# In[5]:


from easydict import EasyDict
args = EasyDict({
    'bs': 32, 
    'epoch' : 100,
    'lr' : 0.005,
    'gpu': 0,
    'out': 'result',
    'resume': '',
})
try:
    __file__.endswith('py')
    import argparse
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', dest='bs', type=int, default=args.bs,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=args.epoch,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--learningrate', '-l', dest='lr', type=float, default=args.lr,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=int, default=args.gpu,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default=args.out,
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default=args.resume,
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', dest='n_in', type=int, default=args.n_in,
                        help='Number of units')
    parser.add_argument('--noplot', dest='plot', action='store_false',
                        help='Disable PlotReport extension')
    args = parser.parse_args()
except:
    print('no argsparse')
    pass


# ### データセット読み込み

# In[6]:


from chainer.datasets import get_cifar10

data_train, data_test = get_cifar10()

x_train = []
y_train = []
for i in range(data_train.__len__()):
    x, y = data_train.__getitem__(i)
    x_train.append(x)
    y_train.append(y)
x_train = np.array(x_train)
y_train = np.array(y_train)

x_test = []
y_test = []
for i in range(data_test.__len__()):
    x, y = data_test.__getitem__(i)
    x_test.append(x)
    y_test.append(y)
x_test = np.array(x_test)
y_test = np.array(y_test)

x_train = np.swapaxes(x_train*255, 1, 3).astype(np.uint8)
x_test = np.swapaxes(x_test*255, 1, 3).astype(np.uint8)

x_train.shape, x_train.dtype, x_test.dtype


# ### モデルを定義

# In[7]:


class Block(chainer.Chain):
    def __init__(self, n_in=32, ch=16):
        super(Block, self).__init__()
        self.n_in = n_in
        self.ch = ch
        
        initialW= I.HeNormal()
        with self.init_scope():
            self.bn1 = L.BatchNormalization(n_in)
            self.conv1 = L.Convolution2D(None, ch, 3, 1, 1, initialW=initialW, nobias=True)
            self.bn2 = L.BatchNormalization(ch)
            self.conv2 = L.Convolution2D(None, ch, 3, 1, 1, initialW=initialW, nobias=True)
            self.bn3 = L.BatchNormalization(ch)
            
            self.conv = L.Convolution2D(None, ch, 1, 1, 0)
    
    def __call__(self, x):
        h = x_ = x
        h = self.bn1(h)
        h = self.conv1(h)
        h = self.bn2(h)
        h = F.relu(h)
        h = self.conv2(h)
        h = self.bn3(h)
        
        if self.n_in != self.ch:
            x_ = self.conv(x_)
        
        return h + x_


# In[8]:


class Section(chainer.ChainList):
    def __init__(self, n_in=32, ch=32, depth=1):
        super(Section, self).__init__()
        self.add_link(Block(n_in, ch))
        for i in range(1, depth):
            self.add_link(Block(ch, ch))
    
    def __call__(self, x):
        for f in self.children():
            x = f(x)
        return x    


# In[9]:


class CNN(chainer.Chain):
    def __init__(self, n_in=32, n_out=10):
        super(CNN, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 16, 3)
            self.section1 = Section(16, 16, 2)
            self.section2 = Section(16, 32, 2)
            self.section3 = Section(32, 64, 2)
            self.conv_out = L.Convolution2D(None, n_out, 3)
            self.fc = L.Linear(None, n_out)
    
    def __call__(self, x):
        x = self.conv1(x)
        x = self.section1(x)
        x = F.average_pooling_2d(x, ksize=2, stride=2)
        x = self.section2(x)
        #print('#2', x.shape)
        x = F.average_pooling_2d(x, ksize=2, stride=2)
        x = self.section3(x)
        x = self.conv_out(x)
        x = F.squeeze(F.average_pooling_2d(x, 5))
        #print('#2', x.shape)
        x = F.average_pooling_2d(x, ksize=2, stride=2)
        return x    


# In[10]:


class CNN(chainer.Chain):
    def __init__(self, n_in=32, n_out=10):
        super(CNN, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 16, 3)
            self.section1 = Section(16, 16, 2)
            self.section2 = Section(16, 32, 2)
            self.section3 = Section(32, 64, 2)
            self.fc = L.Linear(None, n_out)
    
    def __call__(self, x):
        x = self.conv1(x)
        x = self.section1(x)
        #print('#1', x.shape)
        x = F.average_pooling_2d(x, ksize=2, stride=2)
        x = self.section2(x)
        #print('#2', x.shape)
        x = F.average_pooling_2d(x, ksize=2, stride=2)
        x = self.fc(x)
        
        return x    


# In[11]:


n_label = np.unique(y_train).size
model = L.Classifier(CNN(n_label),
                    lossfun=F.softmax_cross_entropy,
                    accfun=F.accuracy)
xp = np
if args.gpu >= 0:
    import cupy as cp
    xp = cp
    chainer.cuda.get_device_from_id(args.gpu).use()
    model.to_gpu()  # Copy the model to the GPU
optimizer = chainer.optimizers.MomentumSGD(args.lr)
optimizer.setup(model)


# ### data augmentation

# In[12]:


x_train.shape, y_train.shape, x_test.shape, y_test.shape

import Augmentor
p = Augmentor.Pipeline()
#p.crop_random(probability=1, percentage_area=0.8)
#p.resize(probability=1, width=32, height=32)
#p.flip_left_right(probability=0.5)
#p.random_erasing(probability=0.5, rectangle_area=0.2)
#p.shear(probability=0.3, max_shear_left=2, max_shear_right=2)
#
g = p.keras_generator_from_array(x_train, y_train, batch_size=args.bs)
g = ((
    xp.array(np.swapaxes((x/255.), 1, 3)).astype(np.float32),
    xp.array(y.astype(np.int8))
    ) for (x,y) in g)


# chainer.trainingを使わず，訓練ループをかく
# chainer.trainingでは，自前のデータのイテレータを使うことができないため．
# Augmentorを使いたい

# ### 訓練と検証

# In[13]:


import time


# In[14]:


def train(step=None):
    total_loss = 0
    total_acc = 0
    n_data = 0
    n_train = len(y_train)
    
    total_t = 0  
    for _ in range(n_train//args.bs):
        xs, ts = next(g) 
        x = chainer.Variable(xs)
        t = chainer.Variable(ts)
        optimizer.update(model, x, t)
        with chainer.using_config('train', True):
            t1 = time.time()
            loss = model(x,t)
            print(time.time()-t1, '#######3')
            
        n_data += len(t.data)
        total_loss += float(loss.data) * len(t.data)
        total_acc += float(model.accuracy.data) * len(t.data)

    loss = total_loss / n_data
    acc = total_acc / n_data
    print('loss: {:.4f}\t acc: {:.4f}'.format(loss, acc))
    
def test(step=None):
    xs = xp.array(np.swapaxes((x_test), 1, 3)).astype(np.float32)
    ts = xp.array(y_test).astype(np.int8)
    x = chainer.Variable(xs)
    t = chainer.Variable(ts)
    loss = model(x,t)

    n_data = len(t.data)
    total_loss = float(loss.data) * len(t.data)
    total_acc = float(model.accuracy.data) * len(t.data)
    loss = total_loss / n_data
    acc = total_acc / n_data
    print('val_loss: {:.4f}\t val_acc: {:.4f}'.format(loss, acc))


# In[19]:


def train(step=None):
    xs, ts = next(g) 
    x = chainer.Variable(xs)
    t = chainer.Variable(ts)
    optimizer.update(model, x, t)
    with chainer.using_config('train', True):
        t1 = time.time()
        loss = model(x,t)

    loss = float(loss.data)
    acc = float(model.accuracy.data)

    if step%100==0:
        print('step:{}'.format(step))
        print('loss: {:.4f}\t acc: {:.4f}'.format(loss, acc))
    
def validation(step=None):
    xs = xp.array(np.swapaxes((x_test), 1, 3)).astype(np.float32)
    ts = xp.array(y_test).astype(np.int8)
    x = chainer.Variable(xs)
    t = chainer.Variable(ts)
    loss = model(x,t)

    loss = float(loss.data)
    acc = float(model.accuracy.data)
    
    print('val_loss: {:.4f}\t val_acc: {:.4f}'.format(loss, acc))


# In[20]:


if __name__ == '__main__':
    #for step in range(args.epoch):
    for step in range(10**7):
        train(step)
        if step%10000==0:
            test(step)

