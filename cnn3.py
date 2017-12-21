
# coding: utf-8

# In[1]:


from __future__ import print_function
import os

import pandas as pd
from PIL import Image
import numpy as np

import chainer
from chainer.dataset import convert
import chainer.links as L
import chainer.functions as F
from chainer import serializers

from chainer.datasets import get_cifar10
from chainer.datasets import get_cifar100

import utils


# ### ハイパーパラメータ

# In[2]:


from easydict import EasyDict
args = EasyDict({
    'bs': 64, 
    'epoch' : 100,
    'lr' : 0.05,
    'gpu': 0,
    'out': 'result',
    'resume': '',
    'n_in': 32, 
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

# In[3]:


dataset_dir = 'flower_images'
df  = pd.DataFrame.from_csv(os.path.join(dataset_dir,'flower_labels.csv'))

df['name'] = df.index
df['path'] = dataset_dir + '/' + df['name']

n_label = df.label.drop_duplicates().count()
n_label
df


# In[4]:


def load_fromdf(dataframe, resize=(96,96)):
    if type(resize) is int:
        resize = (resize, resize)
    
    df = dataframe
    x_data = []
    y_data = []
    for idx, row in df.iterrows():
        y = row['label']
        f = row['path']

        img = Image.open(f).resize(resize, Image.LANCZOS)
        img = img.convert('RGB')
        x = np.array(img)
        x_data.append(x)
        y_data.append(y)
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    return x_data, y_data


# In[5]:


df_train, df_test = utils.train_test_split_df(df, test_size=0.1)
x_train, y_train = load_fromdf(df_train, resize=args.n_in)
x_test, y_test = load_fromdf(df_test, resize=args.n_in)

x_train.shape


# ### モデルを定義

# In[6]:


class Block(chainer.Chain):
    def __init__(self, n_in=32, n_out=32, width=16, depth=1):
        super(Block, self).__init__()
        self.depth = depth
        ch = width
        with self.init_scope():
            self.bn1 = L.BatchNormalization(n_in)
            self.conv1 = L.Convolution2D(None, ch, 1, 1, 0)
            self.bn2 = L.BatchNormalization(ch)
            self.conv2 = L.Convolution2D(None, ch, 1, 1, 0)
            self.bn3 = L.BatchNormalization(ch)
            self.conv3 = L.Convolution2D(None, n_out, 1, 1, 0)
    
    def __call__(self, x):
        h = x
        h = self.bn1(h)
        h = F.relu(h)
        h = self.conv1(h)
        h = self.bn2(h)
        h = F.relu(h)
        h = self.conv2(h)
        h = self.bn3(h)
        h = F.relu(h)
        h = self.conv3(h)
        x = h
        return x    


# In[7]:


class LeNet(chainer.Chain):
    def __init__(self, n_in=32, n_out=10):
        super(LeNet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 16, 3)
            self.block1 = Block(16, 32, width=32, depth=2)
            self.block2 = Block(32, 64, width=32, depth=2)
            self.fc = L.Linear(None, n_out)
    
    def __call__(self, x):
        x = self.conv1(x)
        #x = F.dropout(x, 0.25)
        x = self.block1(x)
        x = F.max_pooling_2d(x, ksize=2, stride=2)
        #x = F.dropout(x, 0.25)
        x = self.block2(x)
        #x = F.dropout(x, 0.25)
        x = F.average_pooling_2d(x, ksize=2, stride=2)
        x = self.fc(x)
        return x    


# In[8]:


model = L.Classifier(LeNet(args.n_in, n_label),
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

# In[9]:


x_train.shape, y_train.shape, x_test.shape, y_test.shape

import Augmentor
p = Augmentor.Pipeline()
p.flip_left_right(probability=0.5)
p.flip_top_bottom(probability=0.5)
g = p.keras_generator_from_array(x_train, y_train, batch_size=args.bs)
g = ((
    xp.array(np.swapaxes((x/255.), 1, 3)).astype(np.float32),
    xp.array(y.astype(np.int8))
    ) for (x,y) in g)


# chainer.trainingを使わず，訓練ループをかく
# chainer.trainingでは，自前のデータのイテレータを使うことができないため．
# Augmentorを使いたい

# ### 訓練と検証

# In[10]:


def train(step=None):
    total_loss = 0
    total_acc = 0
    n_data = 0
    n_train = len(y_train)
    for _ in range(n_train//args.bs):
        xs, ts = next(g) 
        x = chainer.Variable(xs)
        t = chainer.Variable(ts)
        optimizer.update(model, x, t)
        with chainer.using_config('train', True):
            loss = model(x,t)
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


# In[11]:


if __name__ == '__main__':
    for step in range(args.epoch):
        print('step:{}'.format(step))
        train(step)
        test(step)

