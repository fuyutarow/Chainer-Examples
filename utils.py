import os
import shutil
import pandas as pd

import numpy as np
from numpy.random import choice
from PIL import Image

from keras.utils import to_categorical

from datetime import datetime
from pytz import timezone


def now(isabout=False):
    if isabout:
        return datetime.now(timezone('UTC')).isoformat().split('.')[0]
    else:
        return datetime.now(timezone('UTC')).isoformat()


iso2utc = lambda iso: timezone('UTC').localize(pd.to_datetime(iso))

RESIZE = (96, 96)


def format_dirname(dirname):
    dirname += '/'
    return dirname.replace('//', '/')


def shuffle_df(df):
    df = df.reindex(np.random.permutation(df.index)).reset_index(drop=True)
    return df


def df_fromdir(data_dir, columns=['name', 'label']):
    fname_label = []

    labels = os.listdir(data_dir)
    for label in labels:
        for fname in os.listdir(os.path.join(data_dir, label)):
            d = (fname, label)
            fname_label.append(d)
    df = pd.DataFrame(fname_label, columns=columns)
    df['path'] = format_dirname(data_dir) + '/' + df['label'] + '/' + df['name']
    return df


def train_test_split_df(data_frame, test_size=0.1):
    from numpy.random import choice
    import numpy as np

    df = data_frame
    labels = set(df['label'])
    n = min(df['label'].value_counts())
    sampling_size = int(n * test_size)
    sampling_idx = np.array([
        choice(df[df.label == label].index, sampling_size, replace=False)
        for label in labels
    ]).flatten()
    test_df = df.loc[sampling_idx].reset_index(drop=True)
    train_df = df.drop(sampling_idx).reset_index(drop=True)
    return train_df, test_df


def data_fromdir(dir_name, label2id=None, resize=RESIZE):

    labels = set(os.listdir(dir_name))

    l2i = label2id if label2id else {
        label: i
        for i, label in enumerate(labels)
    }

    x_data = []
    y_data = []
    for label in labels:
        y = to_categorical(l2i[label], len(labels)).flatten()

        label_dname = os.path.join(dir_name, label)
        for fname in os.listdir(label_dname):
            f = os.path.join(label_dname, fname)
            x = Image.open(f).resize(resize, Image.LANCZOS)
            x = np.array(x)
            x_data.append(x)
            y_data.append(y)
    x_data = np.array(x_data).astype('f')
    y_data = np.array(y_data).astype('i')
    return x_data, y_data


def load_fromdf(dataframe, label2id=None, resize=RESIZE, rescale=1. / 255):

    df = dataframe
    labels = list(set(df['label']))
    l2i = label2id if label2id else {
        label: i
        for i, label in enumerate(labels)
    }

    x_data = []
    y_data = []
    for idx, row in df.iterrows():
        y = row['label']
        f = row['path']

        x = arr_fromf(f)
        x_data.append(x)
        y_data.append(y)
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    return x_data, y_data


def undersampling_df(data_frame, sampling_size=None):
    df = data_frame
    labels = set(df['label'])

    sampling_size =\
        sampling_size if sampling_size else\
        min(df['label'].value_counts())
    sampling_idx = np.array([
        choice(df[df.label == label].index, sampling_size, replace=False)
        for label in labels
    ]).flatten()
    sampling_data = df.loc[sampling_idx].reset_index()
    return sampling_data


def oversampling_df(data_frame, sampling_size=None):
    df = data_frame
    labels = set(df['label'])

    sampling_size =\
        sampling_size if sampling_size else\
        max(df['label'].value_counts())

    def sampling_idx_each_label(label):
        idx = df[df.label == label].index
        return \
            choice(idx, sampling_size, replace=False) if sampling_size < len(idx) else\
            np.concatenate([idx, choice(idx, sampling_size - len(idx))])

    sampling_idx =\
        np.array([sampling_idx_each_label(label)
                  for label in labels]).flatten()
    sampling_data = df.loc[sampling_idx].reset_index()
    return sampling_data


def arr2img(arr):
    return Image.fromarray(np.uint8(arr))


def arr_fromf(f, resize=RESIZE, rescale=1):
    resize = resize if type(resize) is tuple else (resize, resize)
    img = Image.open(f).resize(resize, Image.LANCZOS)
    if f.endswith('png') or f.endswith('PNG'):
        img = img.convert('RGB')
    return np.asarray(img) * rescale


#
#
#
#
#
#def train_test_split_fromdir(dataset_name, test_size=0.1):
#    def mkdir(dirname):
#        try:
#            os.mkdir(dirname)
#        except:
#            pass
#
#    flist = []
#    for label in os.listdir(os.path.join('dataset', dataset_name)):
#        flist += [
#            (f, label)
#            for f in os.listdir(os.path.join('dataset', dataset_name, label))
#        ]
#    df = pd.DataFrame(flist, columns=['fname', 'label'])
#    train_df, test_df = train_test_split_df(df)
#
#    read_dir = os.path.join('dataset', dataset_name)
#    train_dir = os.path.join('train_test_data', dataset_name, 'train')
#    test_dir = os.path.join('train_test_data', dataset_name, 'validation')
#
#    mkdir(os.path.join('train_test_data', dataset_name))
#    mkdir(train_dir)
#    mkdir(test_dir)
#
#    labels = os.listdir(read_dir)
#    labels
#
#    for label in labels:
#        mkdir(os.path.join(train_dir, label))
#        mkdir(os.path.join(test_dir, label))
#
#    for k, col in train_df.iterrows():
#        read_fname = os.path.join(read_dir, col['label'], col['fname'])
#        write_fname = os.path.join(train_dir, col['label'], col['fname'])
#        shutil.copy(read_fname, write_fname)
#
#    for k, col in test_df.iterrows():
#        read_fname = os.path.join(read_dir, col['label'], col['fname'])
#        write_fname = os.path.join(test_dir, col['label'], col['fname'])
#        shutil.copy(read_fname, write_fname)
#
#
#def load_brain_data_fromdir(dir_name):
#    import itertools
#
#    def strint_separator(f):
#        return [''.join(it) for _, it in itertools.groupby(f, str.isdigit)]
#
#    fname_label = []
#    # for removing .DS_Store
#    flist = [f for f in os.listdir(dir_name) if '.jpg' in f]
#    for fname in flist:
#        label = strint_separator(fname)[0]  # 'N','MS','PD' or'PS'
#        d = (fname, label)
#        fname_label.append(d)
#
#    df = pd.DataFrame(fname_label, columns=['fname', 'label'])
#    return df
#
#
#
#
#def undersampling_fromdir(dir_name, sampling_size=None):
#    from numpy.random import choice
#    x_data, y_data = load_data_fromdir(dir_name)
#
#    n_each_label = np.sum(y_data, axis=0)
#    n_label = y_data.shape[-1]
#    labels = set(range(n_label))
#
#    sampling_size =\
#        sampling_size if sampling_size else\
#        min(n_each_label)
#
#    def sampling_idx_each_label(label):
#        idx = np.argwhere(np.argmax(y_data, axis=1) == label).flatten()
#        return choice(idx, sampling_size, replace=False)
#
#    sampling_idx =\
#        np.array([sampling_idx_each_label(label) for label in labels]).flatten()
#    x_data = x_data[sampling_idx]
#    y_data = y_data[sampling_idx]
#    np.random.shuffle(x_data)
#    np.random.shuffle(y_data)
#    return x_data, y_data
#
#
#def oversampling_fromdir(dir_name, sampling_size=None):
#    from numpy.random import choice
#    x_data, y_data = load_data_fromdir(dir_name)
#
#    n_each_label = np.sum(y_data, axis=0)
#    n_label = y_data.shape[-1]
#    labels = set(range(n_label))
#
#    sampling_size =\
#        sampling_size if sampling_size else\
#        max(n_each_label)
#
#    def sampling_idx_each_label(label):
#        idx = np.argwhere(np.argmax(y_data, axis=1) == label).flatten()
#        return \
#            choice(idx, sampling_size, replace=False) if sampling_size < len(idx) else\
#            np.concatenate([idx, choice(idx, sampling_size - len(idx))])
#
#    sampling_idx =\
#        np.array([sampling_idx_each_label(label) for label in labels]).flatten()
#    x_data = x_data[sampling_idx]
#    y_data = y_data[sampling_idx]
#    np.random.shuffle(x_data)
#    np.random.shuffle(y_data)
#    return x_data, y_data
#
#
#def load_eyes_data_fromfile(file_name):
#    df = pd.read_csv(file_name, delimiter='\t', header=None).dropna()
#    fname_label = []
#    for row in df.iterrows():
#        fname = row[1][0] + '.JPG'
#        label = 'N' if row[1][2] == '["異常なし"]' else 'AB'
#        d = (fname, label)
#        fname_label.append(d)
#    df = pd.DataFrame(fname_label, columns=['fname', 'label'])
#    return df
#
#
#def to_categorical(y, num_classes=None):
#    categories = {}
#    categories = {s: i for (i, s) in enumerate(set(y))}
#    if type(y[0]) is str:
#        y = [categories[yy] for yy in y]
#        print(categories)
#
#    y = np.array(y, dtype='int').ravel()
#    if not num_classes:
#        num_classes = np.max(y) + 1
#
#    n = y.shape[0]
#    categorical = np.zeros((n, num_classes))
#    categorical[np.arange(n), y] = 1
#    return categorical, categories
#
#
#def load_from_dataframe(data_frame, dir_name=None, resize=(128, 128)):
#    from keras.preprocessing.image import ImageDataGenerator
#
#    def img2arr(f):
#        return np.asarray(Image.open(f).resize(resize, Image.LANCZOS))
#
#    df = data_frame
#    y = df['label']
#    y_data, categories = to_categorical(y)
#    X_data = []
#    for row in df.iterrows():
#        f = dir_name + row[1][1]
#        X_data.append(img2arr(f))
#    X_data = np.concatenate([X_data])
#
#    return X_data, y_data, categories
#
#
#def data_augmentation(x_data, y_data, batch_size=32, datagen=None, train=True):
#    from keras.preprocessing.image import ImageDataGenerator
#
#    if not datagen:
#        datagen = \
#            ImageDataGenerator(
#                rescale=1.0,
#                width_shift_range=0.2,
#                height_shift_range=0.2,
#                horizontal_flip=True,
#                # vertical_flip=True,
#                # featurewise_center=True,
#                # samplewise_center=True,
#                shear_range=0.2,
#                zoom_range=0.2,
#                # zca_whitening=True
#            ) if train else\
#            ImageDataGenerator(rescale=1.)
#
#    datagen.fit(x_data)
#    generator = datagen.flow(x_data, y_data, batch_size=batch_size)
#
#    x_data = []
#    y_data = []
#    for _ in range(generator.n):
#        (x, y) = generator.next()
#        x_data.append(x[0])
#        y_data.append(y[0])
#    x_data = np.array(x_data)
#    y_data = np.array(y_data)
#    return x_data, y_data
#
#
#def load_from_dir(data_dir, batch_size=32):
#    from keras.preprocessing.image import ImageDataGenerator
#
#    datagen = ImageDataGenerator(rescale=1.0 / 255)
#    generator = datagen.flow_from_directory(
#        data_dir,
#        target_size=(128, 128),
#        batch_size=batch_size,
#        class_mode='categorical')
#    x_data = []
#    y_data = []
#    for _ in range(generator.n):
#        (x, y) = generator.next()
#        x_data.append(x[0])
#        y_data.append(y[0])
#    x_data = np.array(x_data)
#    y_data = np.array(y_data)
#    return x_data, y_data
