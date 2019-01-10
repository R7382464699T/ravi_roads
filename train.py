import threading
import cv2
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from roads import roads_model

WIDTH = 224
HEIGHT = 224
BATCH_SIZE = 2


class ThreadSafeIterator:

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()

def threadsafe_generator(f):
    """
    A decorator that takes a generator function and makes it thread-safe.
    """

    def g(*args, **kwargs):
        return ThreadSafeIterator(f(*args, **kwargs))

    return g


@threadsafe_generator
def train_generator(df):
    while True:
        shuffle_indices = np.arange(len(df))
        shuffle_indices = np.random.permutation(shuffle_indices)
        
        for start in range(0, len(df), BATCH_SIZE):
            x_batch = []
            y_batch = []
            
            end = min(start + BATCH_SIZE, len(df))
            ids_train_batch = df.iloc[shuffle_indices[start:end]]
            
            for _id in ids_train_batch.values:
                img = cv2.imread('/home/asus/Documents/satellite/mass_roads/train/sat/{}.tiff'.format(_id))
                img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
                
                mask = cv2.imread('/home/asus/Documents/satellite/mass_roads/train/map/{}.tif'.format(_id), cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
                mask = np.expand_dims(mask, axis=-1)
                assert mask.ndim == 3
                
                # === You can add data augmentations here. === #
                if np.random.random() < 0.5:
                    img, mask = img[:, ::-1, :], mask[..., ::-1, :]  # random horizontal flip
                
                x_batch.append(img)
                y_batch.append(mask)
            
            x_batch = np.array(x_batch, np.float32) / 255.
            y_batch = np.array(y_batch, np.float32) / 255.
            
            yield x_batch, y_batch


@threadsafe_generator
def valid_generator(df):
    while True:
        for start in range(0, len(df), BATCH_SIZE):
            x_batch = []
            y_batch = []

            end = min(start + BATCH_SIZE, len(df))
            ids_train_batch = df.iloc[start:end]

            for _id in ids_train_batch.values:
                img = cv2.imread('/home/asus/Documents/satellite/mass_roads/valid/sat/{}.tiff'.format(_id))
                img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)

                mask = cv2.imread('/home/asus/Documents/satellite/mass_roads/valid/map/{}.tif'.format(_id),
                                  cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
                mask = np.expand_dims(mask, axis=-1)
                assert mask.ndim == 3
                
                x_batch.append(img)
                y_batch.append(mask)

            x_batch = np.array(x_batch, np.float32) / 255.
            y_batch = np.array(y_batch, np.float32) / 255.

            yield x_batch, y_batch

from os import walk
if __name__ == '__main__':   
    f1 = []
    for (dirpath, dirnames, filenames) in walk('/home/asus/Documents/satellite/mass_roads/train/sat'):
        f1.extend(filenames)
        break

    df1 = pd.DataFrame(f1)

    df_train = df1
    ids_train = pd.Series(df_train[0]).map(lambda s: s.split('.')[0])
    f2 = []
    for (dirpath, dirnames, filenames) in walk('/home/asus/Documents/satellite/mass_roads/valid/sat'):
        f2.extend(filenames)
        break

    df2 = pd.DataFrame(f2)
    ids_valid = df2
    ids_valid = pd.Series(ids_valid[0]).map(lambda s: s.split('.')[0])

    model = roads_model(
        input_shape=(224, 224, 3)
    )
    callbacks = [ModelCheckpoint(monitor='val_dice_coef',
                                 filepath='model_weights_roadssss.hdf5',
                                 save_best_only=True,
                                 mode='max'),
                                 ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.1,
                                   patience=3,
                                   verbose=1,
                                    min_delta=0.0001,
                                    cooldown=0,
                                    min_lr=0,
                                   mode='min'),]
    model.fit_generator(generator=train_generator(ids_train),
                        steps_per_epoch=np.ceil(float(len(ids_train)) / float(BATCH_SIZE)),
                        epochs=100,
                        verbose=1,
                        callbacks = callbacks,
                        validation_data=valid_generator(ids_valid),validation_steps=np.ceil(float(len(ids_valid)) / float(BATCH_SIZE)))
"""
    callbacks = [EarlyStopping(monitor='val_dice_coef',
                               patience=10,
                               verbose=1,
                               min_delta=1e-4,
                               mode='max'),
                 ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.2,
                                   patience=3,
                                   verbose=1,
                                    min_delta=0.0001,
                                    cooldown=0,
                                    min_lr=0,
                                   mode='max'),
                 ModelCheckpoint(monitor='val_dice_coef',
                                 filepath='model_weights.hdf5',
                                 save_best_only=True,
                                 mode='max')]
"""
    
