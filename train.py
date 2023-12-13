import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from time import time
from numpy import array
import re
from pickle import dump, load
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model, Sequential
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector,\
                         Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization, add
from keras.optimizers import Adam, RMSprop
from keras import Input
from keras.callbacks import ModelCheckpoint
from utils import max_length, vocab_size, embedding_dim, embedding_matrix, train_features, data_generator, captions, w2i, df, encode, IMAGE_PATH


embedding_matrix = load(open("embedding_matrix.pkl", "rb"))
model = InceptionV3()

model_new = Model(model.input, model.layers[-2].output)

images = {}
captions = {}
for i in range(len(df)):
    images[df['image'][i]] = np.array(Image.open(IMAGE_PATH + df['image'][i]))
    try:
        captions[df['image'][i]].append(df['caption'][i])
    except:
        captions[df['image'][i]] = [df['caption'][i]]

with open("encoded_captions.pkl", "wb") as file:
    dump(captions, file)

encoding_image = {}
for id, img in images.items():
    encoding_image[id] = encode(img)

train_features = encoding_image
with open("encoded_train_images.pkl", "wb") as file:
    dump(encoding_image, file)

# train_features = load(open("encoded_train_images.pkl", "rb"))

model = "a"

inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)


inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)

decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)

model.layers[2].set_weights([embedding_matrix])
model.layers[2].trainable = False


model.compile(loss='categorical_crossentropy', optimizer='adam')

model.optimizer.lr = 0.0001
epochs = 5
batch_size = 16
steps = len(train_features)

checkpoint_path = "checkpoint/cp.ckpt"
cp_callback = ModelCheckpoint(filepath=checkpoint_path,save_best_only=False, save_weights_only=True, verbose=1)

generator = data_generator(captions=captions, images=train_features, w2i=w2i, max_length=max_length, batch_size=batch_size)
model.fit(generator, epochs=epochs, steps_per_epoch=steps, verbose=1, callbacks=[cp_callback])

model.save_weights('model/model.h5')
model.save('model/model.h5')

