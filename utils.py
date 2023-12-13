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


CAPTION_PATH = 'dataset/captions.txt'
IMAGE_PATH = 'dataset/images/'

model = InceptionV3()
model_new = Model(model.input, model.layers[-2].output)

def caption_preprocessing(text, remove_digits=True):
    pattern=r'[^a-zA-z0-9\s]'
    text=re.sub(pattern,'',text)
    text=text.split()
    text = [word.lower() for word in text]
    text = [word for word in text if word.isalpha()]
    text =  ' '.join(text)
    text = 'startseq ' + text + ' endseq'
    return text


def encode(image):

    img = np.resize(image, (299, 299, 3 ))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    fea_vec = model_new.predict(img)
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
    return fea_vec

def data_generator(captions, images, w2i, max_length, batch_size):

  X_image, X_cap, y = [], [], []
  n = 0
  while 1:
    for id, caps in captions.items():
      n += 1
      image = images[id]
      for cap in caps:
        seq = [w2i[word] for word in cap.split(' ') if word in w2i]

        for i in range(1, len(seq)):
          in_seq, out_seq = seq[:i], seq[i]
          in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
          out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
          X_image.append(image)
          X_cap.append(in_seq)
          y.append(out_seq)
      if n == batch_size:
        yield ([np.array(X_image), np.array(X_cap)], np.array(y))
        X_image, X_cap, y = [], [], []
        n = 0

df = pd.read_csv(CAPTION_PATH)
train, val = np.split(df.sample( frac=1,random_state=42), [int(.8*len(df)),])

df['caption'] = df['caption'].apply(caption_preprocessing)  

word_counts = {}
max_length = 0
for text in df['caption']:
    words = text.split()
    max_length = len(words) if (max_length < len(words)) else max_length
    for w in words:
        try:
            word_counts[w] +=1
        except:
            word_counts[w] = 1

word_count_threshold = 10
vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]

i2w = {}
w2i = {}
id = 1
for w in vocab:
    w2i[w] = id
    i2w[id] = w
    id += 1

embedding_dim = 200
vocab_size = len(vocab) + 1

# import pickle

# with open('w2i.pickle', 'wb') as handle:
#     pickle.dump(w2i, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('i2w.pickle', 'wb') as handle:
#     pickle.dump(i2w, handle, protocol=pickle.HIGHEST_PROTOCOL)


# captions = load(open("encoded_captions.pkl", "rb"))
# len(captions)

