from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.applications.inception_v3 import preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
import numpy as np
import pickle
import os

model = InceptionV3()
model_new = Model(model.input, model.layers[-2].output)
max_length = 37
current_directory = os.path.dirname(os.path.abspath(__file__))
w2i_file_path = os.path.join(current_directory, 'w2i.pickle')
i2w_file_path = os.path.join(current_directory, 'i2w.pickle')
with open(w2i_file_path, 'rb') as handle:
    w2i = pickle.load(handle)
with open(i2w_file_path, 'rb') as handle:
    i2w = pickle.load(handle)

def encode(image):
    img = np.resize(image, (299, 299, 3 ))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    fea_vec = model_new.predict(img)
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
    return fea_vec

def greedySearch(photo):
    model_new = load_model('model/model.h5')
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [w2i[w] for w in in_text.split() if w in w2i]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model_new.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = i2w[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

def predict_image(image):
    img = np.array(image)
    encoded_image = encode(img).reshape((1,2048))
    predict = greedySearch(encoded_image)
    return predict

if __name__ == '__main__':
    test_image_path = 'test/78984436_ad96eaa802.jpg'
    print(predict_image(test_image_path))