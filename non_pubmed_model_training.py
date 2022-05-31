import pandas as pd
from sentence_transformers import SentenceTransformer
import preprocessor as p
from random import shuffle
from pickle import dump
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
import tensorflow as tf
import numpy as np

tf.random.set_seed(42)

model = SentenceTransformer('all-MiniLM-L6-v2')
df_train = pd.read_csv('data/training_testing_data.csv', encoding='utf-8')


def shuffle_data(x_train, y_train):
    indices = list(range(x_train.shape[0]))
    shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]
    return x_train, y_train
    
def create_model():
    model_medfact = Sequential()
    model_medfact.add(Dense(500, activation='relu'))
    model_medfact.add(Dropout(0.5))

    model_medfact.add(Dense(1, activation='sigmoid'))
    
    return model_medfact

def embed_text(x_train, y_train, text_list, label_list=[]):
    p.set_options(p.OPT.URL)
    clean_text = [p.clean(str(i)) for i in text_list]
    x_train = model.encode(clean_text)
    y_train = np.array(label_list)
    return x_train, y_train

    

x_train = []
y_train = []

x_train, y_train = embed_text(x_train, y_train, list(df_train.desc), list(df_train.label))

x_train, y_train = shuffle_data(x_train, y_train)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=50)

model_medfact = create_model()

BATCH_SIZE = 32
model_medfact.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, clipvalue=1.0), loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), metrics=['acc'])
model_medfact.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=20, verbose=1, validation_data=(x_val, y_val))

model_medfact.save_weights('models/general_model')

