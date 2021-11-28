import re
import nltk
from time import time
# from emoji import demojize
import os
import sys
from pathlib import Path
from pathlib import Path
import pandas as pd
# from nlp import Dataset
import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Embedding, SpatialDropout1D, LSTM
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Bidirectional, Conv1D, Dense, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelBinarizer

# print(os.getcwd())

# ------------------------load tokenizer------------------------
tokenizer_path=Path('D:/PycharmProjects/chatbot/tokenizer.pickle').resolve()
with tokenizer_path.open('rb') as file:
  tokenizer=pickle.load(file)

# ----------------------------------------------------------

input_dim = min(tokenizer.num_words, len(tokenizer.word_index) + 1)
# num_classes = len(df['label'].unique())
num_classes=13
embedding_dim = 500
input_length = 100
lstm_units = 128
lstm_dropout = 0.1
recurrent_dropout = 0.1
spatial_dropout=0.2
filters=64
kernel_size=3
# ------------------------
input_layer = Input(shape=(input_length,))
output_layer = Embedding(
  input_dim=input_dim,
  output_dim=embedding_dim,
  input_shape=(input_length,)
)(input_layer)

output_layer = SpatialDropout1D(spatial_dropout)(output_layer)

output_layer = Bidirectional(
LSTM(lstm_units, return_sequences=True,
     dropout=lstm_dropout, recurrent_dropout=recurrent_dropout)
)(output_layer)
output_layer = Conv1D(filters, kernel_size=kernel_size, padding='valid',
                    kernel_initializer='glorot_uniform')(output_layer)

avg_pool = GlobalAveragePooling1D()(output_layer)
max_pool = GlobalMaxPooling1D()(output_layer)
output_layer = concatenate([avg_pool, max_pool])

output_layer = Dense(num_classes, activation='softmax')(output_layer)

model = Model(input_layer, output_layer)
# ---------------------------------------------------------

model_weights_path=Path('D:/PycharmProjects/chatbot/model_2epc_13cl.h5').resolve()
model.load_weights(model_weights_path.as_posix())

# -------------------------loading test set------------------------------------
test_df = pd.read_csv('D:/PycharmProjects/chatbot/test.csv')
# ---------------------preprocess test befor applying the model----------------
sequences = [text.split() for text in test_df.Text]
list_tokenized= tokenizer.texts_to_sequences(sequences)
x_data= pad_sequences(list_tokenized, maxlen=input_length)
# ----------------------load the encoder----------------------------------------
encoder_path=Path('D:/PycharmProjects/chatbot/encoder.pickle').resolve()
with encoder_path.open('rb') as file:
  encoder=pickle.load(file)
# -------------------predict emotions of test set -------------------------------
y_pred=model.predict(x_data)
# for index, value in enumerate(np.sum(y_pred, axis=0)/len(y_pred)):
#   print(encoder.classes_[index]+": "+ str(value))

y_pred_argmax = y_pred.argmax(axis=1)
data_len = len(y_pred_argmax)
for index, value in enumerate(np.unique(y_pred_argmax)):
  print(encoder.classes_[index] + ": " + str(len(y_pred_argmax[y_pred_argmax == value]) / data_len))