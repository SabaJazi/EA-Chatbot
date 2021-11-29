
from pathlib import Path
import pandas as pd
import pickle
import numpy as np
from tensorflow.keras.layers import Input, Embedding, SpatialDropout1D, LSTM
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Bidirectional, Conv1D, Dense, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
import random
orig = './output'
# print(os.getcwd())

# ------------------------load tokenizer------------------------
tokenizer_path=Path('D:/PycharmProjects/chatbot/tokenizer_2.pickle').resolve()
with tokenizer_path.open('rb') as file:
  tokenizer=pickle.load(file)

# ----------------------------------------------------------

input_dim = min(tokenizer.num_words, len(tokenizer.word_index) + 1)
# num_classes = len(df['label'].unique())
num_classes=5
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

model_weights_path=Path('D:/PycharmProjects/chatbot/model_with_weight.h5').resolve()
model.load_weights(model_weights_path.as_posix())
# model = keras.models.load_model('D:/PycharmProjects/chatbot/model_with_weight.h5')
# -------------------------loading test set------------------------------------
test_df = pd.read_csv('D:/PycharmProjects/chatbot/test.csv')
# ---------------------preprocess test befor applying the model----------------
sequences = [text.split() for text in test_df.Text]
list_tokenized= tokenizer.texts_to_sequences(sequences)
x_data= pad_sequences(list_tokenized, maxlen=input_length)
# ----------------------load the encoder----------------------------------------
encoder_path=Path('D:/PycharmProjects/chatbot/encoder_2.pickle').resolve()
with encoder_path.open('rb') as file:
  encoder=pickle.load(file)
# -------------------predict emotions of test set -------------------------------
def call_model(input):
    print(input)
    sequences = input.split()
    list_tokenized = tokenizer.texts_to_sequences(sequences)
    x_data = pad_sequences(list_tokenized, maxlen=input_length)
    y_pred = model.predict(x_data)
    #y_pred_argmax=y_pred.argmax(axis=1)
    #data_len = len(y_pred_argmax)
    # print(data_len)
    emotions=[]
    values = []
    max_pred=0
    for index, value in enumerate(np.sum(y_pred,axis=0)/len(y_pred)):
        #print(encoder.classes_[index] + ": " + str(len(y_pred_argmax[y_pred_argmax == value]) / data_len))
        print(encoder.classes_[index] + ": " + str(value))
        # if (len(y_pred_argmax[y_pred_argmax == value]) / data_len) > max_pred:
        #     max_pred =  (len(y_pred_argmax[y_pred_argmax == value]) / data_len)
        #     emotions.append([encoder.classes_[index],str(len(y_pred_argmax[y_pred_argmax == value]) / data_len)])
        emotions.append(encoder.classes_[index])
        values.append(value)
    # encoder.classes_[index]
    print('emotion:',emotions)
    index = values.index(max(values))
    print(emotions[index])
    return(emotions[index])
# ----------------------------------------------------------------
def fetch_answer(emotion):
    print('emotion:', emotion)
    if emotion == 'other':
        return "OK"
    path = orig + "/" + emotion + ".txt"
    with open(path, 'r') as f:
        res = (f.readlines())
        num = random.randint(0, len(res))

    f.close
    print(res[num])
    return (res[num])
