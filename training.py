from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Input, Embedding, SpatialDropout1D, LSTM
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Bidirectional, Conv1D, Dense, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelBinarizer

# --------------------------------------
# filename = 'D:/PycharmProjects/chatbot/cleaned_1.csv'
# df = pd.read_csv(Path(filename).resolve())
df=pd.read_csv("D:/PycharmProjects/chatbot/cleaned_1.csv")
df['text']=df['text'].astype(str)

# df.head()
# exit()
train_data, validation_data = train_test_split(df, test_size=0.2)
# ------------------------load/create tokenizer------------------------
num_words = 10000

tokenizer = Tokenizer(num_words=num_words, lower=True)
tokenizer.fit_on_texts(df['text'])

file_to_save = Path('D:/PycharmProjects/chatbot/tokenizer_1.pickle').resolve()
with file_to_save.open('wb') as file:
    pickle.dump(tokenizer, file)

# ---------------------------------------------
input_dim = min(tokenizer.num_words, len(tokenizer.word_index) + 1)
num_classes = len(df['label'].unique())
embedding_dim = 500
input_length = 100
lstm_units = 128
lstm_dropout = 0.1
recurrent_dropout = 0.1
spatial_dropout = 0.2
filters = 64
kernel_size = 3
# -------------------------------------------------------
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
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
# --------------------------------------------------------
train_sequences = [text.split() for text in train_data.text]
validation_sequences = [text.split() for text in validation_data.text]
list_tokenized_train = tokenizer.texts_to_sequences(train_sequences)
list_tokenized_validation = tokenizer.texts_to_sequences(validation_sequences)
x_train = pad_sequences(list_tokenized_train, maxlen=input_length)
x_validation = pad_sequences(list_tokenized_validation, maxlen=input_length)

encoder = LabelBinarizer()
encoder.fit(df.label.unique())

encoder_path = Path('D:/PycharmProjects/chatbot/', 'encoder_1.pickle')
with encoder_path.open('wb') as file:
    pickle.dump(encoder, file)
# -------------------------------------------------------------------
y_train = encoder.transform(train_data.label)
y_validation = encoder.transform(validation_data.label)
# ------------------------------------
batch_size = 128
epochs = 2

history=model.fit(
    x_train,
    y=y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_validation, y_validation)
)

model_file = Path('D:/PycharmProjects/chatbot/model_1.h5').resolve()
model.save_weights(model_file.as_posix())

# with open('D:/PycharmProjects/chatbot/history_1.h5', 'wb') as file_pi:
#   pickle.dump(history, file_pi)

history = pd.DataFrame(history.history)
hist_csv_file = 'history_1.csv'
with open(hist_csv_file, mode='w') as f:
    history.to_csv(f)