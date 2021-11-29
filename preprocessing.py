import pandas as pd
from pathlib import Path
import re
import nltk
from time import time
from textblob import TextBlob
# --------------------------------------
filename = 'D:/PycharmProjects/chatbot/tweet_emotions.csv'
df = pd.read_csv(Path(filename).resolve())


# ---------------------------------------------------
def preprocess(texts):
    start = time()
    # Lowercasing
    texts = texts.str.lower()

    # Remove special chars
    texts = texts.str.replace(r"(http|@)\S+", "")
    # texts = texts.apply(demojize)
    texts = texts.str.replace(r"::", ": :")
    texts = texts.str.replace(r"’", "'")
    texts = texts.str.replace(r"[^a-z\':_]", " ")

    # Remove repetitions
    pattern = re.compile(r"(.)\1{2,}", re.DOTALL)
    texts = texts.str.replace(pattern, r"\1")

    # Transform short negation form
    texts = texts.str.replace(r"(can't|cannot)", 'can not')
    texts = texts.str.replace(r"n't", ' not')

    # Remove stop words
    # stopwords = nltk.corpus.stopwords.words('english')
    # stopwords.remove('not')
    # stopwords.remove('nor')
    # stopwords.remove('no')
    # texts = texts.apply(
    #     lambda x: ' '.join([word for word in x.split() if word not in stopwords])
    # )
    texts = texts.apply(
        lambda x: ' '.join([word for word in x.split()])
    )
    return texts


# -----------------------------------------------------------------------
df ['cleaned'] = preprocess(df['content'])
df = df.drop(columns=['tweet_id', 'content'])
df=df.rename(columns={"sentiment": "label", "cleaned": "text"})
# train_data, validation_data = train_test_split(df, test_size=0.2)
# --------------------------- technique 1 ----------------------------------
df['label'] = df['label'].apply(
    lambda x: x if x in ['happiness', 'sadness', 'worry', 'neutral', 'love'] else "other")
# ----------------------saving preprocessed dataset---------------------------

df.to_csv('cleaned_1.csv', index=False)
