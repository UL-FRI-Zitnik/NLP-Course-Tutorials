#!/usr/bin/env python
# coding: utf-8

# # Muliple neural network analysis
# 
# <sup>This notebook is a part of Natural Language Processing class at the University of Ljubljana, Faculty for computer and information science. Please contact [slavko.zitnik@fri.uni-lj.si](mailto:slavko.zitnik@fri.uni-lj.si) for any comments.</sub>
# 
# ## IMDB sentiment analysis example
# 
# First we download the IMDB dataset. We present each word with a specific index from a vocabulary of 10000 words.

# In[1]:


import numpy as np
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.preprocessing.text import text_to_word_sequence

# number of distinct words
vocabulary_size = 10000

# number of words per review
max_review_length = 500

# load Keras IMDB movie reviews dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocabulary_size)

print(f'Number of reviews: {len(X_train)}.')
print(f'First review: \n\t{X_train[0]}.')
print(f'First label: {y_train[0]}.')
print(f'Length of first review before padding: {len(X_train[0])}.')

# padding reviews
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
print(f"\nAfter padding:")
print(f'First review: \n\t{X_train[0]}.')
print(f'Length of first review after padding: {len(X_train[0])}.')


# Mapping between real words and indexes:

# In[2]:


# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved, so we map the index for our use
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# Decode review text
def decode_review(text_ids, cls):
    text = ' '.join([reverse_word_index.get(i, '?') for i in text_ids if i not in [0,1,2,3]])
    label = 'POSITIVE' if cls == 1 else 'NEGATIVE'
    return f"\tText: {text}\n\tLabel: {label}"

# First review
print(f"First review: \n{decode_review(X_test[0], y_test[0])}")
# Last review
print(f"\nLast review: \n{decode_review(X_test[len(X_test)-1], y_test[len(X_test)-1])}")


# Below we create multiple models and evaluate their performance:
# 
# * **FFN**: Input to the models are word indices directly fed into a Dense layer.
# * **FFN with embeddings**: After creation of embedding vectors, the same architecture as in *FFN* is used.
# * **CNN**: Similar to the *FFN with embeddings* model with a convoluational and pooling layer immediatelly after embedding layer,
# * **RNN**: Simple RNN model with 100 hidden dimensions and prediction at the end.
# * **CNN+RNN**: A combination of *CNN* and *RNN* models above.
# 
# The runtime for the below script should take about a half hour using Tesla V100 32GB GPU.

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D, MaxPooling1D
import pandas as pd

EMBEDDING_DIM = 256

# Fully connected neural network
model_ffn = Sequential()
model_ffn.add(Dense(250, activation='relu',input_dim=max_review_length))
model_ffn.add(Dense(1, activation='sigmoid'))
model_ffn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history_ffn = model_ffn.fit(X_train, y_train, epochs=20, batch_size=128, verbose=2)
scores_ffn = model_ffn.evaluate(X_test, y_test, verbose=0)

# Fully connected neural network with embeddings
model_ffne = Sequential()
model_ffne.add(Embedding(vocabulary_size, EMBEDDING_DIM, input_length=max_review_length))
model_ffne.add(Flatten())
model_ffne.add(Dense(250, activation='relu'))
model_ffne.add(Dense(1, activation='sigmoid'))
model_ffne.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history_ffne = model_ffne.fit(X_train, y_train, epochs=20, batch_size=128, verbose=2)
scores_ffne = model_ffne.evaluate(X_test, y_test, verbose=0)

# Convolutional neural network
model_cnn = Sequential()
model_cnn.add(Embedding(vocabulary_size, EMBEDDING_DIM, input_length=max_review_length))
model_cnn.add(Conv1D(filters=200, kernel_size=3, padding='same', activation='relu'))
model_cnn.add(MaxPooling1D(pool_size=2))
model_cnn.add(Flatten())
model_cnn.add(Dense(250, activation='relu'))
model_cnn.add(Dense(1, activation='sigmoid'))
model_cnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history_cnn = model_cnn.fit(X_train, y_train, epochs=20, batch_size=128, verbose=2)
scores_cnn = model_cnn.evaluate(X_test, y_test, verbose=0)

# Recurrent Neural Network
model_rnn = Sequential()
model_rnn.add(Embedding(vocabulary_size, EMBEDDING_DIM, input_length=max_review_length))
model_rnn.add(SimpleRNN(100))
model_rnn.add(Dense(1, activation='sigmoid'))
model_rnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history_rnn = model_rnn.fit(X_train, y_train, epochs=3, batch_size=64)
scores_rnn = model_rnn.evaluate(X_test, y_test, verbose=0)

# Convolutional + Recurrent Neural Network
model_cnn_rnn = Sequential()
model_cnn_rnn.add(Embedding(vocabulary_size, EMBEDDING_DIM, input_length=max_review_length))
model_cnn_rnn.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model_cnn_rnn.add(MaxPooling1D(pool_size=2))
model_cnn_rnn.add(SimpleRNN(100))
model_cnn_rnn.add(Dense(1, activation='sigmoid'))
model_cnn_rnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history_cnn_rnn = model_cnn_rnn.fit(X_train, y_train, epochs=10, batch_size=64)
scores_cnn_rnn = model_cnn_rnn.evaluate(X_test, y_test, verbose=0)

# Evaluation
df = pd.DataFrame({'Model': ['FFNN', 'FFNN with Embeddings', 'CNN', 'RNN', 'CNN+RNN'],
                   'Accuracy': [str(round(scores_ffn[1]*100, 2)) + '%',
                                str(round(scores_ffne[1]*100, 2)) + '%',
                                str(round(scores_cnn[1]*100, 2)) + '%',
                                str(round(scores_rnn[1]*100, 2)) + '%',
                                str(round(scores_cnn_rnn[1]*100, 2)) + '%']})
print(df)


# The code above should output something similar to the following:
# 
# ```
#                             Model Accuracy
#           0                  FFNN   50.79%
#           1  FFNN with Embeddings   87.64%
#           2                   CNN   87.55%
#           3                   RNN   61.14%
#           4               CNN+RNN   82.62%
# ```
# 
# Surprisingly, the fully connected neural network with embeddings and convolutional neural network outperform the remaining networks. FFN is a very simple network with a single 250 dimensional dense layer.
# 
# CNNs can be robust to the position and orientation of learned objects in the image, while the same principle can be used on sequences, such as the one-dimensional sequence of words in a movie review. A simple RNN seems not to be very competitive but together with a CNN it achieves a decent performance.
# 
# Now, let's test the models with some custom examples:

# In[ ]:


def movie_sentiment(reviews, 
                    models=[model_ffn, model_ffne, model_cnn, model_rnn, model_cnn_rnn],
                    titles=['FFN', 'FFNE', 'CNN', 'RNN', 'CNN+RNN']):
    df = pd.DataFrame(columns=['review'] + titles)
    i = 0
    for review in reviews:
        words = set(text_to_word_sequence(review))
        words = [word_index[w] for w in words]
        words = sequence.pad_sequences([words], maxlen=max_review_length)
        df.loc[i] = [review] + titles
        df.loc[i]['review'] = review
        for j, model in enumerate(models):
            proba = model.predict(words)
            sentiment = '+' if proba>0.5 else '-'
            df.loc[i][titles[j]] = sentiment
        i = i + 1
    return df
    
text1 = 'I like it'
text2 = 'I dont like it'
text3 = 'After 30 min I still did not know what the movie is about'
text4 = 'It is so good that I will never ever watch it again. Boring experience.'
text5 = "It is like the Titanic movie!"
text6 = "That is the best movie I have ever seen."
text7 = "That is the worst movie I have ever seen."
print(movie_sentiment([text1, text2, text3, text4, text5, text6, text7]))


# The test above should output something similar to the following:
# 
# ```
#                                               review FFN FFNE CNN RNN CNN+RNN  | EXPECTED
# 0                                          I like it   -    +   +   +       -  |        + 
# 1                                     I dont like it   +    -   +   +       -  |        -
# 2  After 30 min I still did not know what the mov...   +    +   +   -       -  |        -
# 3  It is so good that I will never ever watch it ...   -    +   +   -       -  |        -
# 4                      It is like the Titanic movie!   +    +   +   +       -  |        +
# 5           That is the best movie I have ever seen.   +    +   +   +       -  |        +
# 6          That is the worst movie I have ever seen.   -    -   -   -       -  |        -
# -----------------------------------------------------------------------------------------
#                                  CORRECT PREDICTIONS   4    5   4   6       4  |        7
# ```
# 
# It seems that RNN performed as the best on these few short examples.
# 
# Try different architectures, models, hyperparameters (e.g. embedding dimensions), ... and you might improve the results.

# ## References
# 
# * The example above follows the post by Michel Kana, PhD: [Sentiment analysis: a practical benchmark](https://towardsdatascience.com/sentiment-analysis-a-benchmark-903279cab44a)
