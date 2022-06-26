total_epoch = 50
dataset_filename = "training_data_small.csv"
dictionary_filename = "dictionary-lstm-training.csv"
checkpoint_filepath="lstm-weights.{epoch:02d}-{acc:.3f}-{loss:.3f}.hdf5"
saved_model_name = "final_lstm_model.hd5"

# read file, put into dataset
import pandas as pd
dataset = pd.read_csv(dataset_filename)
x_train = dataset.iloc[:, 0].values;
y_train = dataset.iloc[:, 1].values;

df = pd.DataFrame(data=x_train,columns=['sentence']);


import numpy as np
df_split = np.array_split(df, 1000)

import string
import re

vocabSize = 0
dictionary = set()
maxNumOfWord = 0

# remove punctuation && double whitespace
for dataframe in df_split:
    for i in dataframe.index:
        translator = str.maketrans('', '', string.punctuation)
        dataframe.loc[i, "sentence"]= str.lower(re.sub(' +',' ', dataframe.loc[i,'sentence'].translate(translator))).split(); 
        sentence = dataframe.loc[i,'sentence']        
        numOfWord = len(sentence)
        
        for word in sentence:
           if word not in dictionary : 
                dictionary.add(word)
                vocabSize = vocabSize + 1
            
        if maxNumOfWord < numOfWord :
            maxNumOfWord = numOfWord

print(maxNumOfWord)
print(vocabSize)
#print(dictionary)         

import json
jsonData = {'lstm':[{'filename': dataset_filename, 'max_num_of_word': maxNumOfWord, 'vocab_size': vocabSize }]}

with open('jsondata.txt', 'w') as outfile:  
    json.dump(jsonData, outfile)

import csv

dictionary = list(dictionary)
model_dict = {}

i = 1
for word in dictionary:
        model_dict[word] = i
        i = i + 1           

with open(dictionary_filename,'w', newline='') as resultFile:
    w = csv.writer(resultFile)
    w.writerows(model_dict.items())

dfList = []
for dataframe in df_split:
    dfList.append(dataframe)
df = pd.concat(dfList, ignore_index=True)        
    
df.to_csv("sentence.csv", sep=',', encoding='utf-8')

encoded_sentence = []
for dataframe in df_split:
    for i in dataframe.index:
        sentence = dataframe.loc[i, "sentence"]
        idx = 0
        for word in sentence:
            sentence[idx] = model_dict[word]
            idx = idx + 1
        encoded_sentence.append(sentence)
        
# to make sure mapping on training and testing in dataset is same using hashing_trick    
with open("encoded_sentence.csv", "w", newline='') as resultFile:
    writer = csv.writer(resultFile, dialect='excel')
    writer.writerows(encoded_sentence)

from keras.preprocessing.sequence import pad_sequences
padded_sentences = pad_sequences(encoded_sentence, maxlen=maxNumOfWord, padding='pre')

# encode emotion to integer
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(y_train)
#print(list(label_encoder.inverse_transform([0,1,2,3,4,5,6])))
   


#It must specify 3 arguments:
#input_dim: This is the size of the vocabulary in the text data. For example, if your data is integer encoded to values between 0-10, then the size of the vocabulary would be 11 words.
#output_dim: This is the size of the vector space in which words will be embedded. It defines the size of the output vectors from this layer for each word. For example, it could be 32 or 100 or even larger. Test different values for your problem.
#input_length: This is the length of input sequences, as you would define for any input layer of a Keras model. For example, if all of your input documents are comprised of 1000 words, this would be 1000.

# create LSTM model	
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.embeddings import Embedding
from keras.layers import LSTM
from keras.optimizers import Adam

model = Sequential();
model.add(Embedding(vocabSize, 50, input_length = maxNumOfWord))
model.add(LSTM(50, dropout=0.2, recurrent_dropout=0.0))
model.add(Dense(7, activation='softmax'))
optimizer = Adam(lr=0.00001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

print(model.summary())

# checkpoint
from keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# fit the model
model.fit(padded_sentences, integer_encoded, epochs=total_epoch, batch_size= 100, callbacks=callbacks_list)

# evaluate the model
loss, accuracy = model.evaluate(padded_sentences, integer_encoded)
print('Accuracy: %f' % (accuracy*100))

# check weight
embeddings = model.layers[0].get_weights()[0]
lstm = model.layers[1].get_weights()[0];

embedding_7 = embeddings[7];

# save model
model.save(saved_model_name)