total_epoch = 50
remaining_epoch = 9
init_epoch = total_epoch - remaining_epoch
dataset_filename = "training_data_new.csv"
latest_model = "lstm-weights.41-0.999-0.004.hdf5"
checkpoint_filepath = "lstm-weights.{epoch:02d}-{acc:.3f}-{loss:.3f}.hdf5"

import json
with open('jsondata.txt') as json_file:  
    data = json.load(json_file)
    for p in data['lstm']:
        vocabSize = p['vocab_size']
        maxNumOfWord = p['max_num_of_word']

print(vocabSize)
print(maxNumOfWord)

import pandas as pd
dataset = pd.read_csv(dataset_filename)
y_train = dataset.iloc[:, 1].values;

encoded_sentences = []

with open("encoded_sentence.csv") as resultFile:
    encoded_sentences =[line.split(',') for line in resultFile]


for i in range(len(encoded_sentences)):
    encoded_sentences[i] = list(map(int, encoded_sentences[i]))
    

from keras.preprocessing.sequence import pad_sequences
padded_sentences = pad_sequences(encoded_sentences, maxlen=maxNumOfWord, padding='pre')

# encode emotion to integer
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(y_train)
print(list(label_encoder.inverse_transform([0,1,2,3,4,5,6])))

# load the model that has been saved on checkpoint
# create LSTM model	
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

from keras.optimizers import Adam
new_model = load_model(latest_model)
optimizer = Adam(lr=0.00001)
new_model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

print(new_model.summary())

# checkpoint
checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# fit the model
new_model.fit(padded_sentences, integer_encoded, epochs=total_epoch, batch_size= 100, callbacks=callbacks_list, initial_epoch = init_epoch)

# evaluate the model
loss, accuracy = new_model.evaluate(padded_sentences, integer_encoded)
print('Accuracy: %f' % (accuracy*100))

# save model
new_model.save("final_lstm_model.hd5")