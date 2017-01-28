#!/bin/python3

import numpy as np
import pandas as pd

from sklearn.cross_validation import KFold

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional, Convolution1D, MaxPooling1D
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import f1_score

# Training
nfolds = 5
nb_epoch = 50
batch_size = 128

# conv
nb_filter = 64
filter_length = 5
pool_length = 4

# LSTM
lstm_timesteps = 5
lstm_input_dim = 50
lstm_output_size = 70

def nn_model():
   model = Sequential()
   model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='relu',               
                        input_shape=(lstm_timesteps, lstm_input_dim)))
   model.add(Dropout(0.3))

   model.add(Bidirectional(LSTM(lstm_output_size, activation='tanh', inner_activation='hard_sigmoid', return_sequences=False)))
   model.add(Dropout(0.5))
   
   model.add(Dense(8, activation='softmax', init = 'he_normal'))
   
   model.compile(loss = 'categorical_crossentropy', optimizer = 'adadelta', metrics=['categorical_accuracy'])
   return(model)

df = pd.read_csv('data/vectorized.txt', sep = ' ', header = 0)
X = df.iloc[:,1:].values

print('X shape:', X.shape)
# reshape into a 5x50x1 tensor to fit it into the conv/lstm
X = X.reshape(X.shape[0], -1, lstm_input_dim).astype('float32')
y = to_categorical(df.iloc[:,0])

print('X shape:', X.shape)
print('#samples: ', len(X))
print('#labels: ', len(y))

folds = KFold(len(y), n_folds = nfolds, shuffle = True)
currentFold = 0
foldScores = []
for (inTrain, inTest) in folds:
   xtr = X[inTrain]
   ytr = y[inTrain]
   xte = X[inTest]
   yte = y[inTest]
   
   print('Fold ', currentFold, ' starting...')
   model = nn_model()
   callbacks = [
      EarlyStopping(monitor='val_loss', patience = 3, verbose = 0),
      ModelCheckpoint(filepath=('models/{}_{}.hdf5'.format('model_weights', currentFold)), verbose=1, save_best_only = True)
   ]
      
   model.fit(xtr, ytr, batch_size=batch_size, nb_epoch=nb_epoch,
                     verbose=1, validation_data=(xte, yte),
                     callbacks=callbacks)
   ypred = model.predict(xte)
   # convert the probabilities back into a single integer class label
   ypred_max = ypred.argmax(axis=1)  
   yte_max = yte.argmax(axis=1)
            
   score = f1_score(yte_max, ypred_max, average = 'weighted')   
   print('Fold ', currentFold, '- F1: ', score)
   foldScores.append(score)
   currentFold += 1

print('avg fold scores: ', np.mean(foldScores))