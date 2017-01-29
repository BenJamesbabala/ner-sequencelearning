#!/bin/python3

import yaml

import numpy as np
import pandas as pd

from sklearn.cross_validation import KFold

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional, Convolution1D, MaxPooling1D
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import f1_score, confusion_matrix

# training
nfolds = 5
nb_epoch = 50
batch_size = 512
nlabels = 8

# conv
nb_filter = 120
filter_length = 5

# LSTM
lstm_timesteps = 5
lstm_input_dim = 50
lstm_units = 150

cfg = yaml.load(open("data/meta.yaml", "r"))
if cfg['context']:
   lstm_timesteps = cfg['context']
if cfg['embedding_dim']:
   lstm_input_dim = cfg['embedding_dim']
if cfg['nlabels']:
   nlabels = cfg['nlabels']

print('lstm timesteps: {}, lstm input dim: {}, num output labels: {}'.format(lstm_timesteps, lstm_input_dim, nlabels))

def nn_model():
   model = Sequential()
   model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',                        
                        input_shape=(lstm_timesteps, lstm_input_dim)))
   model.add(PReLU())
   model.add(BatchNormalization())
   model.add(Dropout(0.3))

   model.add(Bidirectional(LSTM(lstm_units, activation='tanh', inner_activation='sigmoid', return_sequences=False)))
   model.add(Dropout(0.3))
   
   model.add(Dense(nlabels, activation='softmax', init = 'he_normal'))
   
   model.compile(loss = 'categorical_crossentropy', optimizer = 'adadelta', metrics=['categorical_accuracy'])
   return(model)

df = pd.read_csv('data/vectorized.txt', sep = ' ', header = 0)
X = df.iloc[:,1:].values

print('X shape: ', X.shape)
# reshape again into temporal structure
X = X.reshape(X.shape[0], -1, lstm_input_dim).astype('float32')
y = to_categorical(df.iloc[:,0])

print('X temporal reshape: ', X.shape)
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
      ModelCheckpoint(filepath=('models/model_fold_{}.hdf5'.format(currentFold)), verbose=0, save_best_only = True)
   ]
      
   model.fit(xtr, ytr, batch_size=batch_size, nb_epoch=nb_epoch,
                     verbose=1, validation_data=(xte, yte),
                     callbacks=callbacks)
   ypred = model.predict(xte)
   # convert the probabilities back into a single integer class label
   ypred_max = ypred.argmax(axis=1)  
   yte_max = yte.argmax(axis=1)
            
   score = f1_score(yte_max, ypred_max, average = 'weighted')   
   foldScores.append(score)
   print("Confusion matrix:\n%s" % confusion_matrix(yte_max, ypred_max))
   print('Fold ', currentFold, '- F1: ', score)   
   print('avg f1 fold scores so far: ', np.mean(foldScores))
   currentFold += 1

print('f1 fold scores: ', foldScores)
print('final avg f1 fold scores: ', np.mean(foldScores))