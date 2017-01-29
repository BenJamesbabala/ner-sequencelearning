#!/bin/python3

import yaml
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
from keras.utils.np_utils import to_categorical
from keras.models import load_model

def predict_test_file(fname, lstm_input_dim, nlabels):
   print('loading data from file ', fname)
   df = pd.read_csv(fname, sep = ' ', header = 0)
   X = df.iloc[:,1:].values

   print('X shape: ', X.shape)
   # reshape again into temporal structure
   X = X.reshape(X.shape[0], -1, lstm_input_dim).astype('float32')
   y = to_categorical(df.iloc[:,0])

   print('X temporal reshape: ', X.shape)
   print('#samples: ', len(X))
   print('#labels: ', len(y))

   # we are averaging over all models and then just taking the max
   m_preds = np.zeros((X.shape[0], nlabels))
   for model in models:
      m_preds = m_preds + model.predict(X)

   m_preds = m_preds / len(models)
      
   ypred_max = m_preds.argmax(axis=1)
   yte_max = y.argmax(axis=1)
            
   score = f1_score(yte_max, ypred_max, average = 'weighted')   
   print("Confusion matrix:\n%s" % confusion_matrix(yte_max, ypred_max))
   print('F1 Score ', score)

# read all the keras models from the CV as an ensemble
models = []
for path in glob.glob('models/model*.hdf5'):
   print('loading ', path)
   models.append(load_model(path))
   
 
# TODO unify the duped code with train.py
lstm_input_dim = 50
nlabels = 8

cfg = yaml.load(open("data/meta.yaml", "r"))
if cfg['embedding_dim']:
   lstm_input_dim = cfg['embedding_dim']
if cfg['nlabels']:
   nlabels = cfg['nlabels']

predict_test_file('data_test_a/vectorized.txt', lstm_input_dim, nlabels)
predict_test_file('data_test_b/vectorized.txt', lstm_input_dim, nlabels)


