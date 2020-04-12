#!/usr/bin/env python
# coding: utf-8

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers, callbacks, optimizers
import pandas as pd
import re
import numpy as np
from sklearn.preprocessing import StandardScaler
import datetime
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

np.random.seed(1)
tf.keras.backend.clear_session() 

folderPath = './'
dataPath = './data/'

# Basic Model
def baseline_model():
    # create model
    model = models.Sequential()
    model.add(layers.Dense(1, input_dim=X_train.shape[1], activity_regularizer=regularizers.l1(0.001)))

    return model
def larger_model():
    # create model
    model = models.Sequential()
    model.add(layers.Dense(X_train.shape[1], input_dim=X_train.shape[1], activation='relu'))
    model.add(layers.Dense(X_train.shape[1]//2, activation='relu'))
    model.add(layers.Dense(1, activity_regularizer=regularizers.l1(0.001)))
 
    return model
def wider_model():
    # create model
    model = models.Sequential()
    model.add(layers.Dense(256, input_dim=X_train.shape[1], activation='relu'))
    model.add(layers.Dense(1, activity_regularizer=regularizers.l1(0.001)))

    return model
def more_larger_model():
    model = models.Sequential()
    model.add(layers.Dense(200, input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))
    model.add(layers.Dense(100, kernel_initializer='normal', activation='relu'))
    model.add(layers.Dense(50, kernel_initializer='normal', activation='relu'))
    model.add(layers.Dense(25, kernel_initializer='normal', activation='relu'))
    model.add(layers.Dense(1, kernel_initializer='normal', activity_regularizer=regularizers.l1(0.001)))

    return model

# Deep and Wide Model
def deep_and_wide_model():
    inp = layers.Input(shape=(X_train.shape[1],))
    
    # Deep
    hidden = layers.Dense(200, kernel_initializer='normal', activation='relu')(inp)
    hidden = layers.Dense(100, kernel_initializer='normal', activation='relu')(hidden)
    hidden = layers.Dense(50, kernel_initializer='normal', activation='relu')(hidden)
    hidden = layers.Dense(25, kernel_initializer='normal', activation='relu')(hidden)
    
    # Concate
    output = layers.concatenate([hidden, inp])
    output = layers.Dense(200, kernel_initializer='normal', activation='relu')(output)
    output = layers.Dense(100, kernel_initializer='normal', activation='relu')(output)
    output = layers.Dense(50, kernel_initializer='normal', activation='relu')(output)
    output = layers.Dense(25, kernel_initializer='normal', activation='relu')(output)
    output = layers.Dense(1, kernel_initializer='normal', activity_regularizer=regularizers.l1(0.001))(output)
    
    model = models.Model(inputs=inp, outputs=output)
    
    return model

# Complex Model
def balance_complex_model():
    input_related = layers.Input(shape=(X_related.shape[1],))
    input_effected = layers.Input(shape=(X_effected.shape[1],))
    
    # Related Side
    dense_re = layers.Dense(200, kernel_initializer='normal', activation='relu')(input_related)
    dense_re = layers.Dense(100, kernel_initializer='normal', activation='relu')(dense_re)

    # Effected Side
    dense_eff = layers.Dense(200, kernel_initializer='normal', activation='relu')(input_effected)
    dense_eff = layers.Dense(100, kernel_initializer='normal', activation='relu')(dense_eff)
    
    # Concate
    output = layers.concatenate([dense_re, dense_eff])
    output = layers.Dense(200, kernel_initializer='normal', activation='relu')(output)
    output = layers.Dense(100, kernel_initializer='normal', activation='relu')(output)
    output = layers.Dense(50, kernel_initializer='normal', activation='relu')(output)
    output = layers.Dense(25, kernel_initializer='normal', activation='relu')(output)
    output = layers.Dense(1, kernel_initializer='normal', activity_regularizer=regularizers.l1(0.001))(output)
    
    model = models.Model(inputs=[input_related, input_effected], outputs=output)
    
    return model
def larger_complex_model():
    input_related = layers.Input(shape=(X_related.shape[1],))
    input_effected = layers.Input(shape=(X_effected.shape[1],))
    
    # Related Side
    dense_re = layers.Dense(200, kernel_initializer='normal', activation='relu')(input_related)
    dense_re = layers.Dense(100, kernel_initializer='normal', activation='relu')(dense_re)
    dense_re = layers.Dense(200, kernel_initializer='normal', activation='relu')(dense_re)

    # Effected Side
    dense_eff = layers.Dense(200, kernel_initializer='normal', activation='relu')(input_effected)
    dense_eff = layers.Dense(200, kernel_initializer='normal', activation='relu')(dense_eff)
    
    # Concate
    output = layers.concatenate([dense_re, dense_eff])
    output = layers.Dense(400, kernel_initializer='normal', activation='relu')(output)
    output = layers.Dense(200, kernel_initializer='normal', activation='relu')(output)
    output = layers.Dense(100, kernel_initializer='normal', activation='relu')(output)
    output = layers.Dense(50, kernel_initializer='normal', activation='relu')(output)
    output = layers.Dense(25, kernel_initializer='normal', activation='relu')(output)
    output = layers.Dense(1, kernel_initializer='normal', activity_regularizer=regularizers.l1(0.001))(output)
    
    model = models.Model(inputs=[input_related, input_effected], outputs=output)
    
    return model
def single_side_complex_model():
    input_related = layers.Input(shape=(X_related.shape[1],))
    input_effected = layers.Input(shape=(X_effected.shape[1],))

    # Effected Side
    dense_eff = layers.Dense(200, kernel_initializer='normal', activation='relu')(input_effected)
    dense_eff = layers.Dense(200, kernel_initializer='normal', activation='relu')(dense_eff)
    
    # Concate
    output = layers.concatenate([input_related, dense_eff])
    output = layers.Dense(200, kernel_initializer='normal', activation='relu')(output)
    output = layers.Dense(100, kernel_initializer='normal', activation='relu')(output)
    output = layers.Dense(50, kernel_initializer='normal', activation='relu')(output)
    output = layers.Dense(25, kernel_initializer='normal', activation='relu')(output)
    output = layers.Dense(1, kernel_initializer='normal', activity_regularizer=regularizers.l1(0.001))(output)
    
    model = models.Model(inputs=[input_related, input_effected], outputs=output)
    
    return model

def save_to_file(result, filename='submission.csv'):
    row_id = 1461
    results = {'Id': [], 'SalePrice': []}
    for r in result:
        price = np.exp(r[0])
        results['Id'].append(row_id)
        results['SalePrice'].append(price)
        row_id += 1

    df = pd.DataFrame(data=results)
    df.to_csv(filename, index=False)

if __name__ == '__main__':

    print('Preparing Training data...')
    X_train = pd.read_csv(dataPath + 'preprocessed_train_x.csv')
    y_train = pd.read_csv(dataPath + 'preprocessed_train_y.csv')

    # Remove ID
    X_train = X_train.drop('Id', axis=1)
    y_train = y_train.drop('Id', axis=1)

    # Devided input into 2 groups
    X_related = X_train[['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']]
    X_effected = X_train.drop(['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt'], axis=1)

    X_train = StandardScaler().fit_transform(X_train)
    y_train = StandardScaler().fit_transform(y_train)
    X_related = StandardScaler().fit_transform(X_related)
    X_effected = StandardScaler().fit_transform(X_effected)

    print('Training model...')
    # Training Model

    model = larger_complex_model()
    adam = optimizers.Adam(lr=0.001)
    model.compile(optimizer=adam,
                loss=tf.keras.metrics.mean_squared_error,
                metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')])

    # Note: If the model is the basic model uncomment this 2 lines of codes
    # model.fit(x=X_train, 
    #           y=y_train, 
    #           epochs=100, batch_size=128)

    # Note: If the model is complex or deep and wide model uncomment this 2 lines of codes
    model.fit(x=[X_related, X_effected], 
            y=y_train, 
            epochs=100, batch_size=128)

    # Save model for further use
    model.save(folderPath + "model/model.h5")
    print("Saved model to disk")

    print('Preparing Testing data')
    X_test = pd.read_csv(dataPath + 'preprocessed_test.csv')
    y_train = pd.read_csv(dataPath + 'preprocessed_train_y.csv')
    # Remove ID
    X_test = X_test.drop('Id', axis=1)
    y_train = y_train.drop('Id', axis=1)
    # Devided input into 2 groups
    X_related_test = X_test[['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']]
    X_effected_test = X_test.drop(['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt'], axis=1)
    # Scale Data
    X_test = StandardScaler().fit_transform(X_test)
    X_related_test = StandardScaler().fit_transform(X_related_test)
    X_effected_test = StandardScaler().fit_transform(X_effected_test)
    scaler = StandardScaler().fit(y_train)

    print('Predicting...')
    result = model.predict([X_related_test, X_effected_test])

    # Inverse transformation
    result = scaler.inverse_transform(result)

    print('Saving Prediction...')
    save_to_file(result, filename=folderPath+'submission.csv')
    
    print('Done!!')