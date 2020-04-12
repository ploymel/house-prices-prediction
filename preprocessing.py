#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import skew
import warnings
warnings.filterwarnings('ignore')

folderPath = './'
dataPath = './data/'

df_train = pd.read_csv(dataPath + 'train.csv')
df_test = pd.read_csv(dataPath + 'test.csv')

# Skewness and kurtosis
print('Skewness: %f' % df_train['SalePrice'].skew())
print('Kurtosis: %f' % df_train['SalePrice'].kurt())

# Missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

# Dealing with missing data
df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)

df_test = df_test.drop((missing_data[missing_data['Total'] > 1]).index,1)

#deleting outlier points
df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]
df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)

# Save the length of training set
train_len = len(df_train)

# Combine 2 datasets together
df_all = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
df_all = df_all.drop('SalePrice', axis=1)

# Applying log transformation
df_train['SalePrice'] = np.log(df_train['SalePrice'])

# Data transformation
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])

# Create column for new variable (one is enough because it's a binary categorical feature)
# if area>0 it gets 1, for area==0 it gets 0
df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)
df_train['HasBsmt'] = 0 
df_train.loc[df_train['TotalBsmtSF']>0,'HasBsmt'] = 1

# Transform data
df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])

# Convert categorical variable into dummy
df_train_dummies = pd.get_dummies(df_train)

# Save data to csv
df_train_dummies[['Id', 'SalePrice']].to_csv(dataPath + 'preprocessed_train_y.csv', index=False)

# Log transform skewed numeric features:
numeric_feats = df_all.dtypes[df_all.dtypes != "object"].index

skewed_feats = df_train[numeric_feats].apply(lambda x: skew(x.dropna())) # Compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

df_all[skewed_feats] = np.log1p(df_all[skewed_feats])
df_all = pd.get_dummies(df_all)

# Filling NA's with the mean of the column:
df_all = df_all.fillna(df_all.mean())

# Save data to csv
df_all[:train_len].to_csv(dataPath + 'preprocessed_train_x.csv', index=False)
df_all[train_len:].to_csv(dataPath + 'preprocessed_test.csv', index=False)

# Thank you Pedro Marcelino for this amazing tutorial [COMPREHENSIVE DATA EXPLORATION WITH PYTHON](https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python#4.-Missing-data).
