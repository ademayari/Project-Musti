# Imports
# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler

#Classifiers
from sklearn.ensemble import RandomForestClassifier


# Common imports
import numpy as np
import pandas as pd
import os
import tarfile
import cv2
import pickle

# to make this notebook's output stable across runs
np.random.seed(42)
RELOAD_DATA = False # Change when you want to reload the dataframe

def target_value(val):
    if val == 'aanwezig':
        return 2
    if val == 'buiten':
        return 1
    return 0


if not os.path.isdir('./model/'):
    os.mkdir('./model/')

if not os.path.isfile('./model/dataframe.sav'):
    RELOAD_DATA = True # When there is not yet a dataframe, create one

if RELOAD_DATA: # Check whether there needs to be created a new dataframe
    #Extract when not already extracted
    if not os.path.isdir('./data/classificatie'):
        if not os.path.isfile('./data/classificatie.tar'):
            raise Exception('Classificatie.tar not fount')

        print('Extracting tar...')
        tar = tarfile.open('./data/classificatie.tar')
        tar.extractall('./data/')
        tar.close()
        print('Extracting tar Done!')

    if not os.path.isdir('./data/classificatie'):
        raise Exception('Extracted files not found')

    samples = []
    sample_counter = 0
    musti = pd.DataFrame()

    for folder in os.listdir('./data/classificatie/'):
        for file in os.listdir(f'./data/classificatie/{folder}'):
            
            img = cv2.imread(f'./data/classificatie/{folder}/{file}', 0)
            img = cv2.normalize(img,np.zeros(img.shape), 0, 1000, cv2.NORM_MINMAX)

            # add them to a dataframe
            imgd = dict()
            imgd['target'] = target_value(folder)
            c = 0
            for i in img.flatten():
                c += 1
                imgd[f'p{c}'] = i
            samples.append(imgd)
            sample_counter+=1
            print(file)

            if sample_counter % 200==0:
                temp_df = pd.DataFrame.from_dict(samples)
                musti = musti.append(temp_df, ignore_index=True)
                samples = []
    temp_df = pd.DataFrame.from_dict(samples)
    musti = musti.append(temp_df, ignore_index=True)
    samples = []

    # smallest_dataset_len = min(len(musti[musti.target == 2]), len(musti[musti.target == 1]), len(musti[musti.target == 0]))
    # for i in range(3):
    #     frac = 1 - smallest_dataset_len / len(musti[musti.target == i])
    #     print(frac, i)
    #     musti = musti.drop(musti.query(f'target == {i}').sample(frac=frac).index)

    print('Saving DataFrame')
    pickle.dump(musti, open('./model/dataframe.sav', 'wb'))
else:
    print('Loading DataFrame')
    musti = pickle.load(open('./model/dataframe.sav', 'rb'))

print('DataFrame Loaded')
print(musti)

X, y = musti.drop('target', axis=1), musti['target']
y = y.astype(np.uint8)  # less RAM space

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'Training set Shape: {X_train.shape}')

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train)


model = RandomForestClassifier()
print('fitting model')
#model = model.fit(X_train, y_train)

param_grid = [
    {'n_estimators': [10, 100, 200], 'max_features': [2, 4, 8], 'random_state': [42]},
    ]

gridsearch = GridSearchCV(model, param_grid, cv=3, verbose=2)
print(gridsearch)
gridsearch.fit(X_train, y_train)

print(gridsearch.best_params_,gridsearch.best_score_)

'''a = cross_val_score(model, X_test, y_test, cv=3)
print(f'\t{a}')
print(f'\tmean: {np.mean(a)}')
'''








