#!/usr/bin/env python

import glob
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

SEED = 11

np.random.seed(SEED)

class Reader:

    TARGETS = 'is_doh'

    EXCLUDE = ['datasrc']

    path = None
    df = None
    columns = []
    features = []
    
    def __init__(self, path):
        self.path = path
        
    def read(self):
        all_files = glob.glob(self.path + "/*/*.csv")
        datasets = []
        for filename in all_files:
            df = pd.read_csv(filename, sep=';', index_col=None)
            datasets.append(df)

        if not datasets:
            print('No files found')
            return
        
        self.df = pd.concat(datasets, axis=0, ignore_index=True)

        for e in self.EXCLUDE:
            self.df.pop(e)
        
        self.columns = self.df.columns
        self.features = [c for c in self.columns if c != self.TARGETS]
        
        train, val = train_test_split(self.df, test_size=0.2, random_state=SEED)
                
        train_y = train.pop(self.TARGETS)
        val_y = val.pop(self.TARGETS)

        # print(train.head())
        
        # x = train.values #returns a numpy array
        # min_max_scaler = MinMaxScaler()
        # x_scaled = min_max_scaler.fit_transform(x)
        # train = pd.DataFrame(x_scaled)

        # x_scaled = min_max_scaler.transform(val)
        # val = pd.DataFrame(x_scaled)
        
        # #print(train.info())
        # print(train.head())

        #return
        
        return {'train': {'X': train, 'y': train_y}, 'val': {'X': val, 'y': val_y}}
        

def main():
    reader = Reader('data/extracted-features')
    reader.read()

if __name__ == '__main__':
    main()
    
