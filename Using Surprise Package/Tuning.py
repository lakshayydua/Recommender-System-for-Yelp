import pandas as pd
import numpy as np
from surprise import SVD,SVDpp,NMF
from surprise import Dataset, dataset
from surprise import evaluate, print_perf
import os
from os.path import dirname
from surprise import GridSearch

# Read the training set file
file1 = '/home/askapoor/Downloads/train_rating.txt'
train_df = pd.read_csv(file1)

# Prepare and configure reader to scikit-surprise.
reader1 = dataset.Reader(rating_scale=(1, 5))

# Read the data in the form customer, product, rating
data = Dataset.load_from_df(train_df[['user_id', 'business_id', 'rating']], reader1)

# Parameter Grid with 36 different combinations
param_grid = {'lr_all': [.003,0.006,.008],
              'n_epochs': [25,60],
              'reg_all': [0.08,.25],
              'n_factors':[15,25,100]}

grid_search = GridSearch(SVD, param_grid, measures=['RMSE', 'MAE', 'FCP'])
# grid_search1 = GridSearch(SVDpp, param_grid, measures=['RMSE', 'MAE', 'FCP'])
# grid_search2 = GridSearch(NMF, param_grid, measures=['RMSE', 'MAE', 'FCP'])

# Splitting the data in 3 folds
data.split(n_folds=3)

# Evaluate performances of our algorithm on the dataset
grid_search.evaluate(data)
# grid_search1.evaluate(data)
# grid_search2.evaluate(data)

print('SVD Score',grid_search.best_score['RMSE'])
print('SVD Params',grid_search.best_params['RMSE'])

# print('SVDpp Score',grid_search1.best_score['MAE'])
# print('SVDpp Params',grid_search1.best_params['MAE'])

# print('NMF Score',grid_search2.best_score['MAE'])
# print('NMF Params',grid_search2.best_params['MAE'])
