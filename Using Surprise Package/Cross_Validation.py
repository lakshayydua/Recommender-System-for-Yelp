import pandas as pd
from surprise import SVDpp,SVD
from surprise import dataset
from surprise import Dataset
from surprise import accuracy
import numpy as np
from surprise import evaluate, print_perf

# Read the training set
file1 = '/home/ldua/DM/train_rating.txt'
train_df = pd.read_csv(file1)

# Read the testing set
testfile = '/home/ldua/DM/test_rating.txt'
test_df = pd.read_csv(testfile)

print(len(test_df))
reader1 = dataset.Reader(rating_scale=(1, 5))

test_df['rating'] = 0

# Read the data in the form customer, product, rating
data = Dataset.load_from_df(train_df[['user_id', 'business_id', 'rating']], reader1)
data_test = Dataset.load_from_df(test_df[['user_id', 'business_id', 'rating']], reader1)

# Cross-Validation Folds k=5
data.split(n_folds=5)

# Parameter setting
algo = SVD(lr_all=0.003, n_epochs=60, reg_all=0.15, n_factors=15)

# Train and evaluate the model 'k' (5) times
for trainset, testset in data.folds():
    algo.train(trainset)
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions, verbose=True)