import pandas as pd
from surprise import SVD
from surprise import dataset
from surprise import Dataset
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

# Read the data in the form of customer, product, rating
data = Dataset.load_from_df(train_df[['user_id', 'business_id', 'rating']], reader1)
data_test = Dataset.load_from_df(test_df[['user_id', 'business_id', 'rating']], reader1)

#Build train set and test set
trainset = data.build_full_trainset()
testset = data_test.build_full_trainset()
testset2 = testset.build_testset()

# Set the parameters values for the model
algo = SVD(lr_all = 0.0035, n_epochs = 60, reg_all = 0.15, n_factors = 15)

# Train the model
print("step 0")
algo.train(trainset)

# Use the model to predict ratings
print("step 1")
pred = algo.test(testset2,verbose=False)

print(pred[0])
print("finish")

data1 = ({'user_id': p.uid, 'business_id':p.iid,'rating': p.est} for p in pred)
data = pd.DataFrame(i for i in data1)

test_df.drop(['rating'],axis=1,inplace=True)
final = test_df.merge(data,on=['user_id','business_id'])

# Write the final result in a CSV file
final = final[['test_id','rating']]
final.to_csv('/home/ldua/DM/final1.csv', index=False)