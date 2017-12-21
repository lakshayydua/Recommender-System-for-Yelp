import pandas as pd
import numpy as np
from scipy import sparse
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# ================= FUNCTIONS ==================
# predict initial bias
def predictBaseRating(user, business):
    return mu + bu[user] + bi[business]

#This function is used for predicting the rating for the next iteration in order to check if the factor update is required using Stochastic Gradient Descent Technique and also for predicting the rating for the test data.
def findNewRating(user, business):
    try:
        rat = np.dot(Qi[business], Pu[user])+bi[business]+bu[user]+mu
        if rat > 5:
            return 5
        elif rat < 1:
            return 1
        else:
            return rat
    except:
        return mu


# ===== calculates RMSE =====
def calculateRMSE():
    sumSqErr = 0
    count = 0
    for u, i, r in zip(urM.row, urM.col, urM.data):
        err = r - findNewRating(u, i)
        sumSqErr = sumSqErr + err * err
        count = count + 1

    meanSqErr = sumSqErr / count
    rmse = np.sqrt(meanSqErr)

    return rmse


# load data from file
rawData = pd.read_csv('/home/askapoor/Downloads/train_rating.txt', sep=",",
                      dtype={"train_id": np.int32, "user_id": np.int32, "business_id": np.int32, "rating": np.int8,
                             "date": str})

#Splitting the training data in to train and test set for cross validation
rawDataTrain, testData = train_test_split(rawData, test_size=0.1, random_state=42)
print(testData)
rawDataTrain.reset_index(drop=True,inplace=True)
# determine the dimension of the data
numOfRow = rawDataTrain.shape[0]
numOfCol = rawDataTrain.shape[1]

#Main rating sparse matrix
userId = rawDataTrain['user_id'].values
businessId = rawDataTrain['business_id'].values
rating = rawDataTrain['rating'].values
userRating = sparse.coo_matrix((rating, (userId, businessId)))

# calculate average rating throughout whole dataset.
mu = np.average(rating)

# determine average movie rating
# determine average user rating
(x, y, z) = sparse.find(userRating)
countUser = np.bincount(x)
#print(countUser)
CountBusiness = np.bincount(y)
#print(CountBusiness)
sumsUser = np.bincount(x, weights=z)
sumsBusiness = np.bincount(y, weights=z)
averageUserRating = sumsUser / countUser
averageBusinessRating = sumsBusiness / CountBusiness
#Initializing  biases depending upon the training values and this just for checking and for assessing the shape. These parameter will again be initialize and learned through Stochastic Gradient Descent.
bu = averageUserRating - mu
bi = averageBusinessRating - mu
# print(bi.shape,bu.shape)
# print('mu',mu)
# print(bu.shape)
# print(bi)
# # ==================== LEARNING: DETERMINE Qi, Pu, bi, bu (Initializing these parameters)  =====================
# declare initial Qi and Pu
N = 200  # common dimension of Qi, Pu
seedVal = np.sqrt(mu / N)
ni = bi.shape[0]
nu = bu.shape[0]
# Qi = np.array([[seedVal/N] * N] * ni)
# Pu = np.array([[seedVal/N] * N] * nu)
# Qi = np.random.rand(ni,N)
# Pu = np.random.rand(nu,N)
Qi = np.random.normal(scale=1/200,size=(ni,N))
Pu = np.random.normal(scale=1/200,size=(nu,N))
# bi = np.random.normal(scale=1/100,size=(145303,1))
# bu = np.random.normal(scale=1/100,size=(693209,1))
bi = np.zeros(145303,np.double)
bu = np.zeros(693209,np.double)
#print(Qi.shape)
# ------------  calculate Qi,Pu,bi and bu through iteration----------------
urM = sparse.coo_matrix(userRating)
lrate = .0035   # learning rate
lamb = .09
iteratioN = 50
rmseIte = [0] * (iteratioN + 1)
tol = .00001

for ite in range(iteratioN):
    for u, i, r in zip(urM.row, urM.col, urM.data):
        rat = np.dot(Qi[i], Pu[u])+bi[i]+bu[u]+mu
        if rat > 5:
            rat = 5
        elif rat < 1:
            rat = 1
        err = r - rat
        Qi[i] = Qi[i] + lrate * (err * Pu[u] - lamb * Qi[i])
        Pu[u] = Pu[u] + lrate * (err * Qi[i] - lamb * Pu[u])
        bi[i] = bi[i]+ lrate * (err - lamb * bi[i])
        bu[u] = bu[u] + lrate * (err - lamb * bu[u])


    # check RMSE for breaking condition
    rmseIte[ite + 1] = calculateRMSE()
    print('RMSE-' + str(ite + 1) + ' = ' + str(rmseIte[ite + 1]))
    if (np.abs(rmseIte[ite + 1] - rmseIte[ite]) < tol):
        break

# ==================== Validating results on the test data======================

print('in test')
test = testData.reset_index(drop=True)
test['prediction'] = 0
print(test)
for index, row in test.iterrows():
    user = test.loc[index, 'user_id']
    business = test.loc[index, 'business_id']
    ratingUI = findNewRating(user, business)
    test.loc[index, 'prediction'] = ratingUI

print("Outside")
# round and ceil if >5 or <1
test.loc[test.rating < 1, 'prediction'] = 1
test.loc[test.rating > 5, 'prediction'] = 5
print('checking')
rmse = mean_squared_error(test['rating'].values, test['prediction'].values)

print("-------------------Main RMSE-------------------",rmse**.5)