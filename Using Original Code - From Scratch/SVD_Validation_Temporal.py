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


# def findNewRating(user, business):
#    return np.dot(Qi[business], Pu[user])

#This function is used for predicting the rating for the next iteration in order to check if the factor update is required using Stochastic Gradient Descent Technique and also for predicting the rating for the test data.
def findNewRating(user, business):
    try:
        rat = np.dot(Qi[business], Pu[user])+bi[business]+bu[user]+mu+Item[connect[user,business],business]
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

#For grouping the training data for Item Temporal Analysis
train_df = rawDataTrain
train_df2 = train_df
train_df2.loc[:,'date'] = pd.to_datetime(train_df2['date'])
train_df2= train_df2[['business_id','rating','date']]
train_df2.loc[:,'date'] = train_df2.loc[:,'date'].dt.year
train_df2 = train_df2.groupby(['business_id','date'],as_index = False).agg(['mean']).reset_index()
train_df2.columns = train_df2.columns.droplevel(1)


#ItemTemporal bias sparse matrix
businessId_t = train_df2['business_id'].values
date_t = train_df2['date'].values
train_df2.loc[:,'year_bias'] = 0
Itemyearbias = sparse.coo_matrix((train_df2.loc[:,'year_bias'].values, (date_t, businessId_t)))

train_df.loc[:,'date'] = pd.to_datetime(train_df.loc[:,'date'])
train_df.loc[:,'date'] = train_df.loc[:,'date'].dt.year
train_df = train_df[['user_id','business_id','date']]
print('train-----------',train_df)
print(train_df.shape)
print(rawDataTrain.shape)

#Sparse matrix for connecting year to ItemTemporal Bias matrix
userId_u = train_df['user_id'].values
businessId_u = train_df['business_id'].values
date_u = train_df['date'].values
connection = sparse.coo_matrix((date_u, (userId_u, businessId_u)))


#Main rating sparse matrix
userId = rawDataTrain['user_id'].values
businessId = rawDataTrain['business_id'].values
rating = rawDataTrain['rating'].values
userRating = sparse.coo_matrix((rating, (userId, businessId)))


# calculate average rating throughout whole dataset
mu = np.average(rating)

# determine average movie rating
# determine average user rating
(x, y, z) = sparse.find(userRating)
countUser = np.bincount(x)
print(countUser)
CountBusiness = np.bincount(y)
print(CountBusiness)
sumsUser = np.bincount(x, weights=z)
sumsBusiness = np.bincount(y, weights=z)
averageUserRating = sumsUser / countUser
averageBusinessRating = sumsBusiness / CountBusiness
#Initializing  biases depending upon the training values and this just for checking and for assessing the shape. These parameter will again be initialize and learned through Stochastic Gradient Descent.
bu = averageUserRating - mu
bi = averageBusinessRating - mu
print(bi.shape,bu.shape)
print('mu',mu)
print(bu.shape)
print(bi)
# ==================== LEARNING: DETERMINE Qi, Pu, bi, bu (Initializing these parameters) =====================
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
bi = np.zeros(145303,np.double)
bu = np.zeros(693209,np.double)
#print(Qi.shape)
# ------------  calculate Qi,Pu,bi, bu and item temporal bias through iteration ----------------
urM = sparse.coo_matrix(userRating)
lrate = .012  # learning rate
lamb = .07
iteratioN = 15
rmseIte = [0] * (iteratioN + 1)
tol = .00001
Item = Itemyearbias.tocsr()
connect = connection.tocsr()
for ite in range(iteratioN):
    for u, i, r in zip(urM.row, urM.col, urM.data):
        # print ("(%d, %d), %s" % (u, i, r))
        #print(u,Pu[u].shape)
        #print(i,bi[i].shape)
        #print(connect[u,i])
        rat = np.dot(Qi[i], Pu[u])+bi[i]+bu[u]+mu+Item[connect[u,i],i]
        #print(rat)
        if rat > 5:
            rat = 5
        elif rat < 1:
            rat = 1
        err = r - rat
        Qi[i] = Qi[i] + lrate * (err * Pu[u] - lamb * Qi[i])
        Pu[u] = Pu[u] + lrate * (err * Qi[i] - lamb * Pu[u])
        bi[i] = bi[i]+ lrate * (err - lamb * bi[i])
        bu[u] = bu[u] + lrate * (err - lamb * bu[u])
        Item[connect[u,i], i] = Item[connect[u,i],i] + lrate * (err - lamb * Item[connect[u,i],i])

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