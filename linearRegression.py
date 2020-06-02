import numpy as np
import projectLib as lib

# shape is movie,user,rating
training = lib.getTrainingData()
validation = lib.getValidationData()

#some useful stats
trStats = lib.getUsefulStats(training)
vlStats = lib.getUsefulStats(validation)
rBar = np.mean(trStats["ratings"])

# we get the A matrix from the training dataset
def getA(training):
    A = np.zeros((trStats["n_ratings"], trStats["n_movies"] + trStats["n_users"]))
    # ???
    for i in range(trStats["n_ratings"]):
        # put 1 for all movie rated
        A[i][trStats["movies"][i]] = 1

        # put 1 for all users
        A[i][trStats["n_movies"] + trStats["users"][i]] = 1
    return A

# # we also get c
def getc(rBar, ratings):
    # c is ratings - rBar
    c = ratings - rBar
    return c


# apply the functions
A = getA(training)
c = getc(rBar, trStats["ratings"])

# compute the estimator b
def param(A, c):
    # b = (A^{T}A)^{-1}A^{T}c
    # use pinv (pseduo inverse) just in case that it is not invertable.
    b = np.matmul(np.matmul(np.linalg.pinv(np.matmul(A.T,A)),A.T),c)
    return b

# compute the estimator b with a regularisation parameter l
# note: lambda is a Python keyword to define inline functions
#       so avoid using it as a variable name!
def param_reg(A, c, l):
    #  b = (A^{T}A + lI)^{-1}A^{T}c
    b = np.matmul(np.matmul(np.linalg.pinv(np.matmul(A.T,A)+l * np.identity(A.shape[1])),A.T),c)
    return b
    # inverse = np.linalg.pinv(A.transpose().dot(A) + l * np.identity(A.shape[1]))  # inv(ATA + lamda*I)
    # b = inverse.dot(A.transpose()).dot(c)
    # return b

# from b predict the ratings for the (movies, users) pair
def predict(movies, users, rBar, b):
    n_predict = len(users)
    p = np.zeros(n_predict)
    for i in range(0, n_predict):
        rating = rBar + b[movies[i]] + b[trStats["n_movies"] + users[i]]
        if rating > 5: rating = 5.0
        if rating < 1: rating = 1.0
        p[i] = rating
    return p

# Unregularised version (<=> regularised version with l = 0)
# b = param(A, c)

# UnRegularised version
l = 0
b = param_reg(A, c, l)

print("Linear regression, l = %f" % l)
print("RMSE for training %f" % lib.rmse(predict(trStats["movies"], trStats["users"], rBar, b), trStats["ratings"]))
print("RMSE for validation %f" % lib.rmse(predict(vlStats["movies"], vlStats["users"], rBar, b), vlStats["ratings"]))

# Finding best regularisation term
import pandas as pd
df = pd.DataFrame()
for i in np.linspace(0,5,30):
    b = param_reg(A,c,i)
    df = df.append([[lib.rmse(predict(trStats["movies"], trStats["users"], rBar, b), trStats["ratings"]),
               lib.rmse(predict(vlStats["movies"], vlStats["users"], rBar, b), vlStats["ratings"]),i]])
df = df.rename(columns={0: 'training_err', 1: 'validation_err', 2: 'regulation'})
df = df.reset_index(drop=True)

import matplotlib.pyplot as plt
# plotting
plt.plot(df['regulation'],df['training_err'],label="training")
plt.plot(df['regulation'],df['validation_err'],label="validation")
plt.xlabel("$\lambda$")
plt.ylabel("RMSE")
plt.legend()
plt.show()

best_lambda = df['regulation'][df['validation_err'].idxmin]
print("Best regularisation = %f" % best_lambda)
b = param_reg(A, c, best_lambda)

print("Linear regression, l = %f" % best_lambda)
print("RMSE for training %f" % lib.rmse(predict(trStats["movies"], trStats["users"], rBar, b), trStats["ratings"]))
print("RMSE for validation %f" % lib.rmse(predict(vlStats["movies"], vlStats["users"], rBar, b), vlStats["ratings"]))
