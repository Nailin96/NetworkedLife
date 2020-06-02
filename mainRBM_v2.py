import numpy as np
import rbm
import projectLib as lib
import pickle
import pandas as pd

training = lib.getTrainingData()
validation = lib.getValidationData()
# You could also try with the chapter 4 data
# training = lib.getChapter4Data()

trStats = lib.getUsefulStats(training)
vlStats = lib.getUsefulStats(validation)

K = 5

# SET PARAMETERS HERE!!!
# number of hidden units
# F = 10
epochs = 30
# gradientLearningRate = 0.1

# file for past trained data
fileName = 'best_W_final'

try:
    # try to load existing trained weight, if any
    best_W = pickle.load(open(fileName, 'rb'))
except:
    # otherwise, train/tune 

    # tuned parameter as far
    mrange = [0.9]
    lrange = [0.001]
    arange = [0.01]
    brange = [10]
    frange = [10]

    # mrange = [0.5,0.7,0.9]
    # lrange = [0.1,0.01,0.001]
    # arange = [0.1,0.01,0.001]
    # brange = [5,10]
    # frange = [6,8,10]


    df = pd.DataFrame()

    for momentum in mrange:
        print("Momentum:",momentum)
        for _lambda in lrange:
            print("Regularisation:",_lambda)
            for alpha in arange:
                print("Learning rate:",alpha)
                for B in brange:
                    print("Number of Batchs:",B)
                    for F in frange:
                        print("Number of Hidden variables:",F)

                        # arbitary large starting rmse
                        min_rmse = 10

                        # Initialise all our arrays
                        W = rbm.getInitialWeights(trStats["n_movies"], F, K)
                        posprods = np.zeros(W.shape)
                        negprods = np.zeros(W.shape)

                        # | Extension | Bias
                        ## Initialise 

                        hidbias = np.zeros(W.shape[1])  # bias for each hidden node
                        visbias = np.zeros([W.shape[0],K]) # bias for each visible rating

                        hidbiasinc = np.zeros(W.shape[1])
                        visbiasinc = np.zeros([W.shape[0],K])

                        grad = np.zeros(W.shape) # gradient tracker for learning



                        for epoch in range(1, epochs+1):
                            print("EPOCH:",epoch)

                            # in each epoch, we'll visit all users in the batch in a random order
                            # print(batch)
                            visitingOrder = np.array(trStats["u_users"])
                            np.random.shuffle(visitingOrder)
                            batches = np.split(visitingOrder,B)

                            for i, batch in enumerate(batches):
                                print("mini-batch: ",i+1,"/",B)

                                np.random.shuffle(batch)

                                # | Extension | Adaptive Learning Rate
                                adapativeLearningRate = alpha / epoch ** 2

                                # | Extension | Mini Batch
                                # numBatches = np.ceil(visitingOrder.shape[0]/B)


                                prevGrad = grad
                                grad = np.zeros(W.shape)

                                for user in batch:
                                    # get the ratings of that user
                                    ratingsForUser = lib.getRatingsForUser(user, training)

                                    # build the visible input
                                    v = rbm.getV(ratingsForUser)

                                    # get the weights associated to movies the user has seen
                                    weightsForUser = W[ratingsForUser[:, 0], :, :]


                                    ### LEARNING ###
                                    # propagate visible input to hidden units
                                    # posHiddenProb = rbm.visibleToHiddenVec(v, weightsForUser)
                                    posHiddenProb = rbm.visibleToHiddenVec(v, weightsForUser,hidbias)
                                    # get positive gradient
                                    # note that we only update the movies that this user has seen!
                                    posprods[ratingsForUser[:, 0], :, :] += rbm.probProduct(v, posHiddenProb)

                                    poshidact = np.sum(posprods, axis=(0,2))
                                    posvisact = np.sum(v,axis=0)

                                    ### UNLEARNING ###
                                    # sample from hidden distribution
                                    sampledHidden = rbm.sample(posHiddenProb)

                                    # propagate back to get "negative data"
                                    # negData = rbm.hiddenToVisible(sampledHidden, weightsForUser)
                                    negData = rbm.hiddenToVisible(sampledHidden, weightsForUser,visbias)

                                    # propagate negative data to hidden units
                                    # negHiddenProb = rbm.visibleToHiddenVec(negData, weightsForUser)
                                    negHiddenProb = rbm.visibleToHiddenVec(negData, weightsForUser,hidbias)

                                    # get negative gradient
                                    # note that we only update the movies that this user has seen!
                                    negprods[ratingsForUser[:, 0], :, :] += rbm.probProduct(negData, negHiddenProb)

                                    # print(negprods.shape)
                                    neghidact = np.sum(negprods, axis=(0,2))
                                    negvisact = np.sum(negData,axis=0)

                                    # we average over the number of users in the batch (if we use mini-batch)
                                    # grad = gradientLearningRate * (posprods - negprods)
                                    # | Extension | Regularisation and Adaptive Learning Rate

                                    grad += adapativeLearningRate * ((posprods - negprods) / trStats["n_users"] - _lambda * W)
                                    
                                    # | Extension | Bias
                                    # we update the bias for visible and hidden variables here
                                    visbiasinc = momentum*visbiasinc + (alpha/trStats["n_users"])*(posvisact-negvisact)
                                    hidbiasinc = momentum*hidbiasinc + (alpha/trStats["n_users"])*(poshidact-neghidact)

                                    visbias = visbias + visbiasinc
                                    hidbias = hidbias + hidbiasinc

                                # | Extension | Momentum
                                W += grad + momentum * prevGrad

                                # Print the current RMSE for training and validation sets
                                # this allows you to control for overfitting e.g
                                # We predict over the training set
                                tr_r_hat = rbm.predict(trStats["movies"], trStats["users"], W, training,predictType="mean")
                                trRMSE = lib.rmse(trStats["ratings"], tr_r_hat)

                                # We predict over the validation set
                                vl_r_hat = rbm.predict(vlStats["movies"], vlStats["users"], W, training,predictType="mean")
                                vlRMSE = lib.rmse(vlStats["ratings"], vl_r_hat)

                                # print("### EPOCH %d ###" % epoch)
                                print("Training loss = %f" % trRMSE)
                                # print("Validation loss = %f" % vlRMSE)

                                # | Extension | Early Stopping
                                if vlRMSE < min_rmse:

                                    min_rmse = vlRMSE
                                    best_W = W

                                print('Validation loss =',vlRMSE,'And Best RMSE so far:', min_rmse)
                        # During Hyper parameter tuning, uncomment the following lines
                        # df = df.append([[momentum,_lambda,alpha,B,F,min_rmse]])
                        # print(df)

    # We dumped the important files
    # pickle.dump(df,open('df_important','wb')) #Only during hyper-parameter tuning
    pickle.dump(best_W,open(fileName, 'wb'))

### END ###
# This part you can write on your own
# you could plot the evolution of the training and validation RMSEs for example
print('Predicting')

# log training and validating error
print('RSME on training',lib.rmse(trStats["ratings"],rbm.predict(trStats["movies"], trStats["users"], best_W, training,predictType="mean")))
print('RSME on validation',lib.rmse(vlStats["ratings"],rbm.predict(vlStats["movies"], vlStats["users"], best_W, training,predictType="mean")))


predictedRatings = np.array([rbm.predictForUser(user, best_W, training,predictType="mean") for user in trStats["u_users"]])
print(np.max(predictedRatings),np.min(predictedRatings))
# Output the 300 x 97 txt
print('Saving new result')
np.savetxt("SUccess+v4.txt", predictedRatings)
