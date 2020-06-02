import numpy as np
import rbm
import projectLib as lib
import pickle

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
fileName = 'best_W_f_10'

try:
    # try to load existing trained weight, if any
    best_W = pickle.load(open(fileName, 'rb'))
except:
    # otherwise, train/tune 

    ## Parameter to tune
    # mrange = np.linspace(0.5,1,6) # momentum
    # lrange = np.logspace(0.0001,1,10) # lambda for regularisation
    # arange = np.logspace(0.0001,1,10) # alpha for learning rate
    # brange = np.linspace(5,55,11,dtype='int') # batch size , every 5
    # frange = np.linspace(1,21,11,dtype='int') # Hidden Units

    # tuned parameter as far
    mrange = [0.9]
    lrange = [0.001]
    arange = [0.01]
    brange = [10]
    frange = [10]

    # Initiate variables : Best parameters
    best_momentum = 0
    best_reg = 0
    best_epoch = 0
    best_alpha = 0
    best_B = 0
    best_F = 0

    # epsilonvb     = 0.01;   # Learning rate for biases of visible units
    # epsilonhb     = 0.01;   # Learning rate for biases of hidden units

    # arbitary large starting rmse
    min_rmse = 10

    for momentum in mrange:
        for _lambda in lrange:
            for alpha in arange:
                for B in brange:
                    for F in frange:
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
                            # in each epoch, we'll visit all users in a random order
                            visitingOrder = np.array(trStats["u_users"])
                            np.random.shuffle(visitingOrder)

                            # | Extension | Adaptive Learning Rate
                            adapativeLearningRate = alpha / epoch ** 2

                            # | Extension | Mini Batch
                            # numBatches = np.ceil(visitingOrder.shape[0]/B)
                            batches = np.split(visitingOrder,B)

                            for batch in batches:

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
                                    posHiddenProb = rbm.visibleToHiddenVec(v, weightsForUser,hidbias)
                                    # get positive gradient
                                    # note that we only update the movies that this user has seen!
                                    posprods[ratingsForUser[:, 0], :, :] = rbm.probProduct(v, posHiddenProb)

                                    # print(posprods.shape)
                                    poshidact = np.sum(posprods, axis=(0,2))
                                    posvisact = np.sum(v,axis=0)

                                    # print(v.shape)
                                    # print(poshidact)
                                    ### UNLEARNING ###
                                    # sample from hidden distribution
                                    sampledHidden = rbm.sample(posHiddenProb)
                                    # propagate back to get "negative data"
                                    negData = rbm.hiddenToVisible(sampledHidden, weightsForUser,visbias)
                                    # propagate negative data to hidden units
                                    negHiddenProb = rbm.visibleToHiddenVec(negData, weightsForUser,hidbias)
                                    # get negative gradient
                                    # note that we only update the movies that this user has seen!
                                    negprods[ratingsForUser[:, 0], :, :] = rbm.probProduct(negData, negHiddenProb)

                                    # print(negprods.shape)
                                    neghidact = np.sum(negprods, axis=(0,2))
                                    negvisact = np.sum(negData,axis=0)

                                    # print(neghidact)
                                    # we average over the number of users in the batch (if we use mini-batch)
                                    # grad = gradientLearningRate * (posprods - negprods)
                                    # | Extension | Regularisation and Adaptive Learning Rate
                                    grad += adapativeLearningRate * ((posprods - negprods) / trStats["n_users"] - _lambda * W)

                                    #W[ratingsForUser[:, 0], :, :] += grad[ratingsForUser[:, 0], :, :]
                                    
                                    # print(posvisact,negvisact)
                                    visbiasinc = momentum*visbiasinc + (alpha/trStats["n_users"])*(posvisact-negvisact)
                                    hidbiasinc = momentum*hidbiasinc + (alpha/trStats["n_users"])*(poshidact-neghidact)

                                    visbias = visbias + visbiasinc
                                    hidbias = hidbias + hidbiasinc

                                # | Extension | Momentum
                                W += grad + momentum * prevGrad

                                # print(visbias,hidbias)
                                # Print the current RMSE for training and validation sets
                                # this allows you to control for overfitting e.g
                                # We predict over the training set
                                tr_r_hat = rbm.predict(trStats["movies"], trStats["users"], W, training)
                                trRMSE = lib.rmse(trStats["ratings"], tr_r_hat)

                                # We predict over the validation set
                                vl_r_hat = rbm.predict(vlStats["movies"], vlStats["users"], W, training)
                                vlRMSE = lib.rmse(vlStats["ratings"], vl_r_hat)

                                # print("### EPOCH %d ###" % epoch)
                                # print("Training loss = %f" % trRMSE)
                                # print("Validation loss = %f" % vlRMSE)

                                # | Extension | Early Stopping
                                if vlRMSE < min_rmse:
                                    best_momentum = momentum
                                    best_reg = _lambda
                                    best_epoch = epoch
                                    best_alpha = alpha
                                    best_B = B
                                    best_F = F
                                    min_rmse = vlRMSE
                                    best_W = W
                                # print(visbias,hidbias)
                                print(vlRMSE,'And Best RMSE so far:', min_rmse)
    ### retrain with best parameter
    print('Best parameters:',[best_momentum, best_reg, best_alpha, best_B, best_F, best_epoch])
    # pickle 
    pickle.dump(best_W,open(fileName, 'wb'))

### END ###
# This part you can write on your own
# you could plot the evolution of the training and validation RMSEs for example
print('Predicting')

# log training and validating error
print('RSME on training',lib.rmse(trStats["ratings"],rbm.predict(trStats["movies"], trStats["users"], best_W, training)))
print('RSME on validation',lib.rmse(vlStats["ratings"],rbm.predict(vlStats["movies"], vlStats["users"], best_W, training)))


predictedRatings = np.array([rbm.predictForUser(user, best_W, training) for user in trStats["u_users"]])
print(np.max(predictedRatings),np.min(predictedRatings))
# Output the 300 x 97 txt
print('Saving new result')
np.savetxt("predictedRatings.txt", predictedRatings)
