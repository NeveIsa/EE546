
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def normalize(features, mean_feature=None):
    # print("Shape of features:",features.shape)
    features = features.T
    N = features.shape[1]
    if not mean_feature: mean_feature = features.sum(axis=1)/N
    centered_features = (features - np.outer(mean_feature,np.ones(N)))
    normed_features = centered_features @ np.diag(1/(np.diag(centered_features.T @ centered_features)**0.5))
    # print(np.diag(normed_features.T @ normed_features))
    return normed_features.T, mean_feature


####  FEATURES ####

def augment_features(feats):
    # print(feats[10,:])
    augfeats = np.ones((feats.shape[0],feats.shape[1]+1))
    augfeats[:,1:] = feats
    # print(augfeats[10,:])
    return augfeats

def test_train_partition(features):
    features = features.T
    N = features.shape[1]
    train_idx = np.random.choice(N,500, replace=False)
    test_idx =list(set(range(N)) - set(train_idx))
    train = features[:,train_idx]
    test = features[:,test_idx]
    return train.T,test.T

####  FEATURES ####


#### OBJECTIVE FUNCTION AND GRADS ####

def predict(xs,theta):
    p = 1/(1 + np.exp(-xs @ theta))
    predlabels =(p > 0.5)*1
    return predlabels

def loss(ys,yshat):
    N = ys.shape[0]
    error = np.abs(ys - yshat)
    error = sum(error)/N
    return error


def objective(xs,ys,theta, lambdaa=0.01):
    obj = -np.dot(xs @ theta, ys)

    e2tx = np.exp(xs@theta)
    logged = np.log(1 + e2tx)
    obj += sum(logged)

    regularisation = lambdaa * np.dot(theta[1:], theta[1:])
    obj += regularisation

    return obj
    

def gradient(xs,ys,theta, lambdaa=0.01):
    # xs is a matrix of size nfeats x nsamples
    # ys is a vector of size nsamples
    
    grad = -xs.T @ ys

    e2tx = np.exp(xs @ theta)
    weights = e2tx / (1 + e2tx)
    grad += xs.T @ weights

    A = np.eye(theta.shape[0])
    A[0,0] = 0 
    grad += 0.5 * lambdaa * (A + A.T) @ theta

    return grad

def accuracy(xs,ys,theta, lambdaa=0.01):
    grad = gradient(xs,ys,theta,lambdaa)
    obj = objective(xs,ys,theta,lambdaa)

    acc = np.dot(grad,grad)/ (1 + abs(obj))
    return acc


#### OBJECTIVE FUNCTION AND GRADS ####


#### DESCENT METHODS ####

def gd(xs,ys,theta,lr=0.01):
    grad = gradient(xs,ys,theta)
    theta = theta - lr * grad
    return theta 

#### DESCENT METHODS ####


#### TRAINING HELPERS ####

def trainloop(xs,ys,descentfn, lr, theta=[], niters=500):
    if len(theta)==0:
        theta = (np.random.rand(xs.shape[1]) - 0.5)*10
    
    for _ in range(niters):
        theta = descentfn(xs,ys,theta,lr=lr)
        yshat = predict(xs,theta)
        # print(sum(theta))
        #print(loss(ys,yshat))

    return loss(yshat,ys), theta

def mainloop(data, descentfn, lr, ntrials=100, acctolerance=None):
    trainlosses = []
    testlosses = []
    
    if not acctolerance:
        for run in tqdm(range(ntrials)):
            train,test = test_train_partition(data)
            # print(train.shape, test.shape)

            tn_patientID, tt_patiendID = train[:,0], test[:,0]
            tn_labels, tt_labels = train[:,1], test[:,1]
            tn_features, tt_features = train[:,2:], test[:,2:]

            tn_normed_features,mean_feature = normalize(tn_features)
            tn_aug_features = augment_features(tn_normed_features)

            # normalize test features
            tt_normed_features,mean_feature = normalize(tt_features)
            tt_aug_features = augment_features(tt_normed_features)
            
            
            # grad = gradient(tn_aug_features, tn_labels, np.random.rand(31))
            trainloss,theta = trainloop(tn_aug_features, tn_labels, descentfn, lr=lr)


            predtest = predict(tt_aug_features, theta)
            testloss = loss(predtest, tt_labels) 

            trainlosses.append(trainloss)
            testlosses.append(testloss)

            
        avgtrainloss = sum(trainlosses)/ntrials
        avgtestloss = sum(testlosses)/ntrials

        print("learning rate:",lr)        
        print(descentfn.__name__, ": avgtrainloss : ", avgtrainloss)
        print(descentfn.__name__, ": avgtestloss : ", avgtestloss) 
 
        return avgtrainloss, avgtestloss

    else:
        iterations = []
        
        pbar = tqdm(range(ntrials))
        for i in pbar:
            train,test = test_train_partition(data)
            # print(train.shape, test.shape)

            tn_patientID, tt_patiendID = train[:,0], test[:,0]
            tn_labels, tt_labels = train[:,1], test[:,1]
            tn_features, tt_features = train[:,2:], test[:,2:]

            tn_normed_features,mean_feature = normalize(tn_features)
            tn_aug_features = augment_features(tn_normed_features)

            # normalize test features
            tt_normed_features, mean_feature = normalize(tt_features)
            tt_aug_features = augment_features(tt_normed_features)
                    
            
            trainloss,theta = trainloop(tn_aug_features, tn_labels, descentfn, lr=lr, niters=1)    
            acc = accuracy(tn_aug_features, tn_labels, theta)

            for it in range(100000):
                trainloss,theta = trainloop(tn_aug_features, tn_labels, descentfn, lr=lr, theta=theta, niters=1)    
                acc = accuracy(tn_aug_features, tn_labels, theta)
                if acc < acctolerance: break        

            iterations.append(it)
            pbar.set_postfix({"itersreq":it})

        avgiters = sum(iterations)/ntrials
        print(f"avgiters:", avgiters)
        return avgiters
#### TRAINING HELPERS ####

    
if __name__ == "__main__":
    data = pd.read_csv("wdbc.data", header=None).to_numpy()
    avgtrainloss, avgtestloss = mainloop(data,gd, lr=0.0075)
    avgiterstaken = mainloop(data,gd, lr=0.0075, ntrials=100, acctolerance=1e-6)
