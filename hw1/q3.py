from colored import fore,back,style
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial
import fire

from jax import grad as autograd
import jax.numpy as jnp
from jax import random
from jax import jit
from jax import device_put
from jax import jit

def normalize(features, mean_feature=[]):
    # print("Shape of features:",features.shape)
    features = features.T
    N = features.shape[1]
    if not len(mean_feature): mean_feature = features.sum(axis=1)/N
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
    # jnp = np
    xs=device_put(xs)
    ys=device_put(ys)
    theta=device_put(theta)
    lambdaa = device_put(lambdaa)
        
    obj = -jnp.dot(xs @ theta, ys)

    e2tx = jnp.exp(xs@theta)
    logged = jnp.log(1 + e2tx)
    obj += logged.sum()

    regularisation = lambdaa * jnp.dot(theta[1:], theta[1:])
    obj += regularisation

    return obj

objective = jit(objective)


# define autogradient
autogradient = jit(autograd(objective, argnums=2))

def gradient(xs,ys,theta, lambdaa=0.01):
    # xs is a matrix of size nfeats x nsamples
    # ys is a vector of size nsamples
    
    grad = -xs.T @ ys

    e2tx = jnp.exp(xs @ theta)
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
    grad = gradient(xs,ys,theta,lambdaa=0.01)
    # agrad = autogradient(xs,ys,theta,0.01)
    # grad = agrad
    
    theta = theta - lr * grad
    return theta

def gdmomentum(xs, ys, theta, lr=0.01, eta=1):
    try:
        gdmomentum.lasttheta
    except:
        gdmomentum.lasttheta = theta

    grad = gradient(xs,ys,theta, lambdaa=0.01)
    # agrad = autogradient(xs,ys,theta,0.01)
    # grad = agrad
    # print(lr,eta)
    theta = theta - lr*grad + eta*(theta - gdmomentum.lasttheta)

    # print(theta - gdmomentum.lasttheta)

    # print(sum(theta),sum(gdmomentum.lasttheta))
    gdmomentum.lasttheta = theta

    return theta

def gdnesterov(xs, ys, theta, lr=0.01, eta=1):
    try:
        gdnesterov.lasttheta
    except:
        gdnesterov.lasttheta = theta

    grad = gradient(xs,ys,theta + eta*(theta - gdnesterov.lasttheta),lambdaa=0.01)
    # agrad = autogradient(xs,ys,theta + eta*(theta - gdnesterov.lasttheta),lambdaa=0.01)
    # grad = agrad
    # print(lr,eta)
    theta = theta - lr*grad + eta*(theta - gdnesterov.lasttheta)

    gdnesterov.lasttheta = theta

    return theta


# gdmomentum = jit(gdmomentum)
#### DESCENT METHODS ####


#### TRAINING HELPERS ####

def trainloop(xs,ys,descentfn, theta=[], niters=500):
    theta_history = []
    if len(theta)==0:
        theta = (np.random.rand(xs.shape[1]) - 0.5)*10
    
    for _ in range(niters):
        theta = descentfn(xs=xs,ys=ys,theta=theta)
        yshat = predict(xs,theta)
        # print(sum(theta))
        #print(loss(ys,yshat))
        theta_history.append(theta)
        
    return loss(yshat,ys), theta, theta_history

    
def mainloop(data, descentfn, ntrials=100, acctolerance=None):
    try:            
        print(f"\n{back.PURPLE_3} {style.BOLD} Running:",descentfn.__name__,style.RESET)
    except:
        print(f"\n{back.PURPLE_3} {style.BOLD} Running:",descentfn.func.__name__,"with",descentfn.keywords,style.RESET)


    trainlosses = []
    testlosses = []
    
    if not acctolerance:
        pbar = tqdm(range(ntrials))
        for run in pbar:
            train,test = test_train_partition(data)
            # print(train.shape, test.shape)

            tn_patientID, tt_patiendID = train[:,0], test[:,0]
            tn_labels, tt_labels = train[:,1], test[:,1]
            tn_features, tt_features = train[:,2:], test[:,2:]

            tn_normed_features,tn_mean_feature = normalize(tn_features)
            tn_aug_features = augment_features(tn_normed_features)

            # normalize test features
            tt_normed_features,mean_feature = normalize(tt_features,tn_mean_feature)
            tt_aug_features = augment_features(tt_normed_features)
            
            
            # grad = gradient(tn_aug_features, tn_labels, np.random.rand(31))
            trainloss,theta,theta_history = trainloop(tn_aug_features, tn_labels, descentfn)


            predtest = predict(tt_aug_features, theta)
            testloss = loss(predtest, tt_labels) 

            trainlosses.append(trainloss)
            testlosses.append(testloss)

            
        avgtrainloss = sum(trainlosses)/ntrials
        avgtestloss = sum(testlosses)/ntrials


        try:            
            print(descentfn.__name__, ": avgtrainloss : ", avgtrainloss)
            print(descentfn.__name__, ": avgtestloss : ", avgtestloss) 
            gdmethodname = descentfn.__name__        
        except:
            print(descentfn.func.__name__, ": avgtrainloss : ", avgtrainloss)
            print(descentfn.func.__name__, ": avgtestloss : ", avgtestloss) 
            gdmethodname = descentfn.func.__name__


        accuracies = [ accuracy(tt_aug_features, tt_labels, __theta) for __theta in theta_history ]
        accuracies = np.array(accuracies) 
        sns.lineplot(x=range(len(theta_history)), y=accuracies, label=gdmethodname)
             
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
            tt_normed_features, tn_mean_feature = normalize(tt_features)
            tt_aug_features = augment_features(tt_normed_features)
                    
            
            trainloss,theta,theta_history = trainloop(tn_aug_features, tn_labels, descentfn, niters=1)    
            acc = accuracy(tn_aug_features, tn_labels, theta)

            for it in range(100000):
                trainloss,theta,theta_history = trainloop(tn_aug_features, tn_labels, descentfn, theta=theta, niters=1)    
                acc = accuracy(tn_aug_features, tn_labels, theta)
                pbar.set_postfix({"itersreq":it,"acc":'%.6f' % acc})
                if acc < acctolerance: break
                        
            iterations.append(it)

        avgiters = sum(iterations)/ntrials
        print(f"avgiters:", avgiters)
        return avgiters
#### TRAINING HELPERS ####



def run(lr=0.01,eta=0.95,ntrials=10):
    data = pd.read_csv("wdbc.data", header=None).to_numpy()

    gd_setup = partial(gd, lr=lr)
    gdmomentum_setup = partial(gdmomentum, lr=lr,eta=eta)
    gdnesterov_setup = partial(gdnesterov, lr=lr,eta=eta)

    avgtrainloss, avgtestloss = mainloop(data,gd_setup,ntrials=ntrials)
    avgtrainloss, avgtestloss = mainloop(data,gdmomentum_setup,ntrials=ntrials)
    avgtrainloss, avgtestloss = mainloop(data,gdnesterov_setup,ntrials=ntrials)
    plt.xlabel("iterations")
    plt.ylabel("accuracy = grad(F)^2/(1+abs(F))")
    plt.title(f"lr={lr}, eta={eta}")
    plt.savefig("gdmethods.png")   

    
    # avgiterstaken = mainloop(data,gd_setup, ntrials=ntrials, acctolerance=1e-6)
    # avgiterstaken = mainloop(data,gdmomentum_setup, ntrials=ntrials, acctolerance=1e-6)
    # avgiterstaken = mainloop(data,gdnesterov_setup, ntrials=ntrials, acctolerance=1e-6)


    
if __name__ == "__main__":    
    # key = random.PRNGKey(1)
    # xs = random.normal(key,(10,2))
    # ys = random.normal(key,(10,))
    # theta = random.normal(key,(2,))
    # agrad = jit(autograd(objective, argnums=2))(xs,ys,theta, 0.01)
    # mygrad = gradient(xs,ys,theta,0.01)
    # print("autograd:", agrad)
    # print("mygrad:", mygrad)

    fire.Fire(run)
