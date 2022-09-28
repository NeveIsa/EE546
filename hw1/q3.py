import numpy as np
import pandas as pd

def normalize(features):
    # print("Shape of features:",features.shape)
    N = features.shape[1]
    mean_feature = features.sum(axis=1)/N
    centered_features = (features - np.outer(mean_feature,np.ones(N)))
    normed_features = centered_features @ np.diag(1/(np.diag(centered_features.T @ centered_features)**0.5))
    # print(np.diag(normed_features.T @ normed_features))
    return normed_features


def test_train_partition(features):
    N = features.shape[1]
    train_idx = np.random.choice(N,500, replace=False)
    test_idx =list(set(range(N)) - set(train_idx))
    train = features[:,train_idx]
    test = features[:,test_idx]
    return train,test

def predict(x,w,b):
    p = 1/(1 + np.exp(-np.dot(w,x)-b))
    label = 1 if p>0.5 else 0
    return label,p


if __name__ == "__main__":
    data = pd.read_csv("wdbc.data", header=None).to_numpy()
    
    patientID = data[:,0]
    labels = data[:,1]
    features = data[:,2:].T

    normed_features = normalize(features)
    train,test = test_train_partition(normed_features)
    # print(train.shape, test.shape)
    
      
    print(predict(np.random.rand(3),np.random.rand(3),1))    
