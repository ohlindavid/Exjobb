import numpy as np
import random
from scipy.signal import decimate

def signalLoader(nchan,files,labels,path,batch_size=1,data_aug=False,doDownsampling=False):
    L = len(files)
    c = list(zip(files,labels))
    np.random.shuffle(c)
    files, labels = zip(*c)
    labels = np.array(labels)

    while True:
        batch_start = 0
        batch_end = batch_size
        while batch_start < L:
            limit = min(batch_end, L)
            X = methodToLoad(files[batch_start:limit],path)
            Y = labels[batch_start:limit,:]

    #       print("MEAN")
    #        X = X - np.mean(X,0)
    #        print(np.mean(X,0))
    #       print("VARIANCE")
    #        X = X/np.sqrt(np.mean(np.var(X,0)))*100
            #X = X/np.sqrt(np.var(X,0))
            #print(np.random.normal(scale=0.00000001 ,size=(np.shape(X))))
        #    X = X + np.random.normal(scale=0.0000001 ,size=(np.shape(X)))
#            X = augment_data(X,doDownsampling,data_aug)
        #    print(np.var(X,0))

            if (nchan > 1):
                yield (np.expand_dims(X,axis=0),Y) #a tuple with two numpy arrays with batch_size samples
            else:
                yield (X,Y)
            batch_start += batch_size
            batch_end += batch_size

def methodToLoad(files,path):
    train_0 = []
    aim_folder_path = path
    for imID in files:
        train_0.append(np.loadtxt(aim_folder_path+imID,delimiter=','))
        train_0 = np.vstack(train_0)
    return train_0

def augment_data(X,doDownsampling,data_aug):
    if doDownsampling:
        X = decimate(X,8,axis=0)
    if data_aug:
        #    if random.randint(0, 1) == 1:
        #        X = -X
        if random.randint(0, 1) >= 1:
            X = X + np.random.normal(scale=0.0001 ,size=(np.shape(X)))
    return X
