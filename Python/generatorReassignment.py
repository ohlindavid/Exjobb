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
            X = np.expand_dims(X,axis=0)
            Y = labels[batch_start:limit,:]
            #print(np.shape(X))
            #print(Y)
            yield (X,Y) #a tuple with two numpy arrays with batch_size samples
            batch_start += batch_size
            batch_end += batch_size

def methodToLoad(files,path):
    train_0 = []
    aim_folder_path = path
    for imID in files:
        # Loads flattened 4-D matrix.
        train_0.append(np.loadtxt(aim_folder_path+imID,delimiter=','))
        train_0 = np.vstack(train_0)
        # Converts to correct 4-D form.
        train_0 = np.reshape(train_0,[6,5,64,321])
        train_0 = np.transpose(train_0,[1,2,3,0])
    return train_0
