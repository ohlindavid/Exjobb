import numpy as np
import random
from scipy.signal import decimate
import matplotlib.pyplot as plt
from settings import imax, Fs0, L0, Lmin, Lmax, Fmin, Fmax

def signalLoader(nchan,Fs,L,files,labels,path,batch_size=1,data_aug=False,doDownsampling=False):
    Ln = len(files)
    c = list(zip(files,labels))
    np.random.shuffle(c)
    files, labels = zip(*c)
    labels = np.array(labels)

    while True:
        batch_start = 0
        batch_end = batch_size
        while batch_start < Ln:
            limit = min(batch_end, Ln)
            X = methodToLoad(files[batch_start:limit],path,nchan,Fs,L)
            #X = np.expand_dims(X,axis=0)
            Y = labels[batch_start:limit,:]
            yield (X,Y) #a tuple with two numpy arrays with batch_size samples
            batch_start += batch_size
            batch_end += batch_size

def methodToLoad(files,path,nchan,Fs,L):
    train_0 = []
    aim_folder_path = path
    for imID in files:
        # Loads flattened 4-D matrix.
        train_0.append(np.loadtxt(aim_folder_path+imID,delimiter=','))
        train_0 = np.vstack(train_0)
        # Converts to correct 4-D form.
        train_0 = np.reshape(train_0,[1,nchan,L0,Fs0])
        train_0 = np.transpose(train_0, [0,3,2,1])
        train_0 = train_0[:,Fmin:(Fmax+1),Lmin:(Lmax+1),:]
        #train_0 = train_0 * (train_0 < imax) + imax*(train_0 >= imax)
        #train_0 = train_0 * (train_0 >= 0)
        #train_0 = np.log2(train_0 + 1)
        #plt.matshow(train_0[0,:,:,0])
        #plt.show()
    return train_0
