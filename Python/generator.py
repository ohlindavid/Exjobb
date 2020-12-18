import numpy as np

def signalLoader(nchan,files,labels,path,batch_size=1):
    L = len(files)
    while True:
        batch_start = 0
        batch_end = batch_size
        while batch_start < L:
            limit = min(batch_end, L)
            X = methodToLoad(files[batch_start:limit],path)
            Y = labels[batch_start:limit,:]
            #Y = np.vstack(Y)
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
