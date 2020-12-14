import numpy as np

def signalLoader(nchan,files,labels,path,batch_size=1):
    L = len(files)
    labels= np.transpose(labels)
    while True:
        batch_start = 0
        batch_end = batch_size
        while batch_start < L:
            limit = min(batch_end, L)
            X = methodToLoad(files[batch_start:limit],path)
            Y = labels[batch_start:limit]
            Y = np.vstack(Y)
#            print(np.mean(X,axis=0))
            X = X - np.mean(X,axis=0)
            sum_of_rows = X.sum(axis=0)
            X = X / sum_of_rows[np.newaxis,:]
#            print(np.shape(sum_of_rows))
            #print(files[batch_start:limit],Y)
            yield (np.expand_dims(X,axis=0),Y) #a tuple with two numpy arrays with batch_size samples
            batch_start += batch_size
            batch_end += batch_size

def methodToLoad(files,path):
    train_0 = []
    aim_folder_path = path
    for imID in files:
        train_0.append(np.loadtxt(aim_folder_path+imID,delimiter=','))
        train_0 = np.vstack(train_0)
    return train_0
