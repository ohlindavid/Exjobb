import numpy as np

def signalLoader(files,labels,batch_size=1):
    L = len(files)
    while True:
        batch_start = 0
        batch_end = batch_size
        while batch_start < L:
            limit = min(batch_end, L)
            X = methodToLoad(files[batch_start:limit])
            Y = labels[batch_start:batch_end]
            Y = np.vstack(Y)
            yield (X,Y) #a tuple with two numpy arrays with batch_size samples
            batch_start += batch_size
            batch_end += batch_size

def methodToLoad(files):
    train_0 = []
    aim_folder_path = "C:/Users/david/Documents/GitHub/exjobb/sim/test_sim_1ch/"

    for imID in files:
        train_0.append(np.loadtxt(aim_folder_path+str(imID)))

    train_0 = np.vstack(train_0)

    return train_0
