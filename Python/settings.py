#User
who = "Oskar"
def path():
    if who=="Oskar":
        return "C:/Users/Oskar/Documents/GitHub/exjobb/AD_comp_to_Bramao/"
    if who=="David":
        return "C:/Users/david/Documents/GitHub/exjobb/sim/test_sim_1ch/"
def pathPred():
    if who=="Oskar":
        return "C:/Users/Oskar/Documents/GitHub/exjobb/AD_comp_to_Bramao_Pred/"
    if who=="David":
        return "C:/Users/david/Documents/GitHub/exjobb/sim/test_sim_1ch/"

#Run
epochs = 75

#Morlet
etas = 25
filters = 25
wtime = 0.36
a_init = 10
b_init_min = 2
b_init_max = 30
train_a = True
train_b = True

#Pooling
poolSize = (1,71)
poolStrides = (1,15)
