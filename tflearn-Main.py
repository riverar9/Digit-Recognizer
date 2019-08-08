#%%
#Here I'm just importing a few things I know i'll need
import numpy as np
import tflearn
from tflearn.data_utils import load_csv

mac = True
#%%
#Depending on the OS i have to change how i get the data... is there a better way?
if mac:
    train_data, train_labels = load_csv(r"dl_data/train.csv", target_column=0,categorical_labels=True, n_classes=10)
    test_data, test_labels = load_csv(r"dl_data/test.csv")
else:
    train_df = load_csv(r"dl_data\train.csv")
    test_df = load_csv(r"dl_data\test.csv")

#%%
#Building a 3 layer NN with 784, 16, 10 nodes
net = tflearn.input_data(shape=[None, 784])
net = tflearn.fully_connected(net,64)
net = tflearn.fully_connected(net,16)
net = tflearn.fully_connected(net,10,activation='softmax')
net = tflearn.regression(net)

#%%
model = tflearn.DNN(net)
model.fit(train_data,train_labels,show_metric=True, batch_size=16)

#%%
