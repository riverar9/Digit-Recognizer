#%%
#Here I'm just importing a few things I know i'll need
import numpy as np
import tflearn
from tflearn.data_utils import load_csv
import pandas as pd

mac = True
#%%
#Creating a preprocessing function for the data preprocessing
def model_preprocessing(data):
    return [[int(float(j))/255 for j in i] for i in data]
#%%
#Depending on the OS i have to change how i get the data... is there a better way?
if mac:
    train_data, train_labels = load_csv(r"dl_data/train.csv", target_column=0,categorical_labels=True, n_classes=10)
    test_data = load_csv(r"dl_data/test.csv")
    for each in range(len(test_data[0])):
        test_data[0][each].append(test_data[1][each])
    test_data = test_data[0]

#%%
#Perfomring that preprocessing
train_data = model_preprocessing(train_data)
test_data = model_preprocessing(test_data)
#%%
#Building a 3 layer NN with 64, 16, 16, 10 nodes
net = tflearn.input_data(shape=[None, 784])
net = tflearn.fully_connected(net,64)
net = tflearn.fully_connected(net,16)
net = tflearn.fully_connected(net,16)
net = tflearn.fully_connected(net,10,activation='softmax')
net = tflearn.regression(net)

#%%
#now we'll train the model
model = tflearn.DNN(net)
model.fit(train_data,train_labels,show_metric=True, batch_size=16, validation_set=0.1)

#%%
#Here we'll throw our predictions into our input
output = model.predict_label(test_data)
submission = []
for each in range(len(output)):
    submission.append([int(each+1),int(output[each].argmax())])

predictions = pd.DataFrame.from_records(submission)
predictions.rename({0:'ImageId',1:'Label'},axis='columns',inplace=True)
predictions.set_index('ImageId', inplace=True)
predictions.to_csv(r"Submissions/tflearn-submission.csv")
#%%
