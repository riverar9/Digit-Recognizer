#%%
import pandas as pd
import numpy as np
import sys

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

mac = True

#%%
if mac:
    train_df = pd.read_csv(r"dl_data/train.csv")
    test_df = pd.read_csv(r"dl_data/test.csv")
else:
    train_df = pd.read_csv(r"dl_data\train.csv")
    test_df = pd.read_csv(r"dl_data\test.csv")

#%%
Y = train_df['label']
X = train_df.drop(['label'], axis=1)
#%%
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=.2)

#%%
model = MLPClassifier(solver='adam',
    hidden_layer_sizes=(16,16),
    verbose=True,
    learning_rate='adaptive', 
    learning_rate_init=0.0005)

model.fit(xtrain,ytrain)

#%%
predictions = model.predict(xtest)
print("Accuracy: {}%".format(100*accuracy_score(predictions,ytest)))

#%%
result = model.predict(test_df)
result_df = (pd.Series(result, name='label')).to_frame()

#%%
if mac:
    result_df.to_csv(r"Submissions/submission.csv")
else:
    result_df.to_csv(r"Submissions\submission.csv")

#%%
