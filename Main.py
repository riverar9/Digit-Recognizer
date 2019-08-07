#%%
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

#%%
train_df = pd.read_csv(r"dl_data\train.csv")
test_df = pd.read_csv(r"dl_data\test.csv")

#%%
Y = train_df['label']
X = train_df.drop(['label'], axis=1)
#%%
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=.2, random_state=42)

#%%
model = MLPClassifier(solver='adam', hidden_layer_sizes=(16,16))

model.fit(xtrain,ytrain)

#%%
predictions = model.predict(xtest)

#%%


#%%
