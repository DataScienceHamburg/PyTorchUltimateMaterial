#%% my example is inspired by https://github.com/glenngara/tutorials/blob/master/elm/elm-tutorial.ipynb

#%% packages
import numpy as np
from scipy import linalg
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import seaborn as sns
from collections import Counter

#%% data prep
X, y = make_blobs(n_samples=10000, n_features=2, centers=5, random_state=1)

#%% train / test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# %% visualize data
sns.scatterplot(X_train[:, 0], X_train[:, 1], hue=y_train)
# %% Hyperparams
INPUT_SIZE = X_train.shape[1]
HIDDEN_SIZE = 1000

# %% 
input_weights = np.random.normal(size=[INPUT_SIZE,HIDDEN_SIZE])
biases = np.random.normal(size=[HIDDEN_SIZE])
# %% Functions
def relu(x):
    return np.maximum(x, 0, x)

def hidden_nodes(X):
    G = np.dot(X, input_weights)
    G = G + biases
    H = relu(G)
    return H

def predict(X):
    out = hidden_nodes(X)
    out = np.dot(out, beta)
    return out

# %% calculate output weights beta
# this are our model weights
beta = np.dot(linalg.pinv(hidden_nodes(X_train)), y_train)

# %% Validation
y_test_pred = predict(X_test)
correct = 0
total = X_test.shape[0]

for i in range(total):
    y_test_pred = np.round(y_test_pred[i], 0)
    y_test_true = y_test[i]
    correct += 1 if y_test_pred == y_test_true else 0
accuracy = correct/total
print(f'Accuracy for {HIDDEN_SIZE} hidden nodes: {accuracy}')
# %% Baseline Accuracy

cnt = Counter(y_test)
#  Baseline Accuracy
np.max(list(cnt.values())) / total
# %%
