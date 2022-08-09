import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml

X = np.read('image.npz'), ["labels"]
y = pd.read_csv("labels.csv"), ["labels"]
print(pd.Series(y).value_counts())
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q','R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
nclasses = len(classes)

samples_per_class = 5
figure = plt.figure(figSize=(nclasses*2, (1+samples_per_class*2)))

idx_cls = 0
for cls in classes:
    idxs = np.flatnonzero(y == cls)
    idxs = np.random.choice(idxs, samples_per_class, replece=False)
    i = 0
    for idx in idxs:
        plt_idx = i * nclasses + idx_cls + 1
        p = plt.subplot(samples_per_class, nclasses, plt_idx);
        p = sns.heatmap(np.reshape(X[idx], (22,30)), cmap=plt.cm.gray,
                xticklabels=False, yticklabels=False, cbar=False);
        p = plt.axis('off');
        i += 1
        idx_cls += 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])

p = plt.figure(figsize=(10,10));
p = sns.heatmap(cm, annot=True, fmt="d", cbar=False)

