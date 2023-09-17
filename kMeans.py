#from skylearn.datasets import make_blobs
import matplotlib.pyplot as plt
from kMean import kMean
import pandas as pd

df = pd.read_csv('data1.csv')

X = df.iloc[:,:].values

km = kMean(n_clusters=3,max_iter=50)
y_means = km.fit_predict(X)

plt.scatter(X[y_means == 0,0],X[y_means == 0,1],color = 'red')
plt.scatter(X[y_means == 1,0],X[y_means == 1,1],color = 'blue')
plt.scatter(X[y_means == 2,0],X[y_means == 2,1],color = 'green')
#plt.scatter(X[y_means == 3,0],X[y_means == 3,1],color = 'yellow')
plt.show()
