import numpy as np
from keras.datasets import mnist
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix

def gray2binary(data, threshold):
    for i in range(len(data)):
        data[i] = map(lambda x: 1 if x >= threshold else 0, data[i])
    return data

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x = x_train.reshape(len(x_train), -1)[:1000]
y = y_train[:1000]

m = np.mean(x)
x = gray2binary(x, m)

gmm = GaussianMixture(n_components=10, covariance_type='full').fit(x)
pred = gmm.predict(x)
print confusion_matrix(y, pred)
